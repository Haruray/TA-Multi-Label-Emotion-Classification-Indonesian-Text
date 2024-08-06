from fastprogress.fastprogress import format_time, master_bar, progress_bar
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
import torch.nn.functional as F
import numpy as np
import torch
import torch.cuda
from torchmetrics.classification import MultilabelMatthewsCorrCoef
from MLEC.loss.lca_loss import lca_loss
import time
from MLEC.trainer.EarlyStopping import EarlyStopping
from MLEC.loss.zlpr_loss import zlpr_loss
import wandb


class Trainer(object):
    """
    Class to encapsulate training and validation steps for a pipeline. Based off the "Tonks Library"
    :param model: PyTorch model to use with the Learner
    :param train_data_loader: dataloader for all of the training data
    :param val_data_loader: dataloader for all of the validation data
    :param filename: the best model will be saved using this given name (str)
    """

    def __init__(
        self,
        model,
        train_data_loader,
        val_data_loader,
        filename,
        early_stopping: EarlyStopping,
        col_names=[],
    ):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.filename = filename
        self.early_stop = early_stopping
        self.col_names = col_names

    def fit(
        self,
        classifier_learning_rate,
        encoder_learning_rate,
        train_batch_size,
        max_epoch,
        project_name,
        run_name,
        device="cuda:0",
    ):
        """
        Fit the PyTorch model
        :param num_epochs: number of epochs to train (int)
        :param args:
        :param device: str (defaults to 'cuda:0')
        """
        # init wandb
        wandb_config = {
            "encoder_learning_rate": encoder_learning_rate,
            "classifier_learning_rate": classifier_learning_rate,
            "train_batch_size": train_batch_size,
            "max_epoch": max_epoch,
            "alpha": self.model.alpha,
            "beta": self.model.beta,
        }
        run = wandb.init(project=project_name, config=wandb_config, name=run_name)

        optimizer, scheduler, step_scheduler_on_batch = self.optimizer(
            encoder_learning_rate, classifier_learning_rate, train_batch_size, max_epoch
        )
        self.model = self.model.to(device)
        pbar = master_bar(range(max_epoch))
        headers = [
            "Epoch",
            "Train_Loss",
            "Val_Loss",
            "F1-Macro",
            "F1-Micro",
            "Precision Macro",
            "Recall Macro",
            "JS-Macro",
            "JS-Samples",
            "MCC",
            "Time",
        ]
        pbar.write(headers, table=True)
        for epoch in pbar:
            epoch += 1
            start_time = time.time()
            self.model.train()
            overall_training_loss = 0.0
            overall_lca_loss = 0.0
            overall_zlpr_loss = 0.0
            for _, batch in enumerate(
                progress_bar(self.train_data_loader, parent=pbar)
            ):
                optimizer.zero_grad()
                (
                    inputs,
                    attention_masks,
                    targets,
                    lengths,
                    label_idxs,
                ) = batch

                num_rows, _, logits, targets = self.model(
                    input_ids=inputs,
                    input_attention_masks=attention_masks,
                    targets=targets,
                    lengths=lengths,
                    label_idxs=label_idxs,
                )
                lca_loss_total = lca_loss(logits, targets)
                zlpr_loss_total = zlpr_loss(logits, targets)

                bce_loss = F.binary_cross_entropy_with_logits(logits, targets).to(
                    device
                )
                targets = targets.cpu().numpy()
                total_loss = (
                    bce_loss * (1 - (self.model.alpha + self.model.beta))
                    + (lca_loss_total * self.model.alpha)
                    + (zlpr_loss_total * self.model.beta)
                )
                overall_training_loss += total_loss.item() * num_rows
                overall_lca_loss += lca_loss_total.item() * num_rows
                overall_zlpr_loss += zlpr_loss_total.item() * num_rows

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                if step_scheduler_on_batch:
                    scheduler.step()

                # Free unused GPU memory
                torch.cuda.empty_cache()

            if not step_scheduler_on_batch:
                scheduler.step()

            overall_training_loss = overall_training_loss / len(
                self.train_data_loader.dataset
            )

            overall_val_loss, pred_dict = self.predict(device, pbar)
            y_true, y_pred = pred_dict["y_true"], pred_dict["y_pred"]

            str_stats = []
            stats = {
                "epoch": epoch,
                "train_loss": overall_training_loss,
                "val_loss": overall_val_loss,
                "f1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=1),
                "f1_Micro": f1_score(y_true, y_pred, average="micro", zero_division=1),
                "precision_macro": precision_score(
                    y_true, y_pred, average="macro", zero_division=1
                ),
                "recall_macro": recall_score(
                    y_true, y_pred, average="macro", zero_division=1
                ),
                "js_Macro": jaccard_score(
                    y_true, y_pred, average="macro", zero_division=1
                ),
                "js_Samples": jaccard_score(
                    y_true, y_pred, average="samples", zero_division=1
                ),
                "mcc": MultilabelMatthewsCorrCoef(num_labels=len(self.col_names))(
                    torch.tensor(y_pred).int(), torch.tensor(y_true).int()
                ).item(),
            }

            for stat in stats.values():
                str_stats.append(
                    "NA"
                    if stat is None
                    else str(stat) if isinstance(stat, int) else f"{stat:.4f}"
                )
            str_stats.append(format_time(time.time() - start_time))
            print("epoch#: ", epoch)
            pbar.write(str_stats, table=True)
            # log stats to wandb
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": overall_training_loss,
                    "val_loss": overall_val_loss,
                    "f1_Macro": stats["f1_Macro"],
                    "f1_Micro": stats["f1_Micro"],
                    "precision_macro": stats["precision_macro"],
                    "recall_macro": stats["recall_macro"],
                    "js_Macro": stats["js_Macro"],
                    "js_Samples": stats["js_Samples"],
                    "mcc": stats["mcc"],
                }
            )

            self.early_stop(stats[self.early_stop.criteria], self.model)
            if self.early_stop.early_stop:
                print("Early stopping")
                break

    def optimizer(
        self,
        encoder_learning_rate,
        classifier_learning_rate,
        train_batch_size,
        max_epoch,
    ):
        """

        :param args: object
        """
        optimizer = AdamW(
            [
                {"params": self.model.encoder.parameters()},
                {
                    "params": self.model.classifier.parameters(),
                    "lr": classifier_learning_rate,
                },
            ],
            lr=encoder_learning_rate,
            correct_bias=True,
        )
        num_train_steps = (
            int(len(self.train_data_loader.dataset)) / train_batch_size
        ) * max_epoch
        num_warmup_steps = int(num_train_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
        step_scheduler_on_batch = True
        return optimizer, scheduler, step_scheduler_on_batch

    def predict(self, device="cuda:0", pbar=None):
        """
        Evaluate the model on a validation set
        :param device: str (defaults to 'cuda:0')
        :param pbar: fast_progress progress bar (defaults to None)
        :returns: overall_val_loss (float), accuracies (dict{'acc': value}, preds (dict)
        """
        current_size = len(self.val_data_loader.dataset)
        # print("len col names: ", len(self.col_names))
        preds_dict = {
            "y_true": np.zeros([current_size, len(self.col_names)]),
            "y_pred": np.zeros([current_size, len(self.col_names)]),
        }
        overall_val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            index_dict = 0
            for _, batch in enumerate(
                progress_bar(
                    self.val_data_loader, parent=pbar, leave=(pbar is not None)
                )
            ):
                (
                    inputs,
                    attention_masks,
                    targets,
                    lengths,
                    label_idxs,
                ) = batch

                num_rows, y_pred, logits, targets = self.model(
                    input_ids=inputs,
                    input_attention_masks=attention_masks,
                    targets=targets,
                    lengths=lengths,
                    label_idxs=label_idxs,
                )
                inter_corr_loss_total = lca_loss(logits, targets)
                intra_corr_loss_total = zlpr_loss(logits, targets)
                bce_loss = F.binary_cross_entropy_with_logits(logits, targets).to(
                    device
                )
                targets = targets.cpu().numpy()
                total_loss = (
                    bce_loss * (1 - (self.model.alpha + self.model.beta))
                    + (inter_corr_loss_total * self.model.alpha)
                    + (intra_corr_loss_total * self.model.beta)
                )
                overall_val_loss += total_loss.item() * num_rows

                current_index = index_dict
                preds_dict["y_true"][
                    current_index : current_index + num_rows, :
                ] = targets
                preds_dict["y_pred"][
                    current_index : current_index + num_rows, :
                ] = y_pred
                index_dict += num_rows
                # Free unused GPU memory
                torch.cuda.empty_cache()

        overall_val_loss = overall_val_loss / len(self.val_data_loader.dataset)
        return overall_val_loss, preds_dict
