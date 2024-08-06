from fastprogress.fastprogress import format_time, progress_bar
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    multilabel_confusion_matrix,
)
from torchmetrics.classification import MultilabelMatthewsCorrCoef
import numpy as np
import torch
import time
import torch.cuda


class EvaluateOnTest(object):
    """
    Class to encapsulate evaluation on the test set. Based off the "Tonks Library"
    :param model: PyTorch model to use with the Learner
    :param test_data_loader: dataloader for all of the validation data
    :param model_path: path of the trained model
    """

    def __init__(
        self, model, test_data_loader, model_path, col_names=[], run_name=None
    ):
        self.model = model
        self.test_data_loader = test_data_loader
        self.model_path = model_path
        self.col_names = col_names
        if run_name is not None:
            self.run_name = run_name
        else:
            self.run_name = self.model.name

    def predict(self, device="cuda:0", pbar=None):
        """
        Evaluate the model on a validation set
        :param device: str (defaults to 'cuda:0')
        :param pbar: fast_progress progress bar (defaults to None)
        :returns: None
        """
        self.model.to(device).load_state_dict(torch.load(self.model_path))
        self.model.eval()
        current_size = len(self.test_data_loader.dataset)
        preds_dict = {
            "y_true": np.zeros([current_size, len(self.col_names)]),
            "y_pred": np.zeros([current_size, len(self.col_names)]),
        }
        start_time = time.time()
        with torch.no_grad():
            index_dict = 0
            for step, batch in enumerate(
                progress_bar(
                    self.test_data_loader, parent=pbar, leave=(pbar is not None)
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
                    target_input_ids=inputs,
                    target_attention_masks=attention_masks,
                    lengths=lengths,
                    label_idxs=label_idxs,
                )
                targets = targets.cpu().numpy()
                current_index = index_dict
                preds_dict["y_true"][
                    current_index : current_index + num_rows, :
                ] = targets
                preds_dict["y_pred"][
                    current_index : current_index + num_rows, :
                ] = y_pred
                index_dict += num_rows
                torch.cuda.empty_cache()

        y_true, y_pred = preds_dict["y_true"], preds_dict["y_pred"]
        # save the predictions to a file
        np.save("y_pred_" + self.run_name + ".npy", y_pred)
        # save the ground truth to a file
        np.save("y_true_" + self.run_name + ".npy", y_true)
        str_stats = []
        stats = {
            "f1-macro": f1_score(y_true, y_pred, average="macro", zero_division=1),
            "f1-micro": f1_score(y_true, y_pred, average="micro", zero_division=1),
            "precision-macro": precision_score(
                y_true, y_pred, average="macro", zero_division=1
            ),
            "recall-macro": recall_score(
                y_true, y_pred, average="macro", zero_division=1
            ),
            "js-macro": jaccard_score(y_true, y_pred, average="macro", zero_division=1),
            "js-samples": jaccard_score(
                y_true, y_pred, average="samples", zero_division=1
            ),
            "mcc": MultilabelMatthewsCorrCoef(num_labels=len(self.col_names))(
                torch.tensor(y_true).int(), torch.tensor(y_pred).int()
            ).item(),
        }

        for stat in stats.values():
            str_stats.append(
                "NA"
                if stat is None
                else str(stat) if isinstance(stat, int) else f"{stat:.4f}"
            )
        str_stats.append(format_time(time.time() - start_time))
        headers = [
            "F1-Macro",
            "F1-Micro",
            "Precision Macro",
            "Recall Macro",
            "JS-Macro",
            "JS-Samples",
            "MCC",
            "Time",
        ]
        print(" ".join("{}: {}".format(*k) for k in zip(headers, str_stats)))
        # print confusion matrix
        print("Confusion Matrix:")
        confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
        # save
        np.save("confusion_matrix_" + self.run_name + ".npy", confusion_matrix)
        print(confusion_matrix)
        return stats
