import numpy as np
import torch


class EarlyStopping:
    """
    Class for early stopping during model training.
    Args:
        filename (str): The filename to save the trained model.
        patience (int, optional): The number of epochs to wait for improvement before stopping. Defaults to 7.
        verbose (bool, optional): Whether to print the early stopping counter. Defaults to True.
        delta (float, optional): The minimum change in the monitored criteria score to be considered as improvement. Defaults to 0.
        criteria (str, optional): The criteria to monitor for improvement. Defaults to "val_loss".
        bigger_better (bool, optional): Whether a higher criteria score indicates better performance. Defaults to False.
    Methods:
        __call__(criteria_score, model):
            Checks if the criteria score has improved and updates the early stopping counter.
        save_checkpoint(model, delta_tolerated=False, prev_best=np.inf):
            Saves the model when the monitored criteria score decreases.
    """

    def __init__(
        self,
        filename,
        patience=7,
        verbose=True,
        delta=0,
        criteria="val_loss",
        bigger_better=False,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.cur_date = filename
        self.criteria = criteria
        self.bigger_better = bigger_better

    def __call__(self, criteria_score, model):
        if self.best_score is None:
            self.best_score = criteria_score
            self.save_checkpoint(model)
        elif criteria_score - self.best_score <= -self.delta and self.bigger_better:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.best_score - criteria_score <= -self.delta and not self.bigger_better:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            prev_best = self.best_score
            is_delta_tolerated = (
                criteria_score < self.best_score
                if self.bigger_better
                else criteria_score > self.best_score
            )
            self.best_score = self.best_score if is_delta_tolerated else criteria_score
            self.save_checkpoint(
                model,
                delta_tolerated=is_delta_tolerated,
                prev_best=prev_best,
            )
            self.counter = 0

    def save_checkpoint(self, model, delta_tolerated=False, prev_best=np.inf):
        """Saves model when validation loss decrease."""
        status = "increased" if self.bigger_better else "decreased"
        if self.verbose:
            if delta_tolerated:
                print("Delta tolerated. Saving model ...")
            else:
                print(
                    f"{self.criteria} {status} ({prev_best:.6f} --> {self.best_score:.6f}).  Saving model ..."
                )
        torch.save(model.state_dict(), "../models/" + self.cur_date + "_checkpoint.pt")
