from typing import Callable, Iterable, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer

from rocket_learn.utils.util import transform_batch


def accuracy(y_pred, y_true):
    if y_pred.ndim == y_true.ndim:
        assert y_pred.ndim == 2
        assert y_pred.shape == y_true.shape
        # Assume y_true contains target probabilities rather than indices
        # Labels might be smoothed (more than one right answer)
        # We just want to check if highest prediction is one of the right answers
        is_true_max = (y_true == y_true.max(dim=-1, keepdim=True).values)

        is_pred_correct = is_true_max[range(len(y_true)), y_pred.argmax(dim=-1)]

        return is_pred_correct.float().mean()
    else:
        y_pred = y_pred.argmax(dim=-1)
        return (y_pred == y_true).float().mean()


class BehavioralCloning:
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            train_dataset: Dataset,
            validation_dataset: Dataset,
            batch_size: int,
            checkpoint_folder: str,
            save_interval: int,
            validation_interval: int,
            loss_fn: Callable = None,
            metrics: Dict[str, Callable] = None,
            device: str = "cuda",
            logger=None,  # Assumed wandb logger
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        self.metrics = {"accuracy": accuracy} if metrics is None else metrics
        self.device = device
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
        self.checkpoint_folder = checkpoint_folder
        self.save_interval = save_interval
        self.validation_interval = validation_interval
        self.logger = logger

        if logger is not None:
            self.logger.watch(self.model)

        self.train_samples = 0
        self.epochs = 0
        self.index = 0

    @classmethod
    def load(cls, save_path: str):
        checkpoint = torch.load(save_path)

        bc = cls(**checkpoint)

        bc.train_samples = checkpoint["train_samples"]
        bc.epochs = checkpoint["epochs"]
        bc.index = checkpoint["index"]

        return bc

    def save(self, folder: str, name: str):
        checkpoint = {
            "model": self.model,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            "device": self.device,
            "train_dataset": self.train_dataset,
            "validation_dataset": self.validation_dataset,
            "batch_size": self.batch_size,
            "checkpoint_folder": self.checkpoint_folder,
            "save_interval": self.save_interval,
            "validation_interval": self.validation_interval,
            "logger": self.logger,
            "train_samples": self.train_samples,
            "epochs": self.epochs,
            "index": self.index,
        }
        torch.save(checkpoint, f"{folder}/{name}")

    def validate(self):
        self.model.eval()
        val_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
        )
        with torch.no_grad():
            for features, labels in val_loader:
                features = transform_batch(features, lambda x: x.to(self.device))
                labels = transform_batch(labels, lambda x: x.to(self.device))

                y_pred = self.model(features)
                loss = self.loss_fn(y_pred, labels)
                metrics = {
                    name: metric(y_pred, labels)
                    for name, metric in self.metrics.items()
                }
                if self.logger is not None:
                    self.logger.log({"val_loss": loss, **metrics}, commit=False)

        self.model.train()

    def train(self):
        self.model.train()

        while True:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
            for features, labels in train_loader:
                features = transform_batch(features, lambda x: x.to(self.device))
                labels = transform_batch(labels, lambda x: x.to(self.device))

                self.optimizer.zero_grad()

                y_pred = self.model(features)
                loss = self.loss_fn(y_pred, labels)
                loss.backward()
                self.optimizer.step()

                metrics = {
                    name: metric(y_pred, labels)
                    for name, metric in self.metrics.items()
                }

                self.index += 1
                self.train_samples += self.train_dataset.batch_size

                if self.index % self.validation_interval == 0:
                    self.validate()

                if self.logger is not None:
                    # Will also commit the validation metrics
                    self.logger.log({"train_loss": loss, **metrics})

                if self.index % self.save_interval == 0:
                    self.save(self.checkpoint_folder, f"model_{self.epochs}_{self.index}")

            self.epochs += 1
            self.validate()
