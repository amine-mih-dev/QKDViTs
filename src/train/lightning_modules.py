from __future__ import annotations

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from vit_pytorch.distill import DistillWrapper

from .model_factory import adapt_classifier_if_needed, get_model_by_name


def _extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    return output


class StandardModel(LightningModule):
    def __init__(
        self,
        model_or_backbone_name,
        num_classes: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        pretrained: bool = True,
        pretrained_weights_path: str | None = None,
        accuracy_task: str = "multiclass",
        max_epochs_for_scheduler: int = 10,
    ) -> None:
        super().__init__()

        hparams_to_save = {
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pretrained": pretrained,
            "pretrained_weights_path": pretrained_weights_path,
            "accuracy_task": accuracy_task,
            "max_epochs_for_scheduler": max_epochs_for_scheduler,
        }
        if isinstance(model_or_backbone_name, str):
            hparams_to_save["backbone_name"] = model_or_backbone_name
        self.save_hyperparameters(hparams_to_save)

        self.model_instance_provided = isinstance(model_or_backbone_name, nn.Module)
        feature_hint = None

        if self.model_instance_provided:
            self.model = model_or_backbone_name
        elif isinstance(model_or_backbone_name, str):
            self.model, feature_hint = get_model_by_name(
                backbone_name_str=model_or_backbone_name,
                num_classes=num_classes,
                pretrained_weights_path=pretrained_weights_path,
                timm_pretrained_flag=pretrained,
                actor_name="StandardModel",
            )
        else:
            raise TypeError(
                "model_or_backbone_name must be an nn.Module or a backbone name string"
            )

        self.model = adapt_classifier_if_needed(
            self.model,
            num_classes,
            in_features_hint=feature_hint,
            model_name_hint="StandardModel",
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(
            task=accuracy_task,
            num_classes=num_classes,
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task=accuracy_task,
            num_classes=num_classes,
        )

        self.train_losses_epoch: list[float] = []
        self.train_accuracies_epoch: list[float] = []
        self.val_losses_epoch: list[float] = []
        self.val_accuracies_epoch: list[float] = []

    def forward(self, x):
        output = self.model(x)
        return _extract_logits(output)

    def _common_step(self, batch, stage_name: str):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)

        if stage_name == "train":
            metric = self.train_accuracy
            on_step = True
        else:
            metric = self.val_accuracy
            on_step = False

        metric(logits, labels)
        self.log(
            f"{stage_name}_loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=(stage_name != "train"),
        )
        self.log(
            f"{stage_name}_acc",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=(stage_name != "train"),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def on_train_epoch_end(self) -> None:
        metrics = getattr(self.trainer, "callback_metrics", {})
        if "train_loss_epoch" in metrics and "train_acc" in metrics:
            self.train_losses_epoch.append(float(metrics["train_loss_epoch"].item()))
            self.train_accuracies_epoch.append(float(metrics["train_acc"].item()))

    def on_validation_epoch_end(self) -> None:
        metrics = getattr(self.trainer, "callback_metrics", {})
        if "val_loss" in metrics and "val_acc" in metrics:
            self.val_losses_epoch.append(float(metrics["val_loss"].item()))
            self.val_accuracies_epoch.append(float(metrics["val_acc"].item()))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        t_max_epochs = int(self.hparams.max_epochs_for_scheduler)
        if self.trainer and self.trainer.max_epochs is not None:
            t_max_epochs = int(self.trainer.max_epochs)
        t_max_epochs = max(1, t_max_epochs)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_metrics_history(self) -> dict[str, list[float]]:
        return {
            "train_losses": self.train_losses_epoch,
            "train_accuracies": self.train_accuracies_epoch,
            "val_losses": self.val_losses_epoch,
            "val_accuracies": self.val_accuracies_epoch,
        }


class DistillerTrainer(LightningModule):
    def __init__(
        self,
        num_classes: int,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float,
        alpha: float,
        learning_rate: float,
        weight_decay: float,
        accuracy_task: str = "multiclass",
        max_epochs_for_scheduler: int = 10,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            {
                "num_classes": num_classes,
                "temperature": temperature,
                "alpha": alpha,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "accuracy_task": accuracy_task,
                "max_epochs_for_scheduler": max_epochs_for_scheduler,
            }
        )

        if not isinstance(teacher_model, nn.Module):
            raise TypeError("teacher_model must be an nn.Module instance")
        if not isinstance(student_model, nn.Module):
            raise TypeError("student_model must be an nn.Module instance")

        self.teacher = adapt_classifier_if_needed(
            teacher_model,
            num_classes,
            model_name_hint="teacher_model",
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = adapt_classifier_if_needed(
            student_model,
            num_classes,
            model_name_hint="student_model",
        )

        self.distill_wrapper = DistillWrapper(
            teacher=self.teacher,
            student=self.student,
            temperature=temperature,
            alpha=alpha,
        )

        self.val_criterion = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(
            task=accuracy_task,
            num_classes=num_classes,
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task=accuracy_task,
            num_classes=num_classes,
        )

        self.train_losses_epoch: list[float] = []
        self.train_accuracies_epoch: list[float] = []
        self.val_losses_epoch: list[float] = []
        self.val_accuracies_epoch: list[float] = []

    def forward(self, x):
        student_output = self.student(x)
        return _extract_logits(student_output)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        distillation_loss = self.distill_wrapper(inputs, labels=labels)
        self.log(
            "train_distill_loss",
            distillation_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        with torch.no_grad():
            student_logits = self(inputs)
        self.train_accuracy(student_logits, labels)
        self.log(
            "train_student_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return distillation_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        student_logits = self(inputs)
        val_loss = self.val_criterion(student_logits, labels)

        self.val_accuracy(student_logits, labels)
        self.log(
            "val_student_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_student_acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return val_loss

    def on_train_epoch_end(self) -> None:
        metrics = getattr(self.trainer, "callback_metrics", {})
        if "train_distill_loss_epoch" in metrics and "train_student_acc" in metrics:
            self.train_losses_epoch.append(
                float(metrics["train_distill_loss_epoch"].item())
            )
            self.train_accuracies_epoch.append(float(metrics["train_student_acc"].item()))

    def on_validation_epoch_end(self) -> None:
        metrics = getattr(self.trainer, "callback_metrics", {})
        if "val_student_loss" in metrics and "val_student_acc" in metrics:
            self.val_losses_epoch.append(float(metrics["val_student_loss"].item()))
            self.val_accuracies_epoch.append(float(metrics["val_student_acc"].item()))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.distill_wrapper.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        t_max_epochs = int(self.hparams.max_epochs_for_scheduler)
        if self.trainer and self.trainer.max_epochs is not None:
            t_max_epochs = int(self.trainer.max_epochs)
        t_max_epochs = max(1, t_max_epochs)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_metrics_history(self) -> dict[str, list[float]]:
        return {
            "train_losses": self.train_losses_epoch,
            "train_accuracies": self.train_accuracies_epoch,
            "val_losses": self.val_losses_epoch,
            "val_accuracies": self.val_accuracies_epoch,
        }


class EvaluationModule(LightningModule):
    def __init__(self, model_to_eval: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.model = model_to_eval
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )
        self.recall = torchmetrics.Recall(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )
        self.f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )

    def forward(self, x):
        output = self.model(x)
        return _extract_logits(output)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        self.accuracy.update(outputs, labels)
        self.precision.update(outputs, labels)
        self.recall.update(outputs, labels)
        self.f1.update(outputs, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self) -> None:
        acc_val = self.accuracy.compute()
        prec_val = self.precision.compute()
        rec_val = self.recall.compute()
        f1_val = self.f1.compute()

        self.log_dict(
            {
                "test_accuracy": acc_val,
                "test_precision": prec_val,
                "test_recall": rec_val,
                "test_f1_score": f1_val,
            }
        )
