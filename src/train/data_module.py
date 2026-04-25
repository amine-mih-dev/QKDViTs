from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=10),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    resize_size = max(240, image_size)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        batch_size: int = 16,
        num_workers: int = 8,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.train_transform = build_train_transform(image_size)
        self.eval_transform = build_eval_transform(image_size)

        self.train_dataset: datasets.ImageFolder | None = None
        self.val_dataset: datasets.ImageFolder | None = None
        self.test_dataset: datasets.ImageFolder | None = None
        self.num_classes: int | None = None
        self.class_names: list[str] = []

    def _validate_dirs(self) -> None:
        missing = [
            str(path)
            for path in (self.train_dir, self.val_dir, self.test_dir)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing dataset directories: {missing}")

    def setup(self, stage: str | None = None) -> None:
        self._validate_dirs()

        if stage in ("fit", None):
            self.train_dataset = datasets.ImageFolder(
                str(self.train_dir),
                transform=self.train_transform,
            )
            self.val_dataset = datasets.ImageFolder(
                str(self.val_dir),
                transform=self.eval_transform,
            )
            self.num_classes = len(self.train_dataset.classes)
            self.class_names = list(self.train_dataset.classes)

        if stage in ("test", None):
            self.test_dataset = datasets.ImageFolder(
                str(self.test_dir),
                transform=self.eval_transform,
            )
            if self.num_classes is None:
                self.num_classes = len(self.test_dataset.classes)
                self.class_names = list(self.test_dataset.classes)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        persistent_workers = self.num_workers > 0
        use_pin_memory = torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
