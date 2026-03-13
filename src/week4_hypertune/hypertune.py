from pathlib import Path
from typing import Dict

import ray
import torch
import torch.nn as nn
from filelock import FileLock
from loguru import logger
from ray import train as ray_train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MAX_EPOCHS = 8
BATCH_SIZE = 32
IMAGE_SIZE = 128
NUM_CLASSES = 102


class SmallCNN(nn.Module):
    def __init__(self, base_channels: int = 32, dropout: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_dataloaders(data_dir: Path):
    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    with FileLock(str(data_dir / ".lock")):
        train_dataset = datasets.Flowers102(
            root=str(data_dir),
            split="train",
            download=True,
            transform=transform_train,
        )

        valid_dataset = datasets.Flowers102(
            root=str(data_dir),
            split="val",
            download=True,
            transform=transform_eval,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, valid_loader


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(config: Dict):
    data_dir = Path(config["data_dir"])

    train_loader, valid_loader = get_dataloaders(data_dir)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    model = SmallCNN(
        base_channels=config["base_channels"],
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        ray_train.report(
            {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epoch": epoch + 1,
            }
    )


if __name__ == "__main__":
    ray.init()

    data_dir = Path("data/raw/flowers").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using data directory: {data_dir}")

    tune_dir = Path("logs/ray_flowers").resolve()
    tune_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "data_dir": str(data_dir),
        "lr": tune.grid_search([1e-4, 3e-4, 1e-3, 3e-3]),
        "base_channels": tune.grid_search([16, 32, 64, 96]),
        "dropout": 0.2,
    }

    reporter = CLIReporter(
        metric_columns=[
            "val_loss",
            "val_accuracy",
            "train_loss",
            "train_accuracy",
            "training_iteration",
        ]
    )

    analysis = tune.run(
        train,
        config=config,
        metric="val_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(tune_dir),
        verbose=1,
    )

    best_trial = analysis.get_best_trial(metric="val_loss", mode="min", scope="last")
    print("\nBest trial config:")
    print(best_trial.config)
    print("Best trial final validation loss:")
    print(best_trial.last_result["val_loss"])
    print("Best trial final validation accuracy:")
    print(best_trial.last_result["val_accuracy"])

    ray.shutdown()