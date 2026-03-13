from pathlib import Path
from typing import Dict

import ray
import torch
import torch.nn as nn
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from torchvision import transforms
import torchvision
from torchvision.models import ResNet18_Weights

from datetime import datetime

NUM_SAMPLES = 1
MAX_EPOCHS = 10


class AugmentPreprocessor:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X = [self.transform(x) for x in X]
        return torch.stack(X), torch.stack(y).long().view(-1)


def _freeze_all(model: nn.Module) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False


def _unfreeze_head(model: nn.Module) -> None:
    # resnet last layer is .fc
    for _, param in model.fc.named_parameters():  # type: ignore[attr-defined]
        param.requires_grad = True


def _maybe_unfreeze_layer4(model: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    for _, param in model.layer4.named_parameters():  # type: ignore[attr-defined]
        param.requires_grad = True


def train(config: Dict):
    """
    The train function should receive a config file, which is a Dict.
    ray will modify the values inside the config before it is passed to the train function.
    """
    from mads_datasets import DatasetFactoryProvider, DatasetType

    data_dir = Path(config.pop("data_dir")).resolve()

    flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)

    # Make images bigger, because we crop to 224 later
    flowersfactory.settings.img_size = (500, 500)

    # Data augmentation + resnet normalization
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    trainprocessor = AugmentPreprocessor(data_transforms["train"])
    validprocessor = AugmentPreprocessor(data_transforms["val"])

    with FileLock(str(data_dir / ".lock")):
        streamers = flowersfactory.create_datastreamer(batchsize=int(config["batchsize"]))
        train_stream = streamers["train"]
        valid_stream = streamers["valid"]

    # Different preprocessors for train and validation
    train_stream.preprocessor = trainprocessor
    valid_stream.preprocessor = validprocessor

    # Metric + model
    accuracy = metrics.Accuracy()

    # Build pretrained ResNet18
    resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Swap head to match flowers classes (default flowers is 5)
    output_size = int(config["output_size"])
    in_features = resnet.fc.in_features  # type: ignore[attr-defined]

    # Minimal head or a small MLP head
    head_hidden = int(config["head_hidden"])
    head_dropout = float(config["head_dropout"])

    if head_hidden > 0:
        resnet.fc = nn.Sequential(  # type: ignore[attr-defined]
            nn.Linear(in_features, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, output_size),
        )
    else:
        resnet.fc = nn.Sequential(  # type: ignore[attr-defined]
            nn.Linear(in_features, output_size),
        )

    # Freeze backbone, train only head (optionally unfreeze last block)
    _freeze_all(resnet)
    _unfreeze_head(resnet)
    _maybe_unfreeze_layer4(resnet, enabled=bool(config["unfreeze_layer4"]))

    trainersettings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=[accuracy],
        logdir=Path("."),
        train_steps=len(train_stream),  # type: ignore
        valid_steps=len(valid_stream),  # type: ignore
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 2},
        earlystop_kwargs=None,
    )

    # Device selection (keep same style)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = "cpu"  # type: ignore

    logger.info(f"Using {device}")
    if device != "cpu":
        logger.warning(f"using acceleration with {device}. Check if it actually speeds up!")

    # Optimizer settings from config
    lr = float(config["lr"])
    weight_decay = float(config["weight_decay"])

    trainer = Trainer(
        model=resnet,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW,  # type: ignore
        traindataloader=train_stream.stream(),
        validdataloader=valid_stream.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        device=str(device),
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
    )

    trainer.loop()


if __name__ == "__main__":
    ray.init()

    data_dir = Path("data/raw/flowers").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")

    tune_dir = Path("logs/ray").resolve()
    search = HyperOptSearch()
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=1,
        reduction_factor=3,
        max_t=MAX_EPOCHS,
    )

    config = {
        # data
        "data_dir": data_dir,
        "batchsize": 32,
        "output_size": 5,

        # head / finetuning knobs (later hypotheses will drive these)
        "head_hidden": 128,
        "head_dropout": 0.0,
        "unfreeze_layer4": tune.choice([False, True]),

        # optimization
        "lr": tune.loguniform(1e-5, 3e-3),
        "weight_decay": 1e-4,
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(tune_dir),
        num_samples=NUM_SAMPLES,
        search_alg=search,
        scheduler=scheduler,
        verbose=1,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"logs/ray/{timestamp}.csv"

    df = analysis.results_df
    df.to_csv(outfile, index=False)

    print(f"Saved results to {outfile}")
    print(df.columns)

    ray.shutdown()