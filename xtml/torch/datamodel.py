from __future__ import annotations
import os
import json
import dataclasses as dc
import logging
import datetime as dt
from typing import Type, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dc.dataclass
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    num_workers: int

    @staticmethod
    def from_cfg(cfg: dict[str, Any]) -> DataLoaderConfig:
        return DataLoaderConfig(**cfg)

    def asdict(self) -> dict[str, Any]:
        return dc.asdict(self)


@dc.dataclass
class TrainConfig:
    device: Type[torch.device]
    epochs: int
    loss_fn: Type[nn.Module]
    loss_fn_params: dict[str, Any]
    optimizer: Type[optim.Optimizer]
    optimizer_params: dict[str, Any]
    checkpoint_dir: str
    model_dir: str

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if isinstance(self.loss_fn, str):
            self.loss_fn = getattr(nn, self.loss_fn)(**self.loss_fn_params)
        if isinstance(self.optimizer, str):
            self.optimizer = getattr(optim, self.optimizer)

    @staticmethod
    def from_cfg(cfg: dict[str, Any]) -> TrainConfig:
        cfg["loss_fn_params"] = cfg.get("loss_fn_params", {})
        cfg["optimizer_params"] = cfg.get("optimizer_params", {})
        return TrainConfig(
            device=cfg["device"],
            epochs=cfg["epochs"],
            loss_fn=cfg["loss_fn"],
            loss_fn_params=cfg["loss_fn_params"],
            optimizer=cfg["optimizer"],
            optimizer_params=cfg["optimizer_params"],
            checkpoint_dir=cfg["checkpoint_dir"],
            model_dir=cfg["model_dir"],
        )

    def asdict(self) -> dict[str, Any]:
        return dc.asdict(self)


class TorchTrainer:
    def __init__(self, model: nn.Module, train_config: TrainConfig) -> None:
        self.model: nn.Module = model.to(train_config.device)
        self.train_config: TrainConfig = train_config
        self.loss_fn: nn.Module = train_config.loss_fn
        self.optimizer: optim.Optimizer = train_config.optimizer(
            model.parameters(), **train_config.optimizer_params
        )
        self.epoch: int = -1
        self.loss: float = -1
        logger.info(f"{self.model=} \n {self.loss_fn=} \n {self.optimizer=}")

        self.checkpoint_dir: Path = Path(train_config.checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.model_dir: Path = Path(train_config.model_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.metadata = {
            "loss_function": str(self.loss_fn),
            "optimizer": str(self.optimizer),
        }

    def train(self, train_loader: DataLoader) -> None:
        self.model.train()
        for epoch in range(self.train_config.epochs):
            if epoch <= self.epoch:
                continue

            total_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.train_config.device)
                targets = targets.to(self.train_config.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            logger.info(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")
            self.epoch = epoch
            self.loss = total_loss / len(train_loader)
            self.save_checkpoint()

    def save_checkpoint(self) -> None:
        now = int(dt.datetime.now().timestamp())
        epoch = str(self.epoch).zfill(3)
        loss = str(f"{self.loss:.2f}").replace(".", "_")

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"T{now}-E{epoch}-L{loss}.pt",
        )
        checkpoint = {
            "epoch": self.epoch,
            "loss": self.loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_config": self.train_config.asdict(),
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_weights(self, checkpoint_path: str, load_optimizer: bool = False) -> None:
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file does not exist at {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.train_config.device)

        # Load model state
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Model weights loaded from checkpoint: {checkpoint_path}")
        else:
            logger.warning(
                f"No model_state_dict found in checkpoint: {checkpoint_path}"
            )

        # Optionally load optimizer state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(f"Optimizer state loaded from checkpoint: {checkpoint_path}")

        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
            logger.info(f"Epoch is restored to {self.epoch=}")

    def export_model(self, export_path: str = None) -> None:
        export_path = export_path if export_path else self.train_config.model_dir
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        model_name = self.model.__class__.__name__
        model_path = os.path.join(export_path, f"{model_name}.pt")

        self.model.eval()  # Switch to evaluation mode before exporting
        scripted_model = torch.jit.script(self.model)  # Script the model
        scripted_model.save(model_path)  # Save the scripted model
        logger.info(f"JIT-scripted model exported to {model_path}")

    def export_metadata(self, export_path: str = None) -> None:
        if export_path is None:
            export_path = self.train_config.model_dir
        metadata_path = os.path.join(export_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
        logger.info(f"Metadata exported to {metadata_path}")

    @staticmethod
    def from_checkpoint(checkpoint_path: str) -> TorchTrainer:
        raise NotImplementedError("Not Supported yet")
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file does not exist at {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path)

        # Restore TrainConfig from the checkpoint
        train_config_dict = checkpoint["train_config"]
        train_config = TrainConfig.from_cfg(train_config_dict)
        # Instantiate the model based on the restored TrainConfig
        # Note: This assumes you have a way to dynamically instantiate your model based on TrainConfig
        model = MyModel()  # Placeholder for model instantiation

        # Create a new ModelWrapper instance
        wrapper = TorchTrainer(model, train_config)
        wrapper.model.load_state_dict(checkpoint["model_state_dict"])
        wrapper.model.to(train_config.device)
        wrapper.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(
            f"Model, optimizer, and training configuration loaded from checkpoint: {checkpoint_path}"
        )
        return wrapper
