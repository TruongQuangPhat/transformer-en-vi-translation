from typing import Any
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

class Trainer:
    """Train and validate a Transformer translation model."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        device: torch.device,
    ) -> None:
        """
        Args:
            model: Transformer translation model.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            criterion: Loss function.
            optimizer: Optimizer for model parameters.
            scheduler: Learning rate scheduler.
            device: Device to run the training on (CPU or GPU).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self) -> float:
        """Train the model for one epoch."""

        self.model.train()

        total_loss = 0.0

        for batch in self.train_loader:
            src_ids = batch["src_ids"].to(self.device)
            tgt_ids = batch["tgt_ids"].to(self.device)

            decoder_input = tgt_ids[:, :-1]
            targets = tgt_ids[:, 1:]

            self.optimizer.zero_grad()

            logits, _, _, _ = self.model(
                src_ids=src_ids,
                tgt_ids=decoder_input,
            )

            loss = self.criterion(logits, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model on the validation set."""

        self.model.eval()

        total_loss = 0.0

        for batch in self.val_loader:
            src_ids = batch["src_ids"].to(self.device)
            tgt_ids = batch["tgt_ids"].to(self.device)

            decoder_input = tgt_ids[:, :-1]
            targets = tgt_ids[:, 1:]

            logits, _, _, _ = self.model(
                src_ids=src_ids,
                tgt_ids=decoder_input,
            )

            loss = self.criterion(logits, targets)

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> tuple[int, float]:
        """
        Load model, optimizer, scheduler, and training state from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            A tuple containing:
            - start_epoch: The next epoch to train.
            - best_val_loss: Best validation loss recorded so far.
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"Starting from epoch {start_epoch}")

        return start_epoch, best_val_loss

    def fit(
        self,
        num_epochs: int,
        checkpoint_dir: Path | None = None,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
    ) -> list[dict[str, float]]:
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs: The total number of epochs to train up to.
            checkpoint_dir: The directory to save checkpoints in.
            start_epoch: The epoch to start training from.
            best_val_loss: Best validation loss recorded before resuming.

        Returns:
            Training history for the current training run.
        """
        history = []

        for epoch in range(start_epoch, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            current_lr = self.scheduler.get_last_lr()[0]

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }

            history.append(record)

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={current_lr:.6e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                if checkpoint_dir is not None:
                    self.save_checkpoint(
                        checkpoint_dir / "best.pt",
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                    )

            if checkpoint_dir is not None:
                self.save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                )

        return history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        best_val_loss: float,
    ) -> None:
        """
        Save the model checkpoint.

        Args:
            path: The path to save the checkpoint.
            epoch: The current epoch.
            best_val_loss: The best validation loss so far.
        """

        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            },
            path,
        )