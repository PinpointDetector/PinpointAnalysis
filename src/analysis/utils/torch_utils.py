import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric

from analysis.utils.torch_data_utils import (
    train_epoch,
    train_epoch_regression,
    validate_epoch,
    validate_epoch_regression,
)
from analysis.utils.torch_geometric_utils import train_epoch_geo, validate_epoch_geo


class EarlyStopper:
    def __init__(self, patience: int = 0, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience + 1:
                print(
                    "Early stopping."
                    f" Current validation loss: {validation_loss:.4f}, "
                    f" best validation loss: {self.min_validation_loss:.4f}, "
                )
                return True
        return False


def load_weights(model: nn.Module, device: torch.device, weights_path: Path) -> None:
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader | torch_geometric.loader.DataLoader,
    test_loader: torch.utils.data.DataLoader | torch_geometric.loader.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    output_path: Path,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    early_stopper_patience: int = 10,
    early_stopper_min_delta: float = 0.0,
    recreate: bool = False,
    regression: bool = False,
    target_label: str | None = None,
) -> None:
    if isinstance(output_path, str):
        output_path = Path(output_path).resolve()
    weights_path = output_path / "model_weights.pth"

    if weights_path.exists() and not recreate:
        print(f"Loading weights from {weights_path.name}.")
        load_weights(model, device, weights_path)
    else:
        start_time = time.time()

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0
        best_val_loss = float("inf")
        metrics_path = output_path / "training_metrics.npz"

        early_stopper = EarlyStopper(
            patience=early_stopper_patience, min_delta=early_stopper_min_delta
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
        )

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            # Training
            if regression:
                train_loss = train_epoch_regression(
                    model=model,
                    train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    target_label=target_label,
                )
                train_acc = np.nan
            else:
                if isinstance(train_loader, torch_geometric.loader.DataLoader):
                    train_loss, train_acc = train_epoch_geo(
                        model, train_loader, criterion, optimizer, device
                    )
                elif isinstance(train_loader, torch.utils.data.DataLoader):
                    train_loss, train_acc = train_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        device,
                        target_label=target_label,
                    )
                else:
                    raise TypeError("Unsupported DataLoader type for training.")
            # Validation
            if regression:
                val_loss = validate_epoch_regression(
                    model=model,
                    test_loader=test_loader,
                    criterion=criterion,
                    device=device,
                    target_label=target_label,
                )
                val_acc = np.nan
            else:
                if isinstance(test_loader, torch_geometric.loader.DataLoader):
                    val_loss, val_acc = validate_epoch_geo(
                        model, test_loader, criterion, device
                    )
                elif isinstance(test_loader, torch.utils.data.DataLoader):
                    val_loss, val_acc = validate_epoch(
                        model, test_loader, criterion, device, target_label=target_label
                    )
                else:
                    raise TypeError("Unsupported DataLoader type for validation.")
            # Early stopping check
            if early_stopper.early_stop(val_loss):
                break
            # Update scheduler
            if scheduler is not None:
                scheduler.step(val_loss)
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            # Print epoch results
            print(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            # Save best model
            # if val_acc > best_val_acc:
            if (regression and val_loss < best_val_loss) or (val_acc > best_val_acc):
                print(f"Saving best model weights at {weights_path}.")
                best_val_acc = val_acc
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    weights_path,
                )
        np.savez(
            metrics_path,
            train_losses=train_losses,
            train_accuracies=train_accuracies,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
        )
        elapsed_time = time.time() - start_time
        print(f"Training completed after {elapsed_time:.1f} seconds.")
        print(
            f"Best validation accuracy: {best_val_acc:.2f}%, best validation loss: {best_val_loss:.4f}"
        )


def get_device(cuda_device: str = "cuda") -> torch.device:
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def show_model(model: nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
