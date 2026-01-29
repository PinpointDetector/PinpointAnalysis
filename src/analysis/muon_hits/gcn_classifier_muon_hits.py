import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analysis.utils.models import MuonHitNet
from analysis.utils.torch_geometric_utils import evaluate_model
from analysis.utils.torch_utils import get_device, show_model, train
from analysis.utils.utils import get_figures_path, get_torch_path, get_weights_path
from analysis.utils.validation_utils import (
    plot_confusion_matrix,
    plot_probabilities,
    plot_roc_curve,
    plot_training_metrics,
)


def train_epoch_node_classification(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor = None,
) -> tuple[float, float]:
    """Train one epoch with node-level predictions."""
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(train_bar):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)

        # Compute loss (all nodes in batch)
        if class_weights is not None:
            loss = F.cross_entropy(out, batch.y, weight=class_weights)
        else:
            loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()

        # Metrics
        # total_loss += loss.item() * batch.y.size(0)
        running_loss += loss.item()
        current_loss = running_loss / (batch_idx + 1)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
        current_acc = 100 * correct / total

        train_bar.set_postfix(
            {"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"}
        )

    return current_loss, current_acc


def validate_epoch_node_classification(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_weights: torch.Tensor = None,
) -> tuple[float, float]:
    """Validate one epoch with node-level predictions."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)

            # Compute loss
            if class_weights is not None:
                loss = F.cross_entropy(out, batch.y, weight=class_weights)
            else:
                loss = criterion(out, batch.y)

            # Metrics
            total_loss += loss.item() * batch.y.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return total_loss / total, 100 * correct / total


def main(
    batch_size: int = 8,
    num_epochs: int = 50,
    hidden_dim: int = 64,
    cuda_device: str = "cuda:0",
    recreate: bool = True,
) -> None:
    geometry_label = "muon_gcn_200um_bins_10000_10003"
    torch_path = get_torch_path() / geometry_label
    weights_path = get_weights_path() / geometry_label
    figures_path = get_figures_path() / geometry_label

    weights_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    dataset = torch.load(torch_path / "gcn.pt", weights_only=False, map_location="cpu")

    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.1, random_state=42
    )
    train_data = torch.utils.data.Subset(dataset, train_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)

    # Save test dataset to disk
    test_dataset = [dataset[i] for i in test_indices]
    torch.save(test_dataset, torch_path / "test_dataset.pt")
    print(
        f"Saved test dataset with {len(test_dataset)} events to "
        f"{torch_path / 'test_dataset.pt'}"
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Calculate class weights for imbalanced dataset
    total_muon = sum([data.y.sum().item() for data in dataset])
    total_non_muon = sum([(data.y == 0).sum().item() for data in dataset])
    weight_muon = total_non_muon / (total_muon + total_non_muon)
    weight_non_muon = total_muon / (total_muon + total_non_muon)
    class_weights = torch.tensor([weight_non_muon, weight_muon], dtype=torch.float)
    print(f"\nClass weights: non-muon={weight_non_muon:.3f}, muon={weight_muon:.3f}")

    # Create model
    device = get_device(cuda_device=cuda_device)
    class_weights = class_weights.to(device)
    model = MuonHitNet(
        input_dim=3,
        hidden_dim=32,
        num_layers=3,
        k=16,
        dropout=0.3,
        num_classes=2,
    )
    model = model.to(device)
    show_model(model)
    # model = MuonAttentionNet(
    #     input_dim=4,
    #     hidden_dim=hidden_dim,
    #     num_layers=3,
    #     num_heads=4,  # Number of attention heads
    #     dropout=0.3,
    #     num_classes=2,
    # )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc = train_epoch_node_classification(
            model, train_loader, criterion, optimizer, device, class_weights
        )
        val_loss, val_acc = validate_epoch_node_classification(
            model, test_loader, criterion, device, class_weights
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                weights_path / "model_weights.pth",
            )

    # Save training metrics
    np.savez(
        weights_path / "training_metrics.npz",
        train_losses=train_losses,
        train_accuracies=train_accs,
        val_losses=val_losses,
        val_accuracies=val_accs,
    )

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-n", "--num_epochs", type=int, default=50)
    parser.add_argument("-f", "--hidden_dim", type=int, default=64)
    parser.add_argument(
        "-g", "--gpu", type=str, default="cuda:0", help="'cuda:0', or 'cuda:1'"
    )
    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        cuda_device=args.gpu,
    )
