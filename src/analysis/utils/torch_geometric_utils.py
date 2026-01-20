import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Train and validate functions for torch_geometric DataLoader


def train_epoch_geo(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train one epoch for torch_geometric DataLoader."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(train_bar):
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        # training metrics
        running_loss += loss.item()
        correct += (output.argmax(dim=1) == batch.y).sum().item()
        total += batch.y.size(0)
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100 * correct / total
        train_bar.set_postfix(
            {"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"}
        )
    return current_loss, current_acc


def validate_epoch_geo(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate one epoch for torch_geometric DataLoader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y)
            # validation metrics
            running_loss += loss.item()
            current_loss = running_loss / (batch_idx + 1)
            correct += (output.argmax(dim=1) == batch.y).sum().item()
            total += batch.y.size(0)

    epoch_acc = 100.0 * correct / total
    return current_loss, epoch_acc


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)

            all_targets.extend(batch.y.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)
