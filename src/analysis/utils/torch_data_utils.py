import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Train and validate functions for default torch.utils.data DataLoader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    target_label: str | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(train_bar):
        if isinstance(target, dict) and target_label is not None:
            target = target[target_label]
        # Handle both single tensors and tuples/lists of tensors
        if isinstance(data, (tuple, list)):
            data = tuple(d.to(device) for d in data)
        else:
            data = data.to(device)
        target = target.to(device)  # Ensure target is float32

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        avg_loss = running_loss / (batch_idx + 1)
        acc = correct / total
        train_bar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{100.0 * acc:.2f}%"})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_epoch_regression(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    target_label: str | None = None,
) -> float:
    model.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(train_bar):
        if isinstance(target, dict) and target_label is not None:
            target = target[target_label]
        # Handle both single tensors and tuples/lists of tensors
        if isinstance(data, (tuple, list)):
            data = tuple(d.to(device) for d in data)
        else:
            data = data.to(device)
        target = target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        train_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})
    return running_loss / len(train_loader)


def validate_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_label: str | None = None,
) -> float:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Handle both single tensors and tuples/lists of tensors
            if isinstance(data, (tuple, list)):
                data = tuple(d.to(device) for d in data)
            else:
                data = data.to(device)
            if isinstance(target, dict) and target_label is not None:
                target = target[target_label]
            target = target.to(device)  # Ensure target is float32

            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            current_loss = running_loss / (batch_idx + 1)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    epoch_acc = 100.0 * correct / total
    return current_loss, epoch_acc


def validate_epoch_regression(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_label: str | None = None,
) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Handle both single tensors and tuples/lists of tensors
            if isinstance(data, (tuple, list)):
                data = tuple(d.to(device) for d in data)
            else:
                data = data.to(device)
            if isinstance(target, dict) and target_label is not None:
                target = target[target_label]
            target = target.to(device).float()  # Ensure target is float32
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
    return running_loss / len(test_loader)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    target_label: str = "label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            # Handle both single tensors and tuples/lists of tensors
            if isinstance(data, (tuple, list)):
                data = tuple(d.to(device) for d in data)
            else:
                data = data.to(device)
            if isinstance(target, dict):
                target = target[target_label]
            target = target.to(device)

            output = model(data)
            probabilities = F.softmax(output, dim=1)
            predictions = output.argmax(dim=1)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)
