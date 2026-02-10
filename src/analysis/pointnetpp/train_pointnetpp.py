import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analysis.pointnetpp.model import NeutrinoPointNetPlusPlus
from analysis.utils.torch_utils import get_device, show_model
from analysis.utils.utils import get_torch_path, get_weights_path


def train_epoch_pointnetpp(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    node_class_weights: torch.Tensor = None,
    graph_class_weights: torch.Tensor = None,
) -> tuple[float, float, float, float, float, float]:
    """Train one epoch with PointNet++ dual classification.

    Returns:
        current_loss, node_acc, node_weighted_acc, graph_acc, graph_weighted_acc, final_loss
    """
    model.train()
    running_loss = 0
    node_correct = 0
    node_total = 0
    graph_correct = 0
    graph_total = 0
    node_weighted_correct = 0
    node_weighted_total = 0
    graph_weighted_correct = 0
    graph_weighted_total = 0

    train_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(train_bar):
        batch = batch.to(device)

        # Skip empty batches
        if batch.x.size(0) == 0:
            continue

        optimizer.zero_grad()

        try:
            # Forward pass - PointNet++ uses (x, pos, batch)
            node_out, graph_out = model(batch.x, batch.pos, batch.batch)

            # Check for batch size mismatch (can happen if graphs become empty after FPS)
            num_graphs = batch.y_graph.size(0)
            if graph_out.size(0) != num_graphs:
                print(
                    f"\nSkipping batch {batch_idx}: graph output size {graph_out.size(0)} != target size {num_graphs}"
                )
                continue

            # Compute node-level loss
            if node_class_weights is not None:
                node_loss = F.cross_entropy(
                    node_out, batch.y, weight=node_class_weights
                )
            else:
                node_loss = criterion(node_out, batch.y)

            # Compute graph-level loss
            if graph_class_weights is not None:
                graph_loss = F.cross_entropy(
                    graph_out, batch.y_graph, weight=graph_class_weights
                )
            else:
                graph_loss = criterion(graph_out, batch.y_graph)
        except RuntimeError as e:
            print(f"\nSkipping batch {batch_idx} due to error: {e}")
            continue

        # Combined loss (equal weight)
        loss = 0.5 * node_loss + 0.5 * graph_loss

        if torch.isnan(loss):
            print(f"\nSkipping batch {batch_idx} due to NaN loss")
            continue

        loss.backward()

        # Gradient clipping - clip but don't skip
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if grad_norm > 5.0:
            print(f"\nWarning: Large gradient norm {grad_norm:.2f} (clipped to 1.0)")

        optimizer.step()

        # Metrics
        running_loss += loss.item()
        current_loss = running_loss / (batch_idx + 1)

        # Node-level accuracy
        node_pred = node_out.argmax(dim=1)
        node_correct += (node_pred == batch.y).sum().item()
        node_total += batch.y.size(0)
        node_acc = 100 * node_correct / node_total

        # Node-level weighted accuracy
        if node_class_weights is not None:
            node_weights = node_class_weights[batch.y]
            node_weighted_correct += (
                ((node_pred == batch.y).float() * node_weights).sum().item()
            )
            node_weighted_total += node_weights.sum().item()
            node_weighted_acc = 100 * node_weighted_correct / node_weighted_total
        else:
            node_weighted_acc = node_acc

        # Graph-level accuracy
        graph_pred = graph_out.argmax(dim=1)
        graph_correct += (graph_pred == batch.y_graph).sum().item()
        graph_total += batch.y_graph.size(0)
        graph_acc = 100 * graph_correct / graph_total

        # Graph-level weighted accuracy
        if graph_class_weights is not None:
            graph_weights = graph_class_weights[batch.y_graph]
            graph_weighted_correct += (
                ((graph_pred == batch.y_graph).float() * graph_weights).sum().item()
            )
            graph_weighted_total += graph_weights.sum().item()
            graph_weighted_acc = 100 * graph_weighted_correct / graph_weighted_total
        else:
            graph_weighted_acc = graph_acc

        train_bar.set_postfix(
            {
                "Loss": f"{current_loss:.4f}",
                "NodeAcc": f"{node_acc:.2f}%",
                "GraphAcc": f"{graph_acc:.2f}%",
            }
        )
        # except (RuntimeError, ValueError) as e:
        #     print(f"\nSkipping batch {batch_idx} due to error: {e}")
        # False

    final_node_weighted_acc = (
        100 * node_weighted_correct / node_weighted_total
        if node_weighted_total > 0
        else node_acc
    )
    final_graph_weighted_acc = (
        100 * graph_weighted_correct / graph_weighted_total
        if graph_weighted_total > 0
        else graph_acc
    )

    return (
        current_loss,
        node_acc,
        final_node_weighted_acc,
        graph_acc,
        final_graph_weighted_acc,
        running_loss / (batch_idx + 1),
    )


def validate_epoch_pointnetpp(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    node_class_weights: torch.Tensor = None,
    graph_class_weights: torch.Tensor = None,
) -> tuple[float, float, float, float, float]:
    """Validate one epoch with PointNet++ dual classification.

    Returns:
        avg_loss, node_acc, node_weighted_acc, graph_acc, graph_weighted_acc
    """
    model.eval()
    total_loss = 0
    node_correct = 0
    node_total = 0
    graph_correct = 0
    graph_total = 0
    node_weighted_correct = 0
    node_weighted_total = 0
    graph_weighted_correct = 0
    graph_weighted_total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = batch.to(device)

            if batch.x.size(0) == 0:
                continue

            try:
                # Forward pass - PointNet++ uses (x, pos, batch)
                node_out, graph_out = model(batch.x, batch.pos, batch.batch)

                # Compute losses
                if node_class_weights is not None:
                    node_loss = F.cross_entropy(
                        node_out, batch.y, weight=node_class_weights
                    )
                else:
                    node_loss = criterion(node_out, batch.y)

                if graph_class_weights is not None:
                    graph_loss = F.cross_entropy(
                        graph_out, batch.y_graph, weight=graph_class_weights
                    )
                else:
                    graph_loss = criterion(graph_out, batch.y_graph)

                loss = 0.5 * node_loss + 0.5 * graph_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nSkipping validation batch {batch_idx} due to NaN/Inf")
                    continue

                total_loss += loss.item() * batch.y.size(0)

                # Node-level metrics
                node_pred = node_out.argmax(dim=1)
                node_correct += (node_pred == batch.y).sum().item()
                node_total += batch.y.size(0)

                if node_class_weights is not None:
                    node_weights = node_class_weights[batch.y]
                    node_weighted_correct += (
                        ((node_pred == batch.y).float() * node_weights).sum().item()
                    )
                    node_weighted_total += node_weights.sum().item()

                # Graph-level metrics
                graph_pred = graph_out.argmax(dim=1)
                graph_correct += (graph_pred == batch.y_graph).sum().item()
                graph_total += batch.y_graph.size(0)

                if graph_class_weights is not None:
                    graph_weights = graph_class_weights[batch.y_graph]
                    graph_weighted_correct += (
                        ((graph_pred == batch.y_graph).float() * graph_weights)
                        .sum()
                        .item()
                    )
                    graph_weighted_total += graph_weights.sum().item()

            except RuntimeError as e:
                print(f"\nSkipping validation batch {batch_idx} due to error: {e}")
                continue

    if node_total == 0 or graph_total == 0:
        return float("inf"), 0.0, 0.0, 0.0, 0.0

    node_acc = 100 * node_correct / node_total
    graph_acc = 100 * graph_correct / graph_total
    avg_loss = total_loss / node_total

    node_weighted_acc = (
        100 * node_weighted_correct / node_weighted_total
        if node_weighted_total > 0
        else node_acc
    )
    graph_weighted_acc = (
        100 * graph_weighted_correct / graph_weighted_total
        if graph_weighted_total > 0
        else graph_acc
    )

    return avg_loss, node_acc, node_weighted_acc, graph_acc, graph_weighted_acc


def main(
    batch_size: int = 4,
    num_epochs: int = 50,
    cuda_device: str = "cuda:0",
) -> None:
    """Train PointNet++ model for neutrino classification."""
    torch_path = get_torch_path()
    output_path = get_torch_path() / "pointnetpp"
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = get_weights_path() / "pointnetpp"
    weights_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logging.info("Loading datasets...")
    nue_dataset = torch.load(torch_path / "10000/pointnetpp/nue.pt", weights_only=False)
    num_dataset = torch.load(torch_path / "10001/pointnetpp/num.pt", weights_only=False)
    nun_dataset = torch.load(torch_path / "10003/pointnetpp/nun.pt", weights_only=False)

    print(
        f"Loaded {len(nue_dataset)} nue, {len(num_dataset)} num, {len(nun_dataset)} nun events"
    )
    dataset = nue_dataset + num_dataset + nun_dataset

    # Train/test split
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.1, random_state=42
    )
    train_data = torch.utils.data.Subset(dataset, train_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)

    # Save test dataset
    test_dataset = [dataset[i] for i in test_indices]
    torch.save(test_dataset, output_path / "test_dataset.pt")
    np.save(output_path / "test_indices.npy", np.array(test_indices))
    print(f"Saved test dataset with {len(test_dataset)} events")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Calculate class weights
    node_labels = np.hstack([data.y.cpu().numpy() for data in dataset])
    node_counts = torch.tensor(
        [np.sum(node_labels == i) for i in range(3)], dtype=torch.float
    )
    node_freqs = node_counts / node_counts.sum()
    alpha = 0.99
    node_class_weights = (1.0 / node_freqs) ** alpha
    node_class_weights = node_class_weights / node_class_weights.mean()
    print(f"\nNode class weights: {node_class_weights}")

    graph_labels = np.array([data.y_graph.cpu().numpy() for data in dataset])
    graph_counts = torch.tensor(
        [np.sum(graph_labels == i) for i in range(3)], dtype=torch.float
    )
    graph_freqs = graph_counts / graph_counts.sum()
    graph_class_weights = (1.0 / graph_freqs) ** alpha
    graph_class_weights = graph_class_weights / graph_class_weights.mean()
    print(f"Graph class weights: {graph_class_weights}")

    # Create model
    device = get_device(cuda_device=cuda_device)
    node_class_weights = node_class_weights.to(device)
    graph_class_weights = graph_class_weights.to(device)

    model = NeutrinoPointNetPlusPlus(
        input_dim=1,  # Only nhits feature
        num_node_classes=3,
        num_graph_classes=3,
        dropout=0.2,
    )
    model = model.to(device)
    show_model(model)

    criterion = nn.CrossEntropyLoss()
    lr = 1e-4  # Balanced learning rate for PointNet++
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Learning rate: {lr}, Weight decay: {weight_decay}")

    # Training setup
    start_epoch = 0
    best_val_loss = float("inf")
    train_losses = []
    train_node_accs = []
    train_graph_accs = []
    val_losses = []
    val_node_accs = []
    val_graph_accs = []

    print("\nStarting PointNet++ training...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        (
            train_loss,
            train_node_acc,
            train_node_weighted_acc,
            train_graph_acc,
            train_graph_weighted_acc,
            _,
        ) = train_epoch_pointnetpp(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            node_class_weights,
            graph_class_weights,
        )

        (
            val_loss,
            val_node_acc,
            val_node_weighted_acc,
            val_graph_acc,
            val_graph_weighted_acc,
        ) = validate_epoch_pointnetpp(
            model,
            test_loader,
            criterion,
            device,
            node_class_weights,
            graph_class_weights,
        )

        train_losses.append(train_loss)
        train_node_accs.append(train_node_acc)
        train_graph_accs.append(train_graph_acc)
        val_losses.append(val_loss)
        val_node_accs.append(val_node_acc)
        val_graph_accs.append(val_graph_acc)

        print(
            f"Train Loss: {train_loss:.4f}, Node: {train_node_acc:.2f}%, Graph: {train_graph_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Node: {val_node_acc:.2f}%, Graph: {val_graph_acc:.2f}%"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_node_acc": val_node_acc,
                    "val_graph_acc": val_graph_acc,
                },
                weights_path / "model_weights.pth",
            )
            print(f"Saved best model with val_loss: {val_loss:.4f}")

    # Save training metrics
    np.savez(
        weights_path / "training_metrics.npz",
        train_losses=train_losses,
        train_node_accuracies=train_node_accs,
        train_graph_accuracies=train_graph_accs,
        val_losses=val_losses,
        val_node_accuracies=val_node_accs,
        val_graph_accuracies=val_graph_accs,
    )

    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PointNet++ for neutrino classification"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("-n", "--num-epochs", type=int, default=50)
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="INFO",
        choices=["INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    main(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        cuda_device=args.gpu,
    )
