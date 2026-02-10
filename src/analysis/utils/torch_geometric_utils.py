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


def evaluate_gravnet_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_faser: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate GravNet model on test data.

    GravNet models use position information (pos) instead of edge_index,
    and predict graph-level classes stored in y_graph.

    Args:
        model: GravNet model to evaluate
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        use_faser: Whether to use FASER features in the model
    Returns:
        Tuple of (targets, predictions, probabilities) as numpy arrays
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)

            try:
                # GravNet uses position information
                if use_faser:
                    # If using FASER features, pass them to the model
                    output = model(data.x, data.pos, data.batch, data.x_faser)
                else:
                    output = model(data.x, data.pos, data.batch)

                # Check for batch size mismatch
                num_graphs = data.y_graph.size(0)
                if output.size(0) != num_graphs:
                    continue

                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)

                all_targets.extend(data.y_graph.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

            except RuntimeError as e:
                # Skip batches with errors
                continue

    return np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)


def evaluate_gravnet_model_nodes(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_faser: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate GravNet model on test data for node-level classification.

    GravNet models use position information (pos) instead of edge_index,
    and predict node-level classes stored in y.

    Args:
        model: GravNet model to evaluate (returns node predictions)
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        use_faser: Whether to use FASER features in the model
    Returns:
        Tuple of (targets, predictions, probabilities) as numpy arrays
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)

            # Skip if no node labels available
            if not hasattr(data, "y") or data.y is None:
                continue

            try:
                # GravNet uses position information
                if use_faser:
                    # If using FASER features, pass them to the model
                    output = model(data.x, data.pos, data.batch, data.x_faser)
                else:
                    output = model(data.x, data.pos, data.batch)

                # Handle models that return (node_out, graph_out) or just node_out
                if isinstance(output, tuple):
                    # Model returns (node_out, graph_out), take only node predictions
                    node_output = output[0]
                else:
                    # Model returns only node output
                    node_output = output

                # Check for node count mismatch
                num_nodes = data.y.size(0)
                if node_output.size(0) != num_nodes:
                    continue

                probabilities = torch.softmax(node_output, dim=1)
                predictions = node_output.argmax(dim=1)

                all_targets.extend(data.y.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

            except RuntimeError as e:
                # Skip batches with errors
                continue

    return np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)


def evaluate_pointnetpp_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_faser: bool = False,
) -> tuple[list, list, list, list, list, list]:
    """Evaluate PointNet/PointNet++ model on test data.

    PointNet models use position information (pos) instead of edge_index,
    and predict graph-level classes stored in y_graph.

    For models that return both node and graph predictions, returns predictions
    for both levels. For models that return only graph predictions, returns
    empty lists for node-level outputs.

    Args:
        model: PointNet/PointNet++ model to evaluate
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        use_faser: Whether to use FASER features in the model
    Returns:
        Tuple of (node_targets, node_predictions, node_probabilities,
                  graph_targets, graph_predictions, graph_probabilities)
        Each as a list of numpy arrays (one per batch)
    """
    model.eval()
    all_node_targets = []
    all_node_predictions = []
    all_node_probabilities = []
    all_graph_targets = []
    all_graph_predictions = []
    all_graph_probabilities = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)

            # Skip empty batches
            if data.x.size(0) == 0:
                continue

            try:
                # PointNet uses position information
                if use_faser:
                    # If using FASER features, pass them to the model
                    model_output = model(data.x, data.pos, data.batch, data.x_faser)
                else:
                    model_output = model(data.x, data.pos, data.batch)

                # Handle models that return (node_out, graph_out) or just graph_out
                if isinstance(model_output, tuple):
                    # Model returns (node_out, graph_out)
                    node_out, graph_out = model_output

                    # Node-level predictions
                    node_prob = torch.softmax(node_out, dim=1)
                    node_pred = node_out.argmax(dim=1)

                    all_node_targets.append(data.y.cpu().numpy())
                    all_node_predictions.append(node_pred.cpu().numpy())
                    all_node_probabilities.append(node_prob.cpu().numpy())

                    # Graph-level predictions
                    output = graph_out
                else:
                    # Model returns only graph output
                    output = model_output

                # Check for batch size mismatch
                num_graphs = data.y_graph.size(0)
                if output.size(0) != num_graphs:
                    continue

                # Graph-level predictions
                graph_prob = torch.softmax(output, dim=1)
                graph_pred = output.argmax(dim=1)

                all_graph_targets.append(data.y_graph.cpu().numpy())
                all_graph_predictions.append(graph_pred.cpu().numpy())
                all_graph_probabilities.append(graph_prob.cpu().numpy())

            except RuntimeError as e:
                # Skip batches with errors
                continue

    return (
        all_node_targets,
        all_node_predictions,
        all_node_probabilities,
        all_graph_targets,
        all_graph_predictions,
        all_graph_probabilities,
    )
