import argparse

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from analysis.utils.models import DGCClassifier, GraphClassifier
from analysis.utils.torch_geometric_utils import evaluate_model
from analysis.utils.torch_utils import get_device, show_model, train
from analysis.utils.utils import get_figures_path, get_torch_path, get_weights_path
from analysis.utils.validation_utils import (
    plot_confusion_matrix,
    plot_probabilities,
    plot_roc_curve,
    plot_training_metrics,
)


def main(
    batch_size: int = 8,
    num_epochs: int = 50,
    hidden_dim: int = 64,
    cuda_device: str = "cuda:0",
    recreate: bool = True,
) -> None:
    geometry_label = "gcn_200um_bins_10000_10003"
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

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    device = get_device(cuda_device=cuda_device)
    # model = GraphClassifier(
    #     num_node_features=4, hidden_channels=hidden_dim, num_classes=3
    # )
    model = DGCClassifier(input_dim=4, hidden_dim=hidden_dim, num_classes=3, k=64)
    model = model.to(device)
    show_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Starting training...")
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        output_path=weights_path,
        recreate=recreate,
        target_label="label",
    )
    print("Training completed!")

    print("Evaluating model on test set...")
    y_true, y_pred, y_prob = evaluate_model(
        model=model, test_loader=test_loader, device=device
    )

    # Calculate accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Per-class accuracy
    class_names = ["CC ne", "CC nm", "NC"]
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).mean()
            print(f"{class_name} accuracy: {class_acc:.4f} ({class_acc * 100:.2f}%)")

    # Generate plots
    print("Generating evaluation plots...")

    # Training metrics
    plot_training_metrics(
        weights_path / "training_metrics.npz",
        figure_path=figures_path / "training_metrics.png",
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        figure_path=figures_path / "confusion_matrix.png",
    )

    # ROC curve
    plot_roc_curve(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        figure_path=figures_path / "roc_curve.png",
    )

    # Probability distributions
    plot_probabilities(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        figures_path=figures_path / "probabilities.png",
    )

    # print(f"Results saved to {weights_path}")
    print(f"Figures saved to {figures_path}")


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
