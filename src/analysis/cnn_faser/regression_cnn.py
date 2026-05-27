import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from analysis.utils.datasets import CNN3DDataset, CNNProjectionDataset
from analysis.utils.models import ClassifierProjectionCNN, RegressionCNN
from analysis.utils.torch_data_utils import evaluate_model
from analysis.utils.torch_utils import get_device, show_model, train
from analysis.utils.utils import get_figures_path, get_torch_path, get_weights_path
from analysis.utils.validation_utils import (
    plot_confusion_matrix,
    plot_probabilities,
    plot_roc_curve,
    plot_training_metrics,
)


def main(
    batch_size: int = 128,
    num_epochs: int = 50,
    feature_dim: int = 128,
    cuda_device: str = "cuda:0",
    recreate: bool = True,
) -> None:
    geometry_label = "cnn_projection_200um_bins_10000_10003"
    torch_path = get_torch_path() / geometry_label
    weights_path = get_weights_path() / geometry_label / "vx_regression"
    figures_path = get_figures_path() / geometry_label / "vx_regression"

    weights_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    dataset = torch.load(torch_path / "cnn.pt", weights_only=False, map_location="cpu")

    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.1, random_state=42, stratify=dataset.labels
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    device = get_device(cuda_device=cuda_device)
    model = RegressionCNN(feature_dim=feature_dim)
    if (weights_path / "model_weights.pth").exists():
        print("Loading existing model weights...")
        checkpoint = torch.load(weights_path / "model_weights.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    show_model(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

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
        target_label="delta_vx",
        regression=True,
    )
    print("Training completed!")

    print("Evaluating model on test set...")
    # Evaluate regression model
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            if isinstance(data, (tuple, list)):
                data = tuple(d.to(device) for d in data)
            else:
                data = data.to(device)

            target_val = target["delta_vx"].to(device).float()
            output = model(
                data
            ).squeeze()  # Remove extra dimension from (batch, 1) to (batch,)

            y_true_list.extend(target_val.cpu().numpy())
            y_pred_list.extend(output.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Calculate regression metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    print("Regression Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Target range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

    # Generate plots
    print("Generating evaluation plots...")

    # Training metrics
    plot_training_metrics(
        weights_path / "training_metrics.npz",
        figure_path=figures_path / "training_metrics.png",
    )

    print(f"Figures saved to {figures_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-n", "--num_epochs", type=int, default=50)
    parser.add_argument("-f", "--feature_dim", type=int, default=128)
    parser.add_argument(
        "-g", "--gpu", type=str, default="cuda:0", help="'cuda:0', or 'cuda:1'"
    )
    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        feature_dim=args.feature_dim,
        cuda_device=args.gpu,
    )
