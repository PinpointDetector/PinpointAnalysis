import torch
from sklearn.model_selection import train_test_split

from analysis.utils.utils import get_figures_path, get_torch_path, get_weights_path


def main() -> None:
    geometry_label = "gcn_200um_bins_10000_10003"
    torch_path = get_torch_path() / geometry_label
    weights_path = get_weights_path() / geometry_label
    figures_path = get_figures_path() / geometry_label

    weights_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    dataset = torch.load(torch_path / "gcn.pt", weights_only=False, map_location="cpu")

    _, test_indices = train_test_split(
        range(len(dataset)), test_size=0.1, random_state=42
    )
    # Save test dataset to disk
    test_dataset = [dataset[i] for i in test_indices]
    torch.save(test_dataset, torch_path / "test_dataset.pt")
    print(
        f"Saved test dataset with {len(test_dataset)} events to "
        f"{torch_path / 'test_dataset.pt'}"
    )


if __name__ == "__main__":
    main()
