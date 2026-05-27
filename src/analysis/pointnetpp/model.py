import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    PointNetConv,
    fps,
    global_max_pool,
    global_mean_pool,
    knn,
    knn_graph,
    radius,
)


class SetAbstractionLayer(nn.Module):
    """
    PointNet++ Set Abstraction layer with sampling and k-NN grouping.
    Simplified version using radius grouping.
    """

    def __init__(self, ratio, k, in_channels, out_channels, use_fps=True):
        super().__init__()
        self.ratio = ratio
        self.r = 1.0  # radius for neighborhood (adjusted for normalized coords)

        # Local PointNet MLP (smaller architecture)
        local_nn = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
        )
        self.conv = PointNetConv(local_nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        """
        Args:
            x: Node features [N, in_channels]
            pos: Node positions [N, 3] (x, y, z)
            batch: Batch assignment [N]

        Returns:
            x_out: Downsampled features [M, out_channels]
            pos_out: Downsampled positions [M, 3]
            batch_out: Batch assignment [M]
        """
        # Sample points using FPS
        idx = fps(pos, batch, ratio=self.ratio)

        # Build edge index using radius from source (pos) to target (pos[idx])
        # row = source indices (in pos), col = target indices (in pos[idx])
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=32
        )
        # Edge index: [2, num_edges] where edge goes from row (source) to col (target)
        edge_index = torch.stack([col, row], dim=0)

        # Apply PointNet convolution
        # Pass None for target features - let PointNetConv aggregate from source
        x_out = self.conv((x, None), (pos, pos[idx]), edge_index)

        return x_out, pos[idx], batch[idx]


class FeaturePropagationLayer(nn.Module):
    """
    Feature propagation layer for upsampling in PointNet++.
    """

    def __init__(self, in_channels, out_channels, k=3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
        )

    def forward(self, x_fine, pos_fine, x_coarse, pos_coarse, batch_fine, batch_coarse):
        """
        Args:
            x_fine: Features at fine resolution [N_fine, C_fine]
            pos_fine: Positions at fine resolution [N_fine, 3]
            x_coarse: Features at coarse resolution [N_coarse, C_coarse]
            pos_coarse: Positions at coarse resolution [N_coarse, 3]
            batch_fine: Batch assignment [N_fine]
            batch_coarse: Batch assignment [N_coarse]

        Returns:
            x_out: Interpolated features [N_fine, out_channels]
        """
        # Find k nearest neighbors in coarse for each fine point
        edge_index = knn(pos_coarse, pos_fine, self.k, batch_coarse, batch_fine)

        # Distance-based interpolation
        row, col = edge_index
        diff = pos_fine[row] - pos_coarse[col]
        dist = (diff * diff).sum(dim=-1, keepdim=True)
        dist = torch.clamp(dist, min=1e-10)
        weight = 1.0 / dist

        # Weighted sum of coarse features
        weight_sum = torch.zeros(
            pos_fine.size(0), 1, device=pos_fine.device
        ).index_add_(0, row, weight)
        x_interpolated = torch.zeros(
            pos_fine.size(0), x_coarse.size(1), device=pos_fine.device
        ).index_add_(0, row, weight * x_coarse[col])
        x_interpolated = x_interpolated / (weight_sum + 1e-10)

        # Concatenate with fine features if they exist
        if x_fine is not None:
            x_combined = torch.cat([x_fine, x_interpolated], dim=-1)
        else:
            x_combined = x_interpolated

        return self.mlp(x_combined)


class NeutrinoPointNetPlusPlus(nn.Module):
    """
    PointNet++ based model for calorimeter data with dual prediction heads.

    Predicts both pixel-level labels (e.g., particle type per hit) and
    event-level labels (e.g., CC nue, CC num, NC).

    Architecture:
    - Hierarchical set abstraction layers (downsampling + local aggregation)
    - Feature propagation layers (upsampling for node predictions)
    - Node-level classifier for per-pixel predictions
    - Graph-level classifier for whole-event predictions
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_node_classes: int = 3,
        num_graph_classes: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only, pos is separate)
            num_node_classes: Number of node-level classes (pixel classification)
            num_graph_classes: Number of graph-level classes (event classification)
            dropout: Dropout probability
        """
        super().__init__()

        self.dropout = dropout

        # Set abstraction layers (hierarchical downsampling) - reduced dimensions
        # SA1: sample 50% of points, features 1->32
        self.sa1 = SetAbstractionLayer(
            ratio=0.5,
            k=16,
            in_channels=input_dim,
            out_channels=32,
        )

        # SA2: sample 50% of remaining, features 32->64
        self.sa2 = SetAbstractionLayer(ratio=0.5, k=16, in_channels=32, out_channels=64)

        # Feature propagation layers (upsampling for node predictions) - reduced
        # FP2: upsample from SA2 to SA1 level
        self.fp2 = FeaturePropagationLayer(in_channels=64 + 32, out_channels=32)

        # FP1: upsample from SA1 to original level
        self.fp1 = FeaturePropagationLayer(in_channels=32 + input_dim, out_channels=32)

        # Node-level classifier (per-pixel predictions) - simpler
        self.node_classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(dropout),
            nn.Linear(32, num_node_classes),
        )

        # Graph-level classifier (event predictions) - simpler
        self.graph_classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_graph_classes),
        )

    def forward(self, x, pos, batch):
        """
        Forward pass for pixel-level and event-level classification.

        Args:
            x: Node features [N, input_dim] (e.g., energy)
            pos: Node positions [N, 3] (x, y, z coordinates)
            batch: Batch assignment vector [N] for batched graphs

        Returns:
            node_out: Node predictions [N, num_node_classes]
            graph_out: Graph predictions [batch_size, num_graph_classes]
        """
        # Create batch tensor if not provided (for single graph case)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Store original features and positions for skip connections
        x0, pos0, batch0 = x, pos, batch

        # Hierarchical set abstraction (downsampling) - 2 layers
        x1, pos1, batch1 = self.sa1(x0, pos0, batch0)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)

        # Graph-level predictions (use deepest features)
        x_pooled = global_mean_pool(x2, batch2)
        graph_out = self.graph_classifier(x_pooled)

        # Feature propagation (upsampling for node predictions)
        x_up1 = self.fp2(x1, pos1, x2, pos2, batch1, batch2)
        x_up0 = self.fp1(x0, pos0, x_up1, pos1, batch0, batch1)

        # Node-level predictions
        node_out = self.node_classifier(x_up0)

        return node_out, graph_out


class NeutrinoPointNetPlusPlusGraphOnly(nn.Module):
    """
    PointNet++ model for graph-level classification only.
    More efficient than dual prediction since no upsampling is needed.

    Architecture:
    - Hierarchical set abstraction layers (downsampling + local aggregation)
    - Graph-level classifier for whole-event predictions
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_graph_classes: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only, pos is separate)
            num_graph_classes: Number of graph-level classes (event classification)
            dropout: Dropout probability
        """
        super().__init__()

        self.dropout = dropout

        # Set abstraction layers (hierarchical downsampling)
        # SA1: sample 50% of points, features 1->32
        self.sa1 = SetAbstractionLayer(
            ratio=0.5,
            k=16,
            in_channels=input_dim,
            out_channels=32,
        )

        # SA2: sample 50% of remaining, features 32->64
        self.sa2 = SetAbstractionLayer(ratio=0.5, k=16, in_channels=32, out_channels=64)

        # SA3: sample 50% of remaining, features 64->128
        self.sa3 = SetAbstractionLayer(
            ratio=0.5, k=16, in_channels=64, out_channels=128
        )

        # Graph-level classifier (event predictions)
        self.graph_classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_graph_classes),
        )

    def forward(self, x, pos, batch):
        """
        Forward pass for event-level classification only.

        Args:
            x: Node features [N, input_dim] (e.g., energy)
            pos: Node positions [N, 3] (x, y, z coordinates)
            batch: Batch assignment vector [N] for batched graphs

        Returns:
            graph_out: Graph predictions [batch_size, num_graph_classes]
        """
        # Create batch tensor if not provided (for single graph case)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Hierarchical set abstraction (downsampling) - 3 layers
        x1, pos1, batch1 = self.sa1(x, pos, batch)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)

        # Graph-level predictions (use deepest features)
        x_pooled = global_mean_pool(x3, batch3)
        graph_out = self.graph_classifier(x_pooled)

        return graph_out


class NeutrinoPointNet(nn.Module):
    """
    PointNet model for graph-level classification.
    Simpler than PointNet++ - no hierarchical sampling, just shared MLPs + global pooling.

    Architecture:
    - Shared MLP on all points
    - Global max pooling
    - Classification head
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_graph_classes: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only, pos is separate)
            num_graph_classes: Number of graph-level classes (event classification)
            dropout: Dropout probability
        """
        super().__init__()

        # Shared MLP on each point (processes features + positions)
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim + 3, 64),  # features + xyz
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

        # Graph-level classifier
        self.graph_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_graph_classes),
        )

    def forward(self, x, pos, batch):
        """
        Forward pass for event-level classification.

        Args:
            x: Node features [N, input_dim] (e.g., energy)
            pos: Node positions [N, 3] (x, y, z coordinates)
            batch: Batch assignment vector [N] for batched graphs

        Returns:
            graph_out: Graph predictions [batch_size, num_graph_classes]
        """
        # Create batch tensor if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Concatenate features and positions
        x_with_pos = torch.cat([x, pos], dim=-1)  # [N, input_dim + 3]

        # Apply shared MLP to each point
        point_features = self.point_mlp(x_with_pos)  # [N, 256]

        # Global max pooling over each graph in the batch
        global_features = global_max_pool(point_features, batch)  # [batch_size, 256]

        # Classification
        graph_out = self.graph_classifier(global_features)

        return graph_out
