import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GravNetConv, global_add_pool, global_mean_pool


class NeutrinoGravNet(nn.Module):
    """
    GravNet model for graph-level neutrino event classification.
    Based on https://arxiv.org/pdf/1902.07987.pdf

    GravNet learns a spatial representation where each node has coordinates
    in a learned latent space, then aggregates features from k-nearest neighbors.

    Architecture:
    - GlobalExchange: append global mean to each node
    - 3 GravNet blocks with feature transformation MLPs
    - Global pooling for graph-level prediction
    - Final classification head
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_graph_classes: int = 3,
        dropout: float = 0.2,
        n_feature_transform: int = 16,
        out_channels: int = 16,
        space_dimensions: int = 3,
        propagate_dimensions: int = 16,
        k: int = 16,
        n_gravstack: int = 3,
        batchnorm_momentum: float = 0.05,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only)
            num_graph_classes: Number of graph-level classes (event classification)
            dropout: Dropout probability
            n_feature_transform: Hidden dimension for feature transformation MLPs
            out_channels: Output channels from each GravNet block
            space_dimensions: Dimensionality of learned spatial representation (S)
            propagate_dimensions: Dimensionality of features to propagate (F_LR)
            k: Number of nearest neighbors for aggregation
            n_gravstack: Number of GravNet blocks
            batchnorm_momentum: BatchNorm momentum (use 1-momentum from TF convention)
        """
        super().__init__()

        # Input will be [features, x, y, z] concatenated
        input_with_pos = input_dim + 3

        # After GlobalExchange, input doubles (original + global mean)
        initial_features = input_with_pos * 2

        # GravNet stack 1
        self.ft1_1 = nn.Linear(initial_features, n_feature_transform)
        self.ft1_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft1_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn1 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 2
        self.ft2_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft2_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft2_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn2 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 3
        self.ft3_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft3_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft3_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn3 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn3 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # Graph-level classification head
        # After concatenating all GravNet outputs
        concat_features = n_gravstack * out_channels

        self.graph_pooling = global_mean_pool

        self.graph_classifier = nn.Sequential(
            nn.Linear(concat_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_graph_classes),
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
        x = torch.cat([x, pos], dim=-1)  # [N, input_dim + 3]

        # GlobalExchange: append mean of all features to each node
        # This provides global context to each node
        global_mean = global_mean_pool(x, batch)  # [batch_size, input_dim + 3]
        x = torch.cat([x, global_mean[batch]], dim=-1)  # [N, 2*(input_dim + 3)]

        # List to hold outputs from each GravNet block
        feat = []

        # GravNet stack 1
        x = F.elu(self.ft1_1(x))
        x = F.elu(self.ft1_2(x))
        x = torch.tanh(self.ft1_3(x))
        x = self.gn1(x, batch)
        x = self.bn1(x)
        feat.append(x)

        # GravNet stack 2
        x = F.elu(self.ft2_1(x))
        x = F.elu(self.ft2_2(x))
        x = torch.tanh(self.ft2_3(x))
        x = self.gn2(x, batch)
        x = self.bn2(x)
        feat.append(x)

        # GravNet stack 3
        x = F.elu(self.ft3_1(x))
        x = F.elu(self.ft3_2(x))
        x = torch.tanh(self.ft3_3(x))
        x = self.gn3(x, batch)
        x = self.bn3(x)
        feat.append(x)

        # Concatenate all GravNet block outputs
        x = torch.cat(feat, dim=1)  # [N, n_gravstack * out_channels]

        # Global pooling for graph-level prediction
        x_pooled = self.graph_pooling(
            x, batch
        )  # [batch_size, n_gravstack * out_channels]

        # Graph classification
        graph_out = self.graph_classifier(x_pooled)

        return graph_out


class NeutrinoGravNetWithNodeClassification(nn.Module):
    """
    GravNet model for both graph-level neutrino event classification
    and node-level hit classification.

    Based on https://arxiv.org/pdf/1902.07987.pdf

    Architecture:
    - GlobalExchange: append global mean to each node
    - 3 GravNet blocks with feature transformation MLPs
    - Node-level classification head for hit identification
    - Global pooling for graph-level prediction
    - Graph-level classification head for event classification
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_graph_classes: int = 3,
        num_node_classes: int = 3,
        dropout: float = 0.2,
        n_feature_transform: int = 16,
        out_channels: int = 16,
        space_dimensions: int = 3,
        propagate_dimensions: int = 16,
        k: int = 16,
        n_gravstack: int = 3,
        batchnorm_momentum: float = 0.05,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only)
            num_graph_classes: Number of graph-level classes (event classification)
            num_node_classes: Number of node-level classes (hit classification)
            dropout: Dropout probability
            n_feature_transform: Hidden dimension for feature transformation MLPs
            out_channels: Output channels from each GravNet block
            space_dimensions: Dimensionality of learned spatial representation (S)
            propagate_dimensions: Dimensionality of features to propagate (F_LR)
            k: Number of nearest neighbors for aggregation
            n_gravstack: Number of GravNet blocks
            batchnorm_momentum: BatchNorm momentum (use 1-momentum from TF convention)
        """
        super().__init__()

        # Input will be [features, x, y, z] concatenated
        input_with_pos = input_dim + 3

        # After GlobalExchange, input doubles (original + global mean)
        initial_features = input_with_pos * 2

        # GravNet stack 1
        self.ft1_1 = nn.Linear(initial_features, n_feature_transform)
        self.ft1_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft1_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn1 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 2
        self.ft2_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft2_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft2_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn2 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 3
        self.ft3_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft3_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft3_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn3 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn3 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # After concatenating all GravNet outputs
        concat_features = n_gravstack * out_channels

        # Node-level classification head
        self.node_classifier = nn.Sequential(
            nn.Linear(concat_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_node_classes),
        )

        # Graph-level classification head
        self.graph_pooling = global_mean_pool

        self.graph_classifier = nn.Sequential(
            nn.Linear(concat_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_graph_classes),
        )

    def forward(self, x, pos, batch):
        """
        Forward pass for both node-level and graph-level classification.

        Args:
            x: Node features [N, input_dim] (e.g., energy)
            pos: Node positions [N, 3] (x, y, z coordinates)
            batch: Batch assignment vector [N] for batched graphs

        Returns:
            node_out: Node predictions [N, num_node_classes]
            graph_out: Graph predictions [batch_size, num_graph_classes]
        """
        # Create batch tensor if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Concatenate features and positions
        x = torch.cat([x, pos], dim=-1)  # [N, input_dim + 3]

        # GlobalExchange: append mean of all features to each node
        # This provides global context to each node
        global_mean = global_mean_pool(x, batch)  # [batch_size, input_dim + 3]
        x = torch.cat([x, global_mean[batch]], dim=-1)  # [N, 2*(input_dim + 3)]

        # List to hold outputs from each GravNet block
        feat = []

        # GravNet stack 1
        x = F.elu(self.ft1_1(x))
        x = F.elu(self.ft1_2(x))
        x = torch.tanh(self.ft1_3(x))
        x = self.gn1(x, batch)
        x = self.bn1(x)
        feat.append(x)

        # GravNet stack 2
        x = F.elu(self.ft2_1(x))
        x = F.elu(self.ft2_2(x))
        x = torch.tanh(self.ft2_3(x))
        x = self.gn2(x, batch)
        x = self.bn2(x)
        feat.append(x)

        # GravNet stack 3
        x = F.elu(self.ft3_1(x))
        x = F.elu(self.ft3_2(x))
        x = torch.tanh(self.ft3_3(x))
        x = self.gn3(x, batch)
        x = self.bn3(x)
        feat.append(x)

        # Concatenate all GravNet block outputs
        x = torch.cat(feat, dim=1)  # [N, n_gravstack * out_channels]

        # Node-level classification
        node_out = self.node_classifier(x)

        # Global pooling for graph-level prediction
        x_pooled = self.graph_pooling(
            x, batch
        )  # [batch_size, n_gravstack * out_channels]

        # Graph classification
        graph_out = self.graph_classifier(x_pooled)

        return node_out, graph_out


class NeutrinoGravNetNodesGraphFaser(nn.Module):
    """
    GravNet model for both node-level and graph-level classification with FASER spectrometer data.
    Based on https://arxiv.org/pdf/1902.07987.pdf

    GravNet learns a spatial representation where each node has coordinates
    in a learned latent space, then aggregates features from k-nearest neighbors.
    FASER spectrometer information (event-level) provides additional context for both
    node-level hit classification and graph-level event classification.

    Architecture:
    - GlobalExchange: append global mean to each node
    - 3 GravNet blocks with feature transformation MLPs
    - FASER feature processing MLP
    - Node classification head (node features + broadcast FASER features)
    - Global pooling for graph-level features
    - Graph classification head (pooled graph features + FASER features)
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_graph_classes: int = 3,
        num_node_classes: int = 3,
        faser_dim: int = 5,
        predict_charm: bool = False,
        dropout: float = 0.2,
        n_feature_transform: int = 16,
        out_channels: int = 16,
        space_dimensions: int = 3,
        propagate_dimensions: int = 16,
        k: int = 16,
        n_gravstack: int = 3,
        batchnorm_momentum: float = 0.05,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only)
            num_graph_classes: Number of graph-level classes (event classification)
            num_node_classes: Number of node-level classes (hit classification)
            faser_dim: Dimension of FASER spectrometer features (default: 5 for nhits_0, nhits_1, nhits_2, x, y)
            predict_charm: If True, add an additional classifier for charm interaction prediction
            dropout: Dropout probability
            n_feature_transform: Hidden dimension for feature transformation MLPs
            out_channels: Output channels from each GravNet block
            space_dimensions: Dimensionality of learned spatial representation (S)
            propagate_dimensions: Dimensionality of features to propagate (F_LR)
            k: Number of nearest neighbors for aggregation
            n_gravstack: Number of GravNet blocks
            batchnorm_momentum: BatchNorm momentum (use 1-momentum from TF convention)
        """
        super().__init__()

        # Store predict_charm flag
        self.predict_charm = predict_charm

        # Input will be [features, x, y, z] concatenated
        input_with_pos = input_dim + 3

        # After GlobalExchange, input doubles (original + global mean)
        initial_features = input_with_pos * 2

        # GravNet stack 1
        self.ft1_1 = nn.Linear(initial_features, n_feature_transform)
        self.ft1_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft1_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn1 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 2
        self.ft2_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft2_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft2_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn2 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 3
        self.ft3_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft3_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft3_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn3 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn3 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # After concatenating all GravNet outputs
        concat_features = n_gravstack * out_channels

        # FASER feature processing
        self.faser_mlp = nn.Sequential(
            nn.Linear(faser_dim, 8),
            nn.ReLU(),
        )

        # Node-level classification head (node features + processed FASER features)
        node_combined_features = concat_features + 8
        self.node_classifier = nn.Sequential(
            nn.Linear(node_combined_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_node_classes),
        )

        # Graph-level classification head
        self.graph_pooling = global_mean_pool

        # Graph classifier (pooled graph features + processed FASER features)
        graph_combined_features = concat_features + 8
        self.graph_classifier = nn.Sequential(
            nn.Linear(graph_combined_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_graph_classes),
        )

        # Charm classifier (binary classification: charm interaction yes/no)
        if self.predict_charm:
            self.charm_classifier = nn.Sequential(
                nn.Linear(graph_combined_features, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 2),  # Binary classification (no charm, charm)
            )
        else:
            self.charm_classifier = None

    def forward(self, x, pos, batch, x_faser):
        """
        Forward pass for both node-level and graph-level classification with FASER data.

        Args:
            x: Node features [N, input_dim] (e.g., energy)
            pos: Node positions [N, 3] (x, y, z coordinates)
            batch: Batch assignment vector [N] for batched graphs
            x_faser: FASER spectrometer features [batch_size, faser_dim]

        Returns:
            node_out: Node predictions [N, num_node_classes]
            graph_out: Graph predictions [batch_size, num_graph_classes]
            charm_out: (Optional) Charm predictions [batch_size, 2] if self.charm=True
        """
        # Create batch tensor if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Concatenate features and positions
        x = torch.cat([x, pos], dim=-1)  # [N, input_dim + 3]

        # GlobalExchange: append mean of all features to each node
        # This provides global context to each node
        global_mean = global_mean_pool(x, batch)  # [batch_size, input_dim + 3]
        x = torch.cat([x, global_mean[batch]], dim=-1)  # [N, 2*(input_dim + 3)]

        # List to hold outputs from each GravNet block
        feat = []

        # GravNet stack 1
        x = F.elu(self.ft1_1(x))
        x = F.elu(self.ft1_2(x))
        x = torch.tanh(self.ft1_3(x))
        x = self.gn1(x, batch)
        x = self.bn1(x)
        feat.append(x)

        # GravNet stack 2
        x = F.elu(self.ft2_1(x))
        x = F.elu(self.ft2_2(x))
        x = torch.tanh(self.ft2_3(x))
        x = self.gn2(x, batch)
        x = self.bn2(x)
        feat.append(x)

        # GravNet stack 3
        x = F.elu(self.ft3_1(x))
        x = F.elu(self.ft3_2(x))
        x = torch.tanh(self.ft3_3(x))
        x = self.gn3(x, batch)
        x = self.bn3(x)
        feat.append(x)

        # Concatenate all GravNet block outputs
        x = torch.cat(feat, dim=1)  # [N, n_gravstack * out_channels]

        # Process FASER features
        # When batched by PyTorch Geometric, x_faser is concatenated: [batch_size * faser_dim]
        # Reshape it to [batch_size, faser_dim]
        batch_size = batch.max().item() + 1
        x_faser_reshaped = x_faser.view(batch_size, -1)  # [batch_size, faser_dim]
        x_faser_processed = self.faser_mlp(x_faser_reshaped)  # [batch_size, 8]

        # Broadcast FASER features to all nodes for node-level classification
        x_faser_broadcast = x_faser_processed[batch]  # [N, 8]
        x_node_combined = torch.cat(
            [x, x_faser_broadcast], dim=1
        )  # [N, concat_features + 8]

        # Node-level classification
        node_out = self.node_classifier(x_node_combined)

        # Global pooling for graph-level prediction
        x_pooled = self.graph_pooling(
            x, batch
        )  # [batch_size, n_gravstack * out_channels]

        # Combine graph features and FASER features
        x_graph_combined = torch.cat([x_pooled, x_faser_processed], dim=1)

        # Graph classification
        graph_out = self.graph_classifier(x_graph_combined)

        # Charm classification (if enabled)
        if self.predict_charm:
            charm_out = self.charm_classifier(x_graph_combined)
            return node_out, graph_out, charm_out
        else:
            return node_out, graph_out


class NeutrinoGravNetFASEROld(nn.Module):
    """
    GravNet model for graph-level neutrino event classification with FASER spectrometer data.
    Based on https://arxiv.org/pdf/1902.07987.pdf

    GravNet learns a spatial representation where each node has coordinates
    in a learned latent space, then aggregates features from k-nearest neighbors.
    FASER spectrometer information (event-level) is combined with pooled graph features
    for final classification.

    Architecture:
    - GlobalExchange: append global mean to each node
    - 3 GravNet blocks with feature transformation MLPs
    - Global pooling for graph-level features
    - FASER feature processing MLP
    - Combined classification head (graph features + FASER features)
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_graph_classes: int = 3,
        faser_dim: int = 5,
        predict_charm: bool = False,
        dropout: float = 0.2,
        n_feature_transform: int = 16,
        out_channels: int = 16,
        space_dimensions: int = 3,
        propagate_dimensions: int = 16,
        k: int = 16,
        n_gravstack: int = 3,
        batchnorm_momentum: float = 0.05,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only)
            num_graph_classes: Number of graph-level classes (event classification)
            faser_dim: Dimension of FASER spectrometer features (default: 5 for nhits_0, nhits_1, nhits_2, x, y)
            predict_charm: If True, add an additional classifier for charm interaction prediction
            dropout: Dropout probability
            n_feature_transform: Hidden dimension for feature transformation MLPs
            out_channels: Output channels from each GravNet block
            space_dimensions: Dimensionality of learned spatial representation (S)
            propagate_dimensions: Dimensionality of features to propagate (F_LR)
            k: Number of nearest neighbors for aggregation
            n_gravstack: Number of GravNet blocks
            batchnorm_momentum: BatchNorm momentum (use 1-momentum from TF convention)
        """
        super().__init__()

        # Store predict_charm flag
        self.predict_charm = predict_charm

        # Input will be [features, x, y, z] concatenated
        input_with_pos = input_dim + 3

        # After GlobalExchange, input doubles (original + global mean)
        initial_features = input_with_pos * 2

        # GravNet stack 1
        self.ft1_1 = nn.Linear(initial_features, n_feature_transform)
        self.ft1_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft1_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn1 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 2
        self.ft2_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft2_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft2_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn2 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 3
        self.ft3_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft3_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft3_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn3 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn3 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # Graph-level classification head
        # After concatenating all GravNet outputs
        concat_features = n_gravstack * out_channels

        self.graph_pooling = global_mean_pool

        # FASER feature processing
        self.faser_mlp = nn.Sequential(
            nn.Linear(faser_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        # Combined classifier (graph features + processed FASER features)
        combined_features = concat_features + 8

        self.graph_classifier = nn.Sequential(
            nn.Linear(combined_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_graph_classes),
        )

        # Charm classifier (binary classification: charm interaction yes/no)
        if self.predict_charm:
            self.charm_classifier = nn.Sequential(
                nn.Linear(combined_features, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 2),  # Binary classification (no charm, charm)
            )
        else:
            self.charm_classifier = None

    def forward(self, x, pos, batch, x_faser):
        """
        Forward pass for event-level classification with FASER data.

        Args:
            x: Node features [N, input_dim] (e.g., energy)
            pos: Node positions [N, 3] (x, y, z coordinates)
            batch: Batch assignment vector [N] for batched graphs
            x_faser: FASER spectrometer features [batch_size, faser_dim]

        Returns:
            graph_out: Graph predictions [batch_size, num_graph_classes]
        """
        # Create batch tensor if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Concatenate features and positions
        x = torch.cat([x, pos], dim=-1)  # [N, input_dim + 3]

        # GlobalExchange: append mean of all features to each node
        # This provides global context to each node
        global_mean = global_mean_pool(x, batch)  # [batch_size, input_dim + 3]
        x = torch.cat([x, global_mean[batch]], dim=-1)  # [N, 2*(input_dim + 3)]

        # List to hold outputs from each GravNet block
        feat = []

        # GravNet stack 1
        x = F.elu(self.ft1_1(x))
        x = F.elu(self.ft1_2(x))
        x = torch.tanh(self.ft1_3(x))
        x = self.gn1(x, batch)
        x = self.bn1(x)
        feat.append(x)

        # GravNet stack 2
        x = F.elu(self.ft2_1(x))
        x = F.elu(self.ft2_2(x))
        x = torch.tanh(self.ft2_3(x))
        x = self.gn2(x, batch)
        x = self.bn2(x)
        feat.append(x)

        # GravNet stack 3
        x = F.elu(self.ft3_1(x))
        x = F.elu(self.ft3_2(x))
        x = torch.tanh(self.ft3_3(x))
        x = self.gn3(x, batch)
        x = self.bn3(x)
        feat.append(x)

        # Concatenate all GravNet block outputs
        x = torch.cat(feat, dim=1)  # [N, n_gravstack * out_channels]

        # Global pooling for graph-level prediction
        x_pooled = self.graph_pooling(
            x, batch
        )  # [batch_size, n_gravstack * out_channels]

        # Process FASER features
        # When batched by PyTorch Geometric, x_faser is concatenated: [batch_size * faser_dim]
        # Reshape it to [batch_size, faser_dim]
        batch_size = x_pooled.size(0)
        x_faser_reshaped = x_faser.view(batch_size, -1)  # [batch_size, faser_dim]
        x_faser_processed = self.faser_mlp(x_faser_reshaped)  # [batch_size, 8]

        # Combine graph features and FASER features
        x_combined = torch.cat([x_pooled, x_faser_processed], dim=1)

        # Graph classification
        graph_out = self.graph_classifier(x_combined)

        # Charm classification (if enabled)
        if self.predict_charm:
            charm_out = self.charm_classifier(x_combined)
            return graph_out, charm_out
        return graph_out


class NeutrinoGravNetFASER(nn.Module):
    """
    GravNet model for graph-level neutrino event classification with FASER spectrometer data.
    Based on https://arxiv.org/pdf/1902.07987.pdf

    GravNet learns a spatial representation where each node has coordinates
    in a learned latent space, then aggregates features from k-nearest neighbors.
    FASER spectrometer information (event-level) is combined with pooled graph features
    for final classification.

    Architecture:
    - GlobalExchange: append global mean to each node
    - 3 GravNet blocks with feature transformation MLPs
    - Global pooling for graph-level features
    - FASER feature processing MLP
    - Combined classification head (graph features + FASER features)
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_graph_classes: int = 3,
        faser_dim: int = 5,
        predict_charm: bool = False,
        dropout: float = 0.2,
        n_feature_transform: int = 16,
        out_channels: int = 16,
        space_dimensions: int = 3,
        propagate_dimensions: int = 16,
        k: int = 12,
        n_gravstack: int = 3,
        batchnorm_momentum: float = 0.05,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only)
            num_graph_classes: Number of graph-level classes (event classification)
            faser_dim: Dimension of FASER spectrometer features (default: 5 for nhits_0, nhits_1, nhits_2, x, y)
            predict_charm: If True, add an additional classifier for charm interaction prediction
            dropout: Dropout probability
            n_feature_transform: Hidden dimension for feature transformation MLPs
            out_channels: Output channels from each GravNet block
            space_dimensions: Dimensionality of learned spatial representation (S)
            propagate_dimensions: Dimensionality of features to propagate (F_LR)
            k: Number of nearest neighbors for aggregation
            n_gravstack: Number of GravNet blocks
            batchnorm_momentum: BatchNorm momentum (use 1-momentum from TF convention)
        """
        super().__init__()

        # Store predict_charm flag
        self.predict_charm = predict_charm

        # Input will be [features, x, y, z] concatenated
        input_with_pos = input_dim + 3

        # After GlobalExchange, input doubles (original + global mean)
        initial_features = input_with_pos * 2

        # GravNet stack 1
        self.ft1_1 = nn.Linear(initial_features, n_feature_transform)
        self.ft1_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft1_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn1 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 2
        self.ft2_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft2_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft2_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn2 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 3
        self.ft3_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft3_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft3_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn3 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn3 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # Graph-level classification head
        # After concatenating all GravNet outputs
        concat_features = n_gravstack * out_channels

        self.graph_pooling = global_mean_pool

        # FASER feature processing
        self.faser_mlp = nn.Sequential(
            nn.Linear(faser_dim, 8),
            nn.ReLU(),
        )

        # Combined classifier (graph features + processed FASER features)
        combined_features = concat_features + 8

        self.graph_classifier = nn.Sequential(
            nn.Linear(combined_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_graph_classes),
        )

        # Charm classifier (binary classification: charm interaction yes/no)
        if self.predict_charm:
            self.charm_classifier = nn.Sequential(
                nn.Linear(combined_features, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 2),  # Binary classification (no charm, charm)
            )
        else:
            self.charm_classifier = None

    def forward(self, x, pos, batch, x_faser):
        """
        Forward pass for event-level classification with FASER data.

        Args:
            x: Node features [N, input_dim] (e.g., energy)
            pos: Node positions [N, 3] (x, y, z coordinates)
            batch: Batch assignment vector [N] for batched graphs
            x_faser: FASER spectrometer features [batch_size, faser_dim]

        Returns:
            graph_out: Graph predictions [batch_size, num_graph_classes]
        """
        # Create batch tensor if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Concatenate features and positions
        x = torch.cat([x, pos], dim=-1)  # [N, input_dim + 3]

        # GlobalExchange: append mean of all features to each node
        # This provides global context to each node
        global_mean = global_mean_pool(x, batch)  # [batch_size, input_dim + 3]
        x = torch.cat([x, global_mean[batch]], dim=-1)  # [N, 2*(input_dim + 3)]

        # List to hold outputs from each GravNet block
        feat = []

        # GravNet stack 1
        x = F.elu(self.ft1_1(x))
        x = F.elu(self.ft1_2(x))
        x = torch.tanh(self.ft1_3(x))
        x = self.gn1(x, batch)
        x = self.bn1(x)
        feat.append(x)

        # GravNet stack 2
        x = F.elu(self.ft2_1(x))
        x = F.elu(self.ft2_2(x))
        x = torch.tanh(self.ft2_3(x))
        x = self.gn2(x, batch)
        x = self.bn2(x)
        feat.append(x)

        # GravNet stack 3
        x = F.elu(self.ft3_1(x))
        x = F.elu(self.ft3_2(x))
        x = torch.tanh(self.ft3_3(x))
        x = self.gn3(x, batch)
        x = self.bn3(x)
        feat.append(x)

        # Concatenate all GravNet block outputs
        x = torch.cat(feat, dim=1)  # [N, n_gravstack * out_channels]

        # Global pooling for graph-level prediction
        x_pooled = self.graph_pooling(
            x, batch
        )  # [batch_size, n_gravstack * out_channels]

        # Process FASER features
        # When batched by PyTorch Geometric, x_faser is concatenated: [batch_size * faser_dim]
        # Reshape it to [batch_size, faser_dim]
        batch_size = x_pooled.size(0)
        x_faser_reshaped = x_faser.view(batch_size, -1)  # [batch_size, faser_dim]
        x_faser_processed = self.faser_mlp(x_faser_reshaped)  # [batch_size, 8]

        # Combine graph features and FASER features
        x_combined = torch.cat([x_pooled, x_faser_processed], dim=1)

        # Graph classification
        graph_out = self.graph_classifier(x_combined)

        # Charm classification (if enabled)
        if self.predict_charm:
            charm_out = self.charm_classifier(x_combined)
            return graph_out, charm_out
        return graph_out


class NeutrinoGravNetNodesFaser(nn.Module):
    """
    GravNet model for node-level classification only with FASER spectrometer data.
    Based on https://arxiv.org/pdf/1902.07987.pdf

    GravNet learns a spatial representation where each node has coordinates
    in a learned latent space, then aggregates features from k-nearest neighbors.
    FASER spectrometer information (event-level) provides additional context for
    node-level hit classification.

    Architecture:
    - GlobalExchange: append global mean to each node
    - 3 GravNet blocks with feature transformation MLPs
    - FASER feature processing MLP
    - Node classification head (node features + broadcast FASER features)
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_node_classes: int = 3,
        faser_dim: int = 5,
        dropout: float = 0.2,
        n_feature_transform: int = 16,
        out_channels: int = 16,
        space_dimensions: int = 3,
        propagate_dimensions: int = 16,
        k: int = 12,
        n_gravstack: int = 3,
        batchnorm_momentum: float = 0.05,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only)
            num_node_classes: Number of node-level classes (hit classification)
            faser_dim: Dimension of FASER spectrometer features (default: 5 for nhits_0, nhits_1, nhits_2, x, y)
            dropout: Dropout probability
            n_feature_transform: Hidden dimension for feature transformation MLPs
            out_channels: Output channels from each GravNet block
            space_dimensions: Dimensionality of learned spatial representation (S)
            propagate_dimensions: Dimensionality of features to propagate (F_LR)
            k: Number of nearest neighbors for aggregation
            n_gravstack: Number of GravNet blocks
            batchnorm_momentum: BatchNorm momentum (use 1-momentum from TF convention)
        """
        super().__init__()

        # Input will be [features, x, y, z] concatenated
        input_with_pos = input_dim + 3

        # After GlobalExchange, input doubles (original + global mean)
        initial_features = input_with_pos * 2

        # GravNet stack 1
        self.ft1_1 = nn.Linear(initial_features, n_feature_transform)
        self.ft1_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft1_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn1 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 2
        self.ft2_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft2_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft2_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn2 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 3
        self.ft3_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft3_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft3_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn3 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn3 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # After concatenating all GravNet outputs
        concat_features = n_gravstack * out_channels

        # FASER feature processing
        self.faser_mlp = nn.Sequential(
            nn.Linear(faser_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        # Node-level classification head (node features + processed FASER features)
        node_combined_features = concat_features + 8
        self.node_classifier = nn.Sequential(
            nn.Linear(node_combined_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_node_classes),
        )

    def forward(self, x, pos, batch, x_faser):
        """
        Forward pass for node-level classification with FASER data.

        Args:
            x: Node features [N, input_dim] (e.g., energy)
            pos: Node positions [N, 3] (x, y, z coordinates)
            batch: Batch assignment vector [N] for batched graphs
            x_faser: FASER spectrometer features [batch_size, faser_dim]

        Returns:
            node_out: Node predictions [N, num_node_classes]
        """
        # Create batch tensor if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Concatenate features and positions
        x = torch.cat([x, pos], dim=-1)  # [N, input_dim + 3]

        # GlobalExchange: append mean of all features to each node
        # This provides global context to each node
        global_mean = global_mean_pool(x, batch)  # [batch_size, input_dim + 3]
        x = torch.cat([x, global_mean[batch]], dim=-1)  # [N, 2*(input_dim + 3)]

        # List to hold outputs from each GravNet block
        feat = []

        # GravNet stack 1
        x = F.elu(self.ft1_1(x))
        x = F.elu(self.ft1_2(x))
        x = torch.tanh(self.ft1_3(x))
        x = self.gn1(x, batch)
        x = self.bn1(x)
        feat.append(x)

        # GravNet stack 2
        x = F.elu(self.ft2_1(x))
        x = F.elu(self.ft2_2(x))
        x = torch.tanh(self.ft2_3(x))
        x = self.gn2(x, batch)
        x = self.bn2(x)
        feat.append(x)

        # GravNet stack 3
        x = F.elu(self.ft3_1(x))
        x = F.elu(self.ft3_2(x))
        x = torch.tanh(self.ft3_3(x))
        x = self.gn3(x, batch)
        x = self.bn3(x)
        feat.append(x)

        # Concatenate all GravNet block outputs
        x = torch.cat(feat, dim=1)  # [N, n_gravstack * out_channels]

        # Process FASER features
        # When batched by PyTorch Geometric, x_faser is concatenated: [batch_size * faser_dim]
        # Reshape it to [batch_size, faser_dim]
        batch_size = batch.max().item() + 1
        x_faser_reshaped = x_faser.view(batch_size, -1)  # [batch_size, faser_dim]
        x_faser_processed = self.faser_mlp(x_faser_reshaped)  # [batch_size, 8]

        # Broadcast FASER features to all nodes for node-level classification
        x_faser_broadcast = x_faser_processed[batch]  # [N, 8]
        x_node_combined = torch.cat(
            [x, x_faser_broadcast], dim=1
        )  # [N, concat_features + 8]

        # Node-level classification
        node_out = self.node_classifier(x_node_combined)

        return node_out


class NeutrinoGravNetRegressionFASER(nn.Module):
    """
    GravNet model for energy regression with FASER spectrometer data.
    Based on NeutrinoGravNetFASER but with a regression head instead of classification.

    Predicts continuous energy targets (E_nu, E_lepton, E_roe) per event.

    Architecture:
    - GlobalExchange: append global mean to each node
    - 3 GravNet blocks with feature transformation MLPs
    - Global pooling (mean or sum) to get graph-level representation
    - FASER feature processing MLP
    - Regression head (no final activation)
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_targets: int = 3,
        faser_dim: int = 5,
        pooling: str = "mean",
        dropout: float = 0.2,
        n_feature_transform: int = 16,
        out_channels: int = 16,
        space_dimensions: int = 3,
        propagate_dimensions: int = 16,
        k: int = 12,
        n_gravstack: int = 3,
        batchnorm_momentum: float = 0.05,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only)
            num_targets: Number of regression targets (default: 3 for E_nu, E_lepton, E_roe)
            faser_dim: Dimension of FASER spectrometer features
            pooling: Pooling method for graph-level features ("mean" or "sum")
            dropout: Dropout probability
            n_feature_transform: Hidden dimension for feature transformation MLPs
            out_channels: Output channels from each GravNet block
            space_dimensions: Dimensionality of learned spatial representation (S)
            propagate_dimensions: Dimensionality of features to propagate (F_LR)
            k: Number of nearest neighbors for aggregation
            n_gravstack: Number of GravNet blocks
            batchnorm_momentum: BatchNorm momentum
        """
        super().__init__()

        # Input will be [features, x, y, z] concatenated
        input_with_pos = input_dim + 3

        # After GlobalExchange, input doubles (original + global mean)
        initial_features = input_with_pos * 2

        # GravNet stack 1
        self.ft1_1 = nn.Linear(initial_features, n_feature_transform)
        self.ft1_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft1_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn1 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 2
        self.ft2_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft2_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft2_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn2 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 3
        self.ft3_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft3_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft3_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn3 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn3 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # Pooling for graph-level representation
        concat_features = n_gravstack * out_channels

        if pooling == "mean":
            self.graph_pooling = global_mean_pool
        elif pooling == "sum":
            self.graph_pooling = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        # FASER feature processing
        self.faser_mlp = nn.Sequential(
            nn.Linear(faser_dim, 8),
            nn.ReLU(),
        )

        # Regression head (graph features + processed FASER features)
        combined_features = concat_features + 8

        self.regression_head = nn.Sequential(
            nn.Linear(combined_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_targets),
            # No final activation — raw values for MSE loss
        )

    def forward(self, x, pos, batch, x_faser):
        """
        Forward pass for energy regression with FASER data.

        Args:
            x: Node features [N, input_dim] (e.g., energy)
            pos: Node positions [N, 3] (x, y, z coordinates)
            batch: Batch assignment vector [N] for batched graphs
            x_faser: FASER spectrometer features [batch_size, faser_dim]

        Returns:
            predictions: Regression predictions [batch_size, num_targets]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Concatenate features and positions
        x = torch.cat([x, pos], dim=-1)  # [N, input_dim + 3]

        # GlobalExchange: append mean of all features to each node
        global_mean = global_mean_pool(x, batch)  # [batch_size, input_dim + 3]
        x = torch.cat([x, global_mean[batch]], dim=-1)  # [N, 2*(input_dim + 3)]

        feat = []

        # GravNet stack 1
        x = F.elu(self.ft1_1(x))
        x = F.elu(self.ft1_2(x))
        x = torch.tanh(self.ft1_3(x))
        x = self.gn1(x, batch)
        x = self.bn1(x)
        feat.append(x)

        # GravNet stack 2
        x = F.elu(self.ft2_1(x))
        x = F.elu(self.ft2_2(x))
        x = torch.tanh(self.ft2_3(x))
        x = self.gn2(x, batch)
        x = self.bn2(x)
        feat.append(x)

        # GravNet stack 3
        x = F.elu(self.ft3_1(x))
        x = F.elu(self.ft3_2(x))
        x = torch.tanh(self.ft3_3(x))
        x = self.gn3(x, batch)
        x = self.bn3(x)
        feat.append(x)

        # Concatenate all GravNet block outputs
        x = torch.cat(feat, dim=1)  # [N, n_gravstack * out_channels]

        # Global pooling for graph-level prediction
        x_pooled = self.graph_pooling(x, batch)  # [batch_size, n_gravstack * out_channels]

        # Process FASER features
        # PyG concatenates graph-level [5] tensors flat to [batch_size*5]
        batch_size = x_pooled.size(0)
        x_faser_reshaped = x_faser.view(batch_size, -1)  # [batch_size, faser_dim]
        x_faser_processed = self.faser_mlp(x_faser_reshaped)  # [batch_size, 8]

        # Combine graph features and FASER features
        x_combined = torch.cat([x_pooled, x_faser_processed], dim=1)

        # Regression prediction
        return self.regression_head(x_combined)
