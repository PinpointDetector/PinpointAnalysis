import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import (
    DynamicEdgeConv,
    EdgeConv,
    GATv2Conv,
    GCNConv,
    GraphNorm,
    SAGEConv,
    global_mean_pool,
    knn_graph,
)


class CNN3DNetwork(nn.Module):
    """3D CNN for neutrino interaction classification"""

    def __init__(self, num_classes: int = 3, feature_dim: int = 128):
        super(CNN3DNetwork, self).__init__()

        # 3D Convolutional layers - reduced channels
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)

        # 3D Batch normalization
        self.bn1 = nn.BatchNorm3d(8)
        self.bn2 = nn.BatchNorm3d(16)
        self.bn3 = nn.BatchNorm3d(32)

        # 3D Pooling and dropout
        self.pool = nn.MaxPool3d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.dropout3d = nn.Dropout3d(0.3)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers - simplified
        self.feature_fc = nn.Linear(32 * 4 * 4 * 4, feature_dim)  # 2,048

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout3d(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout3d(x)

        x = self.adaptive_pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3d(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.feature_fc(x))
        x = self.dropout(x)

        return x


class CNNProjectionNetwork(nn.Module):
    """CNN for processing a single 2D projection"""

    def __init__(self, feature_dim: int = 64):
        super(CNNProjectionNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout2d = nn.Dropout2d(0.5)
        self.dropout = nn.Dropout(0.5)

        # Feature extraction
        self.feature_fc = nn.Linear(32 * 4 * 4, feature_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)

        x = self.adaptive_pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.feature_fc(x))
        x = self.dropout(x)

        return x


class ClassifierProjectionCNN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        num_classes: int = 3,
    ):
        super(ClassifierProjectionCNN, self).__init__()
        self.zx_cnn = CNNProjectionNetwork(feature_dim=feature_dim)
        self.zy_cnn = CNNProjectionNetwork(feature_dim=feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # Combine both projection features
            nn.ReLU(),
            nn.Dropout(0.25),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        zx_proj, zy_proj = x

        zx_features = self.zx_cnn(zx_proj)
        zy_features = self.zy_cnn(zy_proj)

        combined_features = torch.cat([zx_features, zy_features], dim=1)

        output = self.classifier(combined_features)

        return output


class ClassifierProjectionCNNFaser(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        num_classes: int = 3,
    ):
        super(ClassifierProjectionCNNFaser, self).__init__()
        self.zx_cnn = CNNProjectionNetwork(feature_dim=feature_dim)
        self.zy_cnn = CNNProjectionNetwork(feature_dim=feature_dim)

        self.faser_mlp = nn.Sequential(
            nn.Linear(3, 32),  # Combine both projection features
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),  # Combine both projection features
            nn.ReLU(),
            nn.Dropout(0.25),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        zx_proj, zy_proj, x_faser = x

        zx_features = self.zx_cnn(zx_proj)
        zy_features = self.zy_cnn(zy_proj)
        x_faser = self.faser_mlp(x_faser)

        combined_features = torch.cat([zx_features, zy_features, x_faser], dim=1)

        output = self.classifier(combined_features)

        return output


class RegressionCNN(nn.Module):
    def __init__(self, feature_dim: int = 64):
        super(RegressionCNN, self).__init__()
        self.zx_cnn = CNNProjectionNetwork(feature_dim=feature_dim)
        self.zy_cnn = CNNProjectionNetwork(feature_dim=feature_dim)

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 1, 128),  # Combine both projection features
            nn.ReLU(),
            nn.Dropout(0.25),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        zx_proj, _ = x

        zx_features = self.zx_cnn(zx_proj)
        # zy_features = self.zy_cnn(zy_proj)

        # combined_features = torch.cat([zx_features, zy_features], dim=1)

        # output = self.regressor(combined_features)
        output = self.regressor(zx_features)

        return output.squeeze(-1)  # Return shape (batch,) instead of (batch, 1)


class GraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_classes=3):
        super(GraphClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings through GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Graph-level pooling (aggregate node features to graph)
        x = global_mean_pool(x, batch)

        # Final classification
        x = self.lin(x)
        return x


class GraphClassifierWithRegression(torch.nn.Module):
    """
    GCN for graph-level classification and energy regression.
    Multi-task learning: predicts both class (0, 1, 2) and energy value.
    """

    def __init__(self, num_node_features, hidden_channels=64, num_classes=3):
        super(GraphClassifierWithRegression, self).__init__()
        # Shared GCN layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # Classification head
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

        # Regression head for energy
        self.regressor = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # Shared node embeddings through GCN layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Graph-level pooling (aggregate node features to graph)
        x = global_mean_pool(x, batch)

        # Classification output
        class_logits = self.classifier(x)
        class_output = F.log_softmax(class_logits, dim=1)

        # Regression output (energy)
        energy_output = self.regressor(x).squeeze(-1)

        return class_output, energy_output


class DGCClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 32,
        k: int = 128,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        """
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of DynamicEdgeConv layers
            k: Number of nearest neighbors for dynamic graph construction
            dropout: Dropout probability
            num_classes: Number of output classes (3 for binary)
        """
        super().__init__()

        self.k = k
        self.dropout = dropout

        def make_mlp(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.conv0 = GCNConv(input_dim, hidden_dim)

        self.dconv1 = DynamicEdgeConv(
            make_mlp(hidden_dim * 2, hidden_dim), k=k, aggr="max"
        )
        self.graph_norm1 = GraphNorm(hidden_dim)

        self.dconv2 = DynamicEdgeConv(
            make_mlp(hidden_dim * 2, hidden_dim), k=k, aggr="max"
        )
        self.graph_norm2 = GraphNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = self.conv0(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.dconv1(x, batch)
        x = self.graph_norm1(x, batch)

        x = self.dconv2(x, batch)
        x = self.graph_norm2(x, batch)

        x_pooled = global_mean_pool(x, batch)

        return self.classifier(x_pooled)


class SimpleGCN(nn.Module):
    """
    Simple, robust GCN for node classification.
    Uses conservative architecture to avoid numerical issues.
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))  # Use BatchNorm1d like MuonHitNet

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer (no activation, smaller dim)
        self.convs.append(GCNConv(hidden_dim, hidden_dim // 2))
        self.norms.append(nn.BatchNorm1d(hidden_dim // 2))

        self.dropout = dropout

        # Simple classifier
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x, edge_index, batch=None):
        # Clamp input to prevent extreme values
        x = torch.clamp(x, min=-10, max=10)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x if i > 0 and x.shape[-1] == self.convs[i].out_channels else None

            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection with dimension check
            if x_res is not None:
                x = x + x_res

            # Clamp after each layer to prevent explosion
            x = torch.clamp(x, min=-10, max=10)

        return self.classifier(x)


class ParticleHitSAGE(nn.Module):
    """
    GraphSAGE-based model for node classification.
    Uses gradual dimension expansion for stability.
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Gradual dimension expansion for stability
        mid_dim = hidden_dim // 2

        # First layer with dimension reduction
        self.convs.append(SAGEConv(input_dim, mid_dim))
        self.norms.append(nn.BatchNorm1d(mid_dim))

        # Second layer to expand
        self.convs.append(SAGEConv(mid_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(max(0, num_layers - 2)):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

        # Classifier with BatchNorm1d for stability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch=None):
        # Clamp input to prevent extreme values
        x = torch.clamp(x, min=-10, max=10)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x if i > 0 and x.shape[-1] == self.convs[i].out_channels else None

            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection with dimension check
            if x_res is not None:
                x = x + x_res

            # Clamp after each layer to prevent explosion
            x = torch.clamp(x, min=-10, max=10)

        return self.classifier(x)


class ParticleHitSAGEClassifier(nn.Module):
    """
    GraphSAGE-based model for both node and graph classification.
    Uses gradual dimension expansion for stability.

    Architecture:
    - GraphSAGE layers: Spatial feature aggregation with neighborhood sampling
    - Node-level classification for particle type (electron/muon/other)
    - Graph-level classification for overall event type
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., x, y, z, n_hits_norm, faser_x, faser_y, nhits_0, nhits_1, nhits_2)
            hidden_dim: Hidden layer dimension
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            num_classes: Number of output classes (3 for electron/muon/other)
        """
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Gradual dimension expansion for stability
        mid_dim = hidden_dim // 2

        # First layer with dimension reduction
        self.convs.append(SAGEConv(input_dim, mid_dim))
        self.norms.append(nn.BatchNorm1d(mid_dim))

        # Second layer to expand
        self.convs.append(SAGEConv(mid_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(max(0, num_layers - 2)):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

        # Node-level classifier with BatchNorm1d for stability
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Graph-level classifier (for aggregated node features)
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for node-level and graph-level particle classification.

        Args:
            x: Node features [N, input_dim]
            edge_index: Pre-built graph edges [2, E] (static k-NN graph)
            batch: Batch assignment vector [N] for batched graphs

        Returns:
            node_out: Node predictions [N, num_classes]
            graph_out: Graph predictions [batch_size, num_classes]
        """
        # Create batch tensor if not provided (for single graph case)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Clamp input to prevent extreme values
        x = torch.clamp(x, min=-10, max=10)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x if i > 0 and x.shape[-1] == self.convs[i].out_channels else None

            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection with dimension check
            if x_res is not None:
                x = x + x_res

            # Clamp after each layer to prevent explosion
            x = torch.clamp(x, min=-10, max=10)

        # Node-level classification
        node_out = self.node_classifier(x)

        # Graph-level classification (aggregate nodes per graph)
        x_pooled = global_mean_pool(x, batch)
        graph_out = self.graph_classifier(x_pooled)

        return node_out, graph_out


class MuonHitNet(nn.Module):
    """
    Hybrid network for particle hit classification in detector.

    Architecture:
    - DynamicEdgeConv: Initial adaptive feature extraction (k-NN in feature space)
    - Static EdgeConv: Spatial feature aggregation on pre-built graph
    - GAT layers: Attention-based propagation to learn track vs shower patterns
    - Node-level classification for particle type (electron/muon/other)
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 32,
        num_gat_layers: int = 2,
        k: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., x, y, z, n_hits_norm, faser_x, faser_y, nhits_0, nhits_1, nhits_2)
            hidden_dim: Hidden layer dimension
            num_gat_layers: Number of GAT layers
            k: Number of nearest neighbors for DynamicEdgeConv
            heads: Number of attention heads in GAT
            dropout: Dropout probability
            num_classes: Number of output classes (3 for electron/muon/other)
        """
        super().__init__()

        self.k = k
        self.dropout = dropout

        # Helper function for MLPs
        def make_mlp(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # Single DynamicEdgeConv layer for initial adaptive feature learning
        self.dynamic_conv = DynamicEdgeConv(
            make_mlp(input_dim * 2, hidden_dim), k=k, aggr="max"
        )
        self.dynamic_norm = GraphNorm(hidden_dim)

        # Static EdgeConv for spatial feature extraction on pre-built graph
        self.edge_conv = EdgeConv(make_mlp(hidden_dim * 2, hidden_dim), aggr="max")
        self.edge_norm = nn.BatchNorm1d(hidden_dim)

        # Multi-head Graph Attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        for i in range(num_gat_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            self.gat_layers.append(
                GATv2Conv(
                    in_channels,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_gat_layers - 1 else False,
                )
            )
            out_channels = hidden_dim * heads if i < num_gat_layers - 1 else hidden_dim
            self.gat_norms.append(nn.BatchNorm1d(out_channels))

        # Node-level classifier
        final_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.BatchNorm1d(final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for node-level particle classification.

        Args:
            x: Node features [N, input_dim]
            edge_index: Pre-built graph edges [2, E] (static k-NN graph)
            batch: Batch assignment vector [N] for batched graphs

        Returns:
            Node predictions [N, num_classes]
        """
        # Create batch tensor if not provided (for single graph case)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial adaptive feature learning with DynamicEdgeConv
        x = self.dynamic_conv(x, batch)
        x = self.dynamic_norm(x, batch)
        x = F.relu(x)

        # Static spatial feature extraction with EdgeConv
        x_res = x
        x = self.edge_conv(x, edge_index)
        x = self.edge_norm(x)
        x = F.relu(x)
        x = x + x_res  # Residual connection

        # Attention-based propagation with GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            x_res = x
            x = gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x_res.shape[-1] == x.shape[-1]:
                x = x + x_res

        # Node-level classification
        out = self.classifier(x)

        return out


class StaticEdgeHitNet(nn.Module):
    """
    Static EdgeConv-based network for particle hit classification.

    Architecture:
    - Multiple EdgeConv layers: Spatial feature aggregation on pre-built graph
    - GAT layers: Attention-based propagation to learn track vs shower patterns
    - Node-level classification for particle type (electron/muon/other)

    Unlike MuonHitNet, this model uses only static EdgeConv (no DynamicEdgeConv),
    operating entirely on the pre-constructed k-NN graph.
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 32,
        num_edge_layers: int = 2,
        num_gat_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., x, y, z, n_hits_norm, faser_x, faser_y, nhits_0, nhits_1, nhits_2)
            hidden_dim: Hidden layer dimension
            num_edge_layers: Number of EdgeConv layers
            num_gat_layers: Number of GAT layers
            heads: Number of attention heads in GAT
            dropout: Dropout probability
            num_classes: Number of output classes (3 for electron/muon/other)
        """
        super().__init__()

        self.dropout = dropout

        # Helper function for MLPs
        def make_mlp(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # Multiple EdgeConv layers for spatial feature extraction
        self.edge_convs = nn.ModuleList()
        self.edge_norms = nn.ModuleList()

        # First EdgeConv layer (input -> hidden)
        self.edge_convs.append(
            EdgeConv(make_mlp(input_dim * 2, hidden_dim), aggr="max")
        )
        self.edge_norms.append(nn.BatchNorm1d(hidden_dim))

        # Additional EdgeConv layers
        for _ in range(num_edge_layers - 1):
            self.edge_convs.append(
                EdgeConv(make_mlp(hidden_dim * 2, hidden_dim), aggr="max")
            )
            self.edge_norms.append(nn.BatchNorm1d(hidden_dim))

        # Multi-head Graph Attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        for i in range(num_gat_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            self.gat_layers.append(
                GATv2Conv(
                    in_channels,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_gat_layers - 1 else False,
                )
            )
            out_channels = hidden_dim * heads if i < num_gat_layers - 1 else hidden_dim
            self.gat_norms.append(nn.BatchNorm1d(out_channels))

        # Node-level classifier
        final_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.BatchNorm1d(final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for node-level particle classification.

        Args:
            x: Node features [N, input_dim]
            edge_index: Pre-built graph edges [2, E] (static k-NN graph)
            batch: Batch assignment vector [N] for batched graphs (unused here)

        Returns:
            Node predictions [N, num_classes]
        """
        # Clamp input to prevent extreme values
        x = torch.clamp(x, min=-10, max=10)

        # Process through EdgeConv layers
        for i, (edge_conv, norm) in enumerate(zip(self.edge_convs, self.edge_norms)):
            x_res = (
                x if i > 0 else None
            )  # Skip residual on first layer (dimension mismatch)

            x = edge_conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x_res is not None and x_res.shape[-1] == x.shape[-1]:
                x = x + x_res

            # Clamp after each layer
            x = torch.clamp(x, min=-10, max=10)

        # Attention-based propagation with GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            x_res = x
            x = gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x_res.shape[-1] == x.shape[-1]:
                x = x + x_res

            # Clamp after each layer
            x = torch.clamp(x, min=-10, max=10)

        # Node-level classification
        out = self.classifier(x)

        return out


class MuonHitNetClassifier(nn.Module):
    """
    Hybrid network for particle hit classification in detector.

    Architecture:
    - DynamicEdgeConv: Initial adaptive feature extraction (k-NN in feature space)
    - Static EdgeConv: Spatial feature aggregation on pre-built graph
    - GAT layers: Attention-based propagation to learn track vs shower patterns
    - Node-level classification for particle type (electron/muon/other)
    - Graph-level classification for overall event type
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 32,
        num_gat_layers: int = 2,
        k: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., x, y, z, n_hits_norm, faser_x, faser_y, nhits_0, nhits_1, nhits_2)
            hidden_dim: Hidden layer dimension
            num_gat_layers: Number of GAT layers
            k: Number of nearest neighbors for DynamicEdgeConv
            heads: Number of attention heads in GAT
            dropout: Dropout probability
            num_classes: Number of output classes (3 for electron/muon/other)
        """
        super().__init__()

        self.k = k
        self.dropout = dropout

        # Helper function for MLPs
        def make_mlp(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # Single DynamicEdgeConv layer for initial adaptive feature learning
        self.dynamic_conv = DynamicEdgeConv(
            make_mlp(input_dim * 2, hidden_dim), k=k, aggr="max"
        )
        self.dynamic_norm = GraphNorm(hidden_dim)

        # Static EdgeConv for spatial feature extraction on pre-built graph
        self.edge_conv = EdgeConv(make_mlp(hidden_dim * 2, hidden_dim), aggr="max")
        self.edge_norm = nn.LayerNorm(hidden_dim)

        # Multi-head Graph Attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        for i in range(num_gat_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            self.gat_layers.append(
                GATv2Conv(
                    in_channels,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_gat_layers - 1 else False,
                )
            )
            out_channels = hidden_dim * heads if i < num_gat_layers - 1 else hidden_dim
            self.gat_norms.append(nn.LayerNorm(out_channels))

        # Node-level classifier
        final_dim = hidden_dim
        self.node_classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.LayerNorm(final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.LayerNorm(final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes),
        )

        # Graph-level classifier (for aggregated node features)
        self.graph_classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.LayerNorm(final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.LayerNorm(final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for node-level and graph-level particle classification.

        Args:
            x: Node features [N, input_dim]
            edge_index: Pre-built graph edges [2, E] (static k-NN graph)
            batch: Batch assignment vector [N] for batched graphs

        Returns:
            node_out: Node predictions [N, num_classes]
            graph_out: Graph predictions [batch_size, num_classes]
        """
        # Create batch tensor if not provided (for single graph case)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial adaptive feature learning with DynamicEdgeConv
        x = self.dynamic_conv(x, batch)
        x = self.dynamic_norm(x, batch)
        x = F.relu(x)

        # Static spatial feature extraction with EdgeConv
        x_res = x
        x = self.edge_conv(x, edge_index)
        x = self.edge_norm(x)
        x = F.relu(x)
        x = x + x_res  # Residual connection

        # Attention-based propagation with GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            x_res = x
            x = gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x_res.shape[-1] == x.shape[-1]:
                x = x + x_res

        # Node-level classification
        node_out = self.node_classifier(x)

        # Graph-level classification (aggregate nodes per graph)
        x_pooled = global_mean_pool(x, batch)
        graph_out = self.graph_classifier(x_pooled)

        return node_out, graph_out


class GraphClassifier(nn.Module):
    """
    Hybrid network for particle hit classification in detector.

    Architecture:
    - DynamicEdgeConv: Initial adaptive feature extraction (k-NN in feature space)
    - Static EdgeConv: Spatial feature aggregation on pre-built graph
    - GAT layers: Attention-based propagation to learn track vs shower patterns
    - Graph-level classification for overall event type
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 32,
        num_gat_layers: int = 2,
        k: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., x, y, z, n_hits_norm, faser_x, faser_y, nhits_0, nhits_1, nhits_2)
            hidden_dim: Hidden layer dimension
            num_gat_layers: Number of GAT layers
            k: Number of nearest neighbors for DynamicEdgeConv
            heads: Number of attention heads in GAT
            dropout: Dropout probability
            num_classes: Number of output classes (3 for electron/muon/other)
        """
        super().__init__()

        self.k = k
        self.dropout = dropout

        # Helper function for MLPs
        def make_mlp(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # Single DynamicEdgeConv layer for initial adaptive feature learning
        self.dynamic_conv = DynamicEdgeConv(
            make_mlp(input_dim * 2, hidden_dim), k=k, aggr="max"
        )
        self.dynamic_norm = GraphNorm(hidden_dim)

        # Static EdgeConv for spatial feature extraction on pre-built graph
        self.edge_conv = EdgeConv(make_mlp(hidden_dim * 2, hidden_dim), aggr="max")
        self.edge_norm = nn.LayerNorm(hidden_dim)

        # Multi-head Graph Attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        for i in range(num_gat_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            self.gat_layers.append(
                GATv2Conv(
                    in_channels,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_gat_layers - 1 else False,
                )
            )
            out_channels = hidden_dim * heads if i < num_gat_layers - 1 else hidden_dim
            self.gat_norms.append(nn.LayerNorm(out_channels))

        # Graph-level classifier (for aggregated node features)
        final_dim = hidden_dim
        self.graph_classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.LayerNorm(final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.LayerNorm(final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for graph-level particle classification.

        Args:
            x: Node features [N, input_dim]
            edge_index: Pre-built graph edges [2, E] (static k-NN graph)
            batch: Batch assignment vector [N] for batched graphs

        Returns:
            graph_out: Graph predictions [batch_size, num_classes]
        """
        # Create batch tensor if not provided (for single graph case)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial adaptive feature learning with DynamicEdgeConv
        x = self.dynamic_conv(x, batch)
        x = self.dynamic_norm(x, batch)
        x = F.relu(x)

        # Static spatial feature extraction with EdgeConv
        x_res = x
        x = self.edge_conv(x, edge_index)
        x = self.edge_norm(x)
        x = F.relu(x)
        x = x + x_res  # Residual connection

        # Attention-based propagation with GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            x_res = x
            x = gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x_res.shape[-1] == x.shape[-1]:
                x = x + x_res

        # Graph-level classification (aggregate nodes per graph)
        x_pooled = global_mean_pool(x, batch)
        graph_out = self.graph_classifier(x_pooled)

        return graph_out
