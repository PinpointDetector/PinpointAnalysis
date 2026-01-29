import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import (
    DynamicEdgeConv,
    GCNConv,
    GraphNorm,
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


class MuonHitNet(nn.Module):
    """
    Dynamic Edge Convolution Network for muon track identification in detector hits.

    Uses DynamicEdgeConv to dynamically compute k-nearest neighbors in feature space
    at each layer, which is ideal for identifying coherent tracks (muons) vs showers.

    Architecture:
    - Input: [x, y, z, n_particles] per hit
    - Multiple DynamicEdgeConv layers that learn spatial-feature relationships
    - Node-level binary classification (muon vs non-muon)
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        k: int = 16,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        """
        Args:
            input_dim: Input feature dimension (x, y, z, n_particles)
            hidden_dim: Hidden layer dimension
            num_layers: Number of DynamicEdgeConv layers
            k: Number of nearest neighbors for dynamic graph construction
            dropout: Dropout probability
            num_classes: Number of output classes (2 for binary)
        """
        super().__init__()

        self.k = k
        self.num_layers = num_layers
        self.dropout = dropout

        # Edge convolution MLP for learning spatial relationships
        def make_mlp(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # First layer: input_dim * 2 because EdgeConv concatenates [x_i, x_j - x_i]
        self.conv1 = DynamicEdgeConv(
            make_mlp(input_dim * 2, hidden_dim), k=k, aggr="max"
        )

        # Additional layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                DynamicEdgeConv(make_mlp(hidden_dim * 2, hidden_dim), k=k, aggr="max")
            )

        # Graph normalization after each conv layer
        self.graph_norms = nn.ModuleList(
            [GraphNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Node-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass for node-level muon classification.

        Args:
            x: Node features [N, input_dim]
            edge_index: Not used (DynamicEdgeConv computes edges dynamically)
            batch: Batch assignment vector [N] for batched graphs

        Returns:
            Node predictions [N, num_classes]
        """
        # First dynamic edge conv layer
        x = self.conv1(x, batch)
        x = self.graph_norms[0](x, batch)

        # Additional layers with residual connections
        for i, conv in enumerate(self.conv_layers):
            x_res = x
            x = conv(x, batch)
            x = self.graph_norms[i + 1](x, batch)
            x = x + x_res  # Residual connection

        # Node-level classification
        out = self.classifier(x)

        return out
