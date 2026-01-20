import torch
import torch.nn as nn
import torch.nn.functional as F


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
