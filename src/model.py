import torch.nn as nn


class DriverCNN(nn.Module):
    """CNN ringan untuk klasifikasi driver dari stack multi-band S1+S2."""

    def __init__(self, in_ch=6, n_classes=7):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))
