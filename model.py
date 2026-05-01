import torch
import torch.nn as nn
import torchvision.models as models


class ContinualModel(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Everything except the final FC layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features

        # Freeze entire backbone — stable features are critical for EWC and LwF to work
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

    def extract_features(self, x):
        with torch.no_grad():
            f = self.feature_extractor(x)
            return f.view(f.size(0), -1)
