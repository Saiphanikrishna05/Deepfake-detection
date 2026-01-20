import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetector(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        return self.sigmoid(x)
