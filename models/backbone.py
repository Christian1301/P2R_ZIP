# P2R_ZIP/models/backbone.py
import torch
import torch.nn as nn
from torchvision import models

def make_backbone(name: str = "vgg16_bn", pretrained: bool = True, out_stride=8):
    """
    Ritorna un feature extractor che produce una mappa conv finale.
    out_channels: 512 per VGG16-BN, 2048 per ResNet50.
    """
    name = name.lower()
    if name == "vgg16_bn":
        m = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None)
        features = m.features  # Sequential
        # taglia via maxpool finale se vuoi stride 8
        # VGG16-BN stride tipico: 32. Riduciamo togliendo l'ultimo pooling.
        # Tenere fino a layer 33 (prima di MaxPool5) per stride 16; per stride 8 serve dilatazione -> qui teniamo stride 16.
        truncated = nn.Sequential(*list(features.children())[:-6])  # cerca un buon trade-off
        out_channels = 512
        return truncated, out_channels
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # prendiamo layer fino a C4 per stride 16
        stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        c2 = m.layer1
        c3 = m.layer2
        c4 = m.layer3
        backbone = nn.Sequential(stem, c2, c3, c4)
        out_channels = 1024
        return backbone, out_channels
    else:
        raise ValueError(f"Backbone {name} non supportato.")

class BackboneWrapper(nn.Module):
    def __init__(self, name="vgg16_bn", pretrained=True):
        super().__init__()
        self.body, self.out_channels = make_backbone(name, pretrained)

    def forward(self, x):
        return self.body(x)
