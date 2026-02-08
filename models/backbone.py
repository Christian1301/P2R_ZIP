# P2R_ZIP/models/backbone.py
import torch
import torch.nn as nn
from torchvision import models

def make_backbone(name: str = "vgg16_bn", pretrained: bool = True):
    """
    Ritorna un feature extractor VGG con stride 16.
    Output channels è 512 sia per VGG16 che VGG19.
    """
    name = name.lower()
    weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
    model_class = models.vgg16_bn

    if name == "vgg19_bn":
        weights = models.VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None
        model_class = models.vgg19_bn
    elif name != "vgg16_bn":
         raise ValueError(f"Backbone {name} non supportato. Usa 'vgg16_bn' o 'vgg19_bn'.")

    model = model_class(weights=weights)
    features = model.features
    truncated_features = nn.Sequential(*list(features.children())[:43])
    out_channels = 512 

    return truncated_features, out_channels

class BackboneWrapper(nn.Module):
    def __init__(self, name="vgg16_bn", pretrained=True):
        super().__init__()
        self.body, self.out_channels = make_backbone(name, pretrained)

    def forward(self, x):
        return self.body(x)