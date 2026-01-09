# P2R_ZIP/models/backbone.py
import torch
import torch.nn as nn
from torchvision import models

def make_backbone(name: str = "vgg16_bn", pretrained: bool = True):
    """
    Ritorna un feature extractor VGG con stride 16.
    Gestisce correttamente i layer di taglio per VGG16 (idx 43) e VGG19 (idx 52).
    Output channels è 512 per entrambi.
    """
    name = name.lower()
    
    if name == "vgg16_bn":
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16_bn(weights=weights)
        cut_index = 43 # VGG16: Taglia prima dell'ultimo MaxPool
        
    elif name == "vgg19_bn":
        weights = models.VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg19_bn(weights=weights)
        cut_index = 52 # VGG19: Ha più layer, il taglio corretto è 52
        
    else:
         raise ValueError(f"Backbone {name} non supportato. Usa 'vgg16_bn' o 'vgg19_bn'.")

    features = model.features
    # Seleziona tutti i layer fino all'ultimo blocco convoluzionale escluso il pooling finale
    truncated_features = nn.Sequential(*list(features.children())[:cut_index])
    out_channels = 512 

    return truncated_features, out_channels

class BackboneWrapper(nn.Module):
    def __init__(self, name="vgg16_bn", pretrained=True):
        super().__init__()
        self.body, self.out_channels = make_backbone(name, pretrained)

    def forward(self, x):
        return self.body(x)