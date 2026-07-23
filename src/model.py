"""Model builder for the damage-type classifier (transfer learning)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def build_model(num_classes: int, backbone: str = config.BACKBONE,
                pretrained: bool = True, freeze_backbone: bool = False):
    """Return a torchvision CNN with its final layer resized to num_classes.

    Transfer learning: we start from ImageNet-pretrained weights and only swap
    the classifier head. Optionally freeze the feature extractor so only the head
    trains (faster, good for very small datasets).
    """
    import torch.nn as nn
    from torchvision import models

    backbone = backbone.lower()

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        head_params = _replace_linear(net, "fc", num_classes)
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        head_params = _replace_linear(net, "fc", num_classes)
    elif backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v2(weights=weights)
        in_f = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_f, num_classes)
        head_params = list(net.classifier[1].parameters())
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_f = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_f, num_classes)
        head_params = list(net.classifier[1].parameters())
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. "
                         "Use resnet18 / resnet50 / mobilenet_v2 / efficientnet_b0.")

    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True

    return net


def _replace_linear(net, attr, num_classes):
    import torch.nn as nn
    layer = getattr(net, attr)
    new = nn.Linear(layer.in_features, num_classes)
    setattr(net, attr, new)
    return list(new.parameters())
