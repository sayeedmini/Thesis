from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class ModelConfig:
    name: str = "resnet50"
    pretrained: bool = True
    dropout: float = 0.0


def _replace_classifier(model: nn.Module, num_classes: int, dropout: float) -> nn.Module:
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Module):
        in_features = model.fc.in_features
        head = [nn.Dropout(p=dropout)] if dropout > 0 else []
        head += [nn.Linear(in_features, num_classes)]
        model.fc = nn.Sequential(*head)
        return model

    if hasattr(model, "classifier") and isinstance(getattr(model, "classifier"), nn.Module):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            last_linear_idx = None
            for i in reversed(range(len(clf))):
                if isinstance(clf[i], nn.Linear):
                    last_linear_idx = i
                    break
            if last_linear_idx is None:
                raise ValueError("Could not find Linear layer in model.classifier to replace.")
            in_features = clf[last_linear_idx].in_features
            new_layers = list(clf)
            new_layers[last_linear_idx] = nn.Linear(in_features, num_classes)
            if dropout > 0:
                new_layers.insert(last_linear_idx, nn.Dropout(p=dropout))
            model.classifier = nn.Sequential(*new_layers)
            return model

        if isinstance(clf, nn.Linear):
            in_features = clf.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, num_classes),
            )
            return model

    raise ValueError(f"Unsupported model type for head replacement: {type(model)}")


def create_model(cfg: ModelConfig, num_classes: int) -> nn.Module:
    name = cfg.name.lower()

    def try_build(builder: Callable, weights_enum_name: str):
        if not cfg.pretrained:
            return builder(weights=None)
        weights_enum = getattr(models, weights_enum_name, None)
        if weights_enum is not None and hasattr(weights_enum, "DEFAULT"):
            return builder(weights=weights_enum.DEFAULT)
        return builder(pretrained=True)

    if name == "resnet18":
        m = try_build(models.resnet18, "ResNet18_Weights")
        return _replace_classifier(m, num_classes, cfg.dropout)
    if name == "resnet50":
        m = try_build(models.resnet50, "ResNet50_Weights")
        return _replace_classifier(m, num_classes, cfg.dropout)
    if name == "efficientnet_b0":
        m = try_build(models.efficientnet_b0, "EfficientNet_B0_Weights")
        return _replace_classifier(m, num_classes, cfg.dropout)
    if name == "mobilenet_v3_small":
        m = try_build(models.mobilenet_v3_small, "MobileNet_V3_Small_Weights")
        return _replace_classifier(m, num_classes, cfg.dropout)

    raise ValueError(
        f"Unknown model '{cfg.name}'. Available: resnet18, resnet50, efficientnet_b0, mobilenet_v3_small."
    )
