"""The model architectures the training studio offers."""

from __future__ import annotations

ARCHITECTURES = {
    "small-cnn": {
        "label": "Small CNN",
        "description": "Four convolution blocks, trained from scratch. Fast, and a good start for small images.",
        "image_size": 64,
        "batch_size": 128,
        "learning_rate": 0.001,
        "needs": None,
        "pretrained_option": False,
    },
    "resnet18": {
        "label": "ResNet-18",
        "description": "A deeper network. Can start from ImageNet weights, which helps with fewer images.",
        "image_size": 128,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "needs": "torchvision",
        "pretrained_option": True,
    },
}
IMAGE_SIZES = (28, 32, 64, 96, 128, 160, 224)


def torchvision_available() -> bool:
    try:
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


def options() -> dict:
    have_tv = torchvision_available()
    return {
        "architectures": [
            {"id": key, **{k: v for k, v in spec.items() if k != "needs"},
             "available": spec["needs"] != "torchvision" or have_tv,
             "unavailable_reason": None if spec["needs"] != "torchvision" or have_tv
             else "Install torchvision to use this architecture."}
            for key, spec in ARCHITECTURES.items()
        ],
        "image_sizes": list(IMAGE_SIZES),
    }


def build(architecture: str, num_classes: int, in_channels: int, pretrained: bool = False):
    import torch.nn as nn

    if architecture == "small-cnn":
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        return nn.Sequential(
            block(in_channels, 32), block(32, 64), block(64, 128), block(128, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes),
        )
    if architecture == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture {architecture!r}")
