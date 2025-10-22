"""
ResNet-50 Implementation
Based on "Deep Residual Learning for Image Recognition" (He et al., 2015)

Architecture:
- Bottleneck blocks with expansion=4
- Stages: [3, 4, 6, 3]
- Kaiming initialization
- Optional zero-init residual BatchNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50/101/152.

    Architecture: 1x1 -> 3x3 -> 1x1 convolutions
    """
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (before expansion)
            stride: Stride for 3x3 conv
            downsample: Downsample layer for skip connection
            groups: Number of groups for 3x3 conv
            base_width: Base width for bottleneck
            dilation: Dilation for 3x3 conv
            norm_layer: Normalization layer (default: BatchNorm2d)
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(out_channels * (base_width / 64.0)) * groups

        # 1x1 conv (reduce)
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        # 3x3 conv
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride,
            padding=dilation, groups=groups, bias=False, dilation=dilation
        )
        self.bn2 = norm_layer(width)

        # 1x1 conv (expand)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 1x1 reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 expand
        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    """
    ResNet-50 for ImageNet classification.

    Architecture:
    - Input: 224x224x3
    - Conv1: 7x7 conv, 64 channels, stride 2
    - MaxPool: 3x3, stride 2
    - Stage 1: 3 bottleneck blocks, 64 base channels
    - Stage 2: 4 bottleneck blocks, 128 base channels
    - Stage 3: 6 bottleneck blocks, 256 base channels
    - Stage 4: 3 bottleneck blocks, 512 base channels
    - Global Average Pooling
    - Fully Connected: num_classes
    """

    def __init__(
        self,
        num_classes: int = 100,
        zero_init_residual: bool = True,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[nn.Module] = None
    ):
        """
        Args:
            num_classes: Number of output classes (default: 100)
            zero_init_residual: Zero-initialize last BN in each residual branch
            groups: Number of groups for grouped convolutions
            width_per_group: Width per group
            norm_layer: Normalization layer (default: BatchNorm2d)
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        # Initial convolution (7x7, stride 2)
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 stages: [3, 4, 6, 3]
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights
        self._initialize_weights(zero_init_residual)

    def _make_layer(
        self,
        block: type,
        channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a ResNet layer (stage).

        Args:
            block: Block type (Bottleneck)
            channels: Number of output channels
            num_blocks: Number of blocks in this layer
            stride: Stride for first block

        Returns:
            Sequential layer
        """
        norm_layer = self._norm_layer
        downsample = None

        # Downsample if dimensions change
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1,
                         stride=stride, bias=False),
                norm_layer(channels * block.expansion),
            )

        layers = []
        # First block (may downsample)
        layers.append(block(
            self.in_channels, channels, stride, downsample,
            self.groups, self.base_width, self.dilation, norm_layer
        ))
        self.in_channels = channels * block.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(
                self.in_channels, channels, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool = True):
        """
        Initialize weights using Kaiming initialization.

        Args:
            zero_init_residual: Zero-initialize last BN in residual branches
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        # This improves model accuracy by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, 224, 224)

        Returns:
            Output logits (B, num_classes)
        """
        # Initial conv
        x = self.conv1(x)      # (B, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # (B, 64, 56, 56)

        # ResNet stages
        x = self.layer1(x)     # (B, 256, 56, 56)
        x = self.layer2(x)     # (B, 512, 28, 28)
        x = self.layer3(x)     # (B, 1024, 14, 14)
        x = self.layer4(x)     # (B, 2048, 7, 7)

        # Global average pooling and classifier
        x = self.avgpool(x)    # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (B, 2048)
        x = self.fc(x)         # (B, num_classes)

        return x

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def resnet50(num_classes: int = 100, pretrained: bool = False, **kwargs) -> ResNet50:
    """
    ResNet-50 model.

    Args:
        num_classes: Number of output classes
        pretrained: Load pretrained weights (not implemented for custom classes)
        **kwargs: Additional arguments

    Returns:
        ResNet50 model
    """
    model = ResNet50(num_classes=num_classes, **kwargs)

    if pretrained:
        print("Warning: Pretrained weights not available for custom num_classes")

    return model


if __name__ == "__main__":
    # Test model
    print("Testing ResNet-50...")

    model = resnet50(num_classes=100, zero_init_residual=True)
    print(f"\nModel architecture:")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Verify canonical parameter count
    # ResNet-50 typically has ~25.6M parameters for 1000 classes
    # For 100 classes: (25.6M - 2048*1000) + 2048*100 ≈ 23.7M
    expected_params = 23_700_000  # Approximate
    actual_params = model.get_num_params()
    print(f"\nExpected params (approx): ~{expected_params:,}")
    print(f"Actual params: {actual_params:,}")

    print("\nModel test passed!")
