"""
HuggingFace Deployment
Deploy trained ResNet-50 model to HuggingFace Hub.
"""

import os
import torch
import json
from pathlib import Path
from typing import Optional, Dict, List
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder
from models.resnet50 import resnet50


class HuggingFaceDeployer:
    """Deploy model to HuggingFace Hub."""

    def __init__(
        self,
        model_path: str,
        repo_name: str,
        num_classes: int = 100,
        token: Optional[str] = None
    ):
        """
        Initialize deployer.

        Args:
            model_path: Path to trained model checkpoint
            repo_name: Name of the HuggingFace repository
            num_classes: Number of classes
            token: HuggingFace API token (optional, can use HF_TOKEN env var)
        """
        self.model_path = Path(model_path)
        self.repo_name = repo_name
        self.num_classes = num_classes
        self.token = token or os.environ.get('HF_TOKEN')

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # Initialize HF API
        self.api = HfApi(token=self.token)

    def load_model_checkpoint(self) -> Dict:
        """Load model checkpoint."""
        print(f"Loading checkpoint from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location='cpu')
        return checkpoint

    def prepare_model_for_deployment(self, output_dir: str) -> str:
        """
        Prepare model files for deployment.

        Args:
            output_dir: Directory to save deployment files

        Returns:
            Path to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load checkpoint
        checkpoint = self.load_model_checkpoint()

        # Create model
        model = resnet50(num_classes=self.num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Save model in PyTorch format
        model_save_path = output_path / "pytorch_model.bin"
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model to {model_save_path}")

        # Save config
        config = {
            "model_type": "resnet50",
            "num_classes": self.num_classes,
            "architecture": "ResNet-50",
            "input_size": [3, 224, 224],
            "image_size": 224,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }

        # Add training metrics if available
        if 'metrics' in checkpoint:
            config['training_metrics'] = checkpoint['metrics']
        if 'epoch' in checkpoint:
            config['trained_epochs'] = checkpoint['epoch']

        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")

        # Create model card (README.md)
        self._create_model_card(output_path, config, checkpoint)

        # Create inference script
        self._create_inference_script(output_path)

        # Create requirements.txt
        self._create_requirements(output_path)

        print(f"\nModel prepared for deployment in {output_path}")
        return str(output_path)

    def _create_model_card(self, output_dir: Path, config: Dict, checkpoint: Dict):
        """Create model card (README.md)."""
        model_card = f"""---
tags:
- image-classification
- resnet
- imagenet
library_name: pytorch
---

# ResNet-50 ImageNet Classifier

This is a ResNet-50 model trained on a subset of ImageNet (100 classes).

## Model Description

- **Architecture**: ResNet-50
- **Number of Classes**: {config['num_classes']}
- **Input Size**: {config['image_size']}x{config['image_size']}
- **Parameters**: {config['num_parameters']:,}
"""

        if 'trained_epochs' in config:
            model_card += f"- **Training Epochs**: {config['trained_epochs']}\n"

        if 'training_metrics' in config and 'val' in config['training_metrics']:
            val_metrics = config['training_metrics']['val']
            model_card += f"\n## Performance\n\n"
            model_card += f"- **Validation Accuracy**: {val_metrics.get('acc', 0):.2f}%\n"
            if 'top5_acc' in val_metrics:
                model_card += f"- **Top-5 Accuracy**: {val_metrics.get('top5_acc', 0):.2f}%\n"
            model_card += f"- **Validation Loss**: {val_metrics.get('loss', 0):.4f}\n"

        model_card += """
## Usage

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load('pytorch_model.bin')
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open('your_image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
```

## Training

This model was trained using:
- **Optimizer**: SGD with Nesterov momentum (0.9)
- **Learning Rate Scheduler**: OneCycleLR
- **Loss**: CrossEntropyLoss with label smoothing (0.1)
- **Data Augmentation**: RandomResizedCrop, HorizontalFlip, ColorJitter, RandomErasing

## Citation

```bibtex
@article{he2015deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={arXiv preprint arXiv:1512.03385},
  year={2015}
}
```
"""

        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(model_card)
        print(f"Created model card: {readme_path}")

    def _create_inference_script(self, output_dir: Path):
        """Create inference script."""
        inference_script = """import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def load_model(model_path='pytorch_model.bin', config_path='config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = ResNet50(num_classes=config['num_classes'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def predict(image_path, model_path='pytorch_model.bin', config_path='config.json'):
    model = load_model(model_path, config_path)
    transform = get_transform()

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    return predicted_class, confidence


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    pred_class, confidence = predict(image_path)
    print(f"Predicted class: {pred_class} (confidence: {confidence:.2%})")
"""

        script_path = output_dir / "inference.py"
        with open(script_path, 'w') as f:
            f.write(inference_script)
        print(f"Created inference script: {script_path}")

    def _create_requirements(self, output_dir: Path):
        """Create requirements.txt."""
        requirements = """torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
"""
        req_path = output_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write(requirements)
        print(f"Created requirements: {req_path}")

    def push_to_hub(self, local_dir: str, private: bool = False):
        """
        Push model to HuggingFace Hub.

        Args:
            local_dir: Local directory with model files
            private: Whether to make the repo private
        """
        print(f"\nPushing to HuggingFace Hub: {self.repo_name}")

        try:
            # Create repo
            create_repo(
                self.repo_name,
                token=self.token,
                private=private,
                exist_ok=True
            )
            print(f"Repository created/verified: {self.repo_name}")

            # Upload folder
            self.api.upload_folder(
                folder_path=local_dir,
                repo_id=self.repo_name,
                repo_type="model",
                token=self.token
            )

            print(f"\nModel successfully deployed!")
            print(f"View at: https://huggingface.co/{self.repo_name}")

        except Exception as e:
            print(f"Error pushing to hub: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Deploy ResNet-50 to HuggingFace Hub")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="HuggingFace repository name (username/repo)")
    parser.add_argument("--num_classes", type=int, default=100,
                       help="Number of classes")
    parser.add_argument("--output_dir", type=str, default="./hf_model",
                       help="Directory to prepare model files")
    parser.add_argument("--token", type=str, help="HuggingFace API token")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private")
    parser.add_argument("--push", action="store_true",
                       help="Push to HuggingFace Hub")

    args = parser.parse_args()

    # Create deployer
    deployer = HuggingFaceDeployer(
        args.model_path,
        args.repo_name,
        args.num_classes,
        args.token
    )

    # Prepare model
    output_dir = deployer.prepare_model_for_deployment(args.output_dir)

    # Push to hub if requested
    if args.push:
        deployer.push_to_hub(output_dir, args.private)
    else:
        print(f"\nModel prepared in {output_dir}")
        print("To push to HuggingFace Hub, run again with --push flag")


if __name__ == "__main__":
    main()
