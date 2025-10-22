# ResNet-50 ImageNet Training Pipeline

A complete, production-ready training pipeline for ResNet-50 on ImageNet-1k (100 classes subset). This project implements state-of-the-art training practices including OneCycleLR scheduling, mixed precision training, and comprehensive logging.

## Features

- **Model**: ResNet-50 with bottleneck blocks (3-4-6-3 architecture)
- **Training**: SGD + Nesterov momentum with OneCycleLR scheduler
- **Data**: Automated HuggingFace ImageNet-1k subset extraction
- **Optimization**: Mixed precision training, gradient clipping, label smoothing
- **Utilities**: LR Finder, AWS S3 integration, HuggingFace deployment
- **Logging**: Comprehensive metrics tracking with JSON and CSV outputs

## Project Structure

```
.
├── data/                   # Data loading and transforms
│   ├── dataset.py         # ImageNet dataset class
│   └── transforms.py      # Data augmentation pipeline
├── models/                 # Model architectures
│   └── resnet50.py        # ResNet-50 implementation
├── training/               # Training components
│   ├── config.py          # Configuration management
│   ├── trainer.py         # Main training loop
│   └── metrics.py         # Metrics tracking
├── utils/                  # Utility functions
│   ├── lr_finder.py       # Learning rate finder
│   ├── checkpointing.py   # Model checkpointing
│   ├── logger.py          # Metrics logging
│   └── sample_images.py   # Image sampling utility
├── scripts/                # Executable scripts
│   ├── download_data.py   # Download ImageNet subset
│   ├── upload_to_s3.py    # Upload data to S3
│   ├── train.py           # Main training script
│   └── find_lr.py         # LR finder script
├── deployment/             # Deployment utilities
│   └── hf_deploy.py       # HuggingFace deployment
├── checkpoints/            # Model checkpoints (created during training)
├── logs/                   # Training logs (created during training)
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 50GB+ free disk space (for dataset)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/S9EVA4.git
cd S9EVA4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download Dataset

Download 100 classes from HuggingFace ImageNet-1k:

```bash
python scripts/download_data.py \
    --output_dir ./imagenet_subset \
    --num_classes 100 \
    --max_train_per_class 1300 \
    --max_val_per_class 50 \
    --seed 42
```

**Note**: First download may take 1-2 hours depending on your internet speed.

### 2. Find Optimal Learning Rate

Run the LR Finder to determine optimal learning rate:

```bash
python scripts/find_lr.py \
    --data_dir ./imagenet_subset \
    --num_classes 100 \
    --batch_size 256 \
    --num_iter 100 \
    --plot_path ./lr_finder_plot.png
```

This will generate:
- `lr_finder_plot.png`: Loss vs. learning rate plot
- `suggested_lr.txt`: Recommended learning rates

### 3. Train Model

Start training with recommended settings:

```bash
python scripts/train.py \
    --data_dir ./imagenet_subset \
    --num_classes 100 \
    --epochs 90 \
    --batch_size 256 \
    --initial_lr 0.07047 \
    --max_lr 0.7047 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

**Training Configuration:**
- **Optimizer**: SGD with Nesterov momentum (0.9)
- **Weight Decay**: 1e-4 (excluded from biases and BatchNorm)
- **Scheduler**: OneCycleLR (div_factor=25, pct_start=0.3, cosine anneal)
- **Loss**: CrossEntropyLoss with label smoothing (0.1)
- **Gradient Clipping**: max_norm=1.0
- **Mixed Precision**: Enabled by default (AMP)

### 4. Monitor Training

Logs are saved in JSON and CSV formats:

```
logs/
└── resnet50_imagenet_YYYYMMDD_HHMMSS/
    ├── config.json           # Training configuration
    ├── metrics.json          # All metrics (JSON)
    ├── train_metrics.csv     # Training metrics (CSV)
    └── val_metrics.csv       # Validation metrics (CSV)
```

### 5. Resume Training

Resume from checkpoint:

```bash
python scripts/train.py \
    --data_dir ./imagenet_subset \
    --resume ./checkpoints/latest.pth \
    [... other args ...]
```

## Advanced Usage

### AWS S3 Integration

Upload dataset to S3 for cloud training:

```bash
python scripts/upload_to_s3.py \
    --local_dir ./imagenet_subset \
    --bucket_name your-bucket-name \
    --s3_prefix imagenet_subset \
    --region us-east-1
```

### Sample Images

Visualize random samples from dataset:

```bash
python utils/sample_images.py \
    --data_dir ./imagenet_subset \
    --split train \
    --classes n01440764 n01443537 \
    --num_samples 5 \
    --visualize \
    --save_plot samples.png
```

Get dataset statistics:

```bash
python utils/sample_images.py \
    --data_dir ./imagenet_subset \
    --stats
```

### HuggingFace Deployment

Deploy trained model to HuggingFace Hub:

```bash
# Prepare model for deployment
python deployment/hf_deploy.py \
    --model_path ./checkpoints/best_model.pth \
    --repo_name username/resnet50-imagenet-100 \
    --num_classes 100 \
    --output_dir ./hf_model

# Push to HuggingFace Hub
python deployment/hf_deploy.py \
    --model_path ./checkpoints/best_model.pth \
    --repo_name username/resnet50-imagenet-100 \
    --num_classes 100 \
    --output_dir ./hf_model \
    --push \
    --token YOUR_HF_TOKEN
```

## Model Architecture

### ResNet-50 Specifications

- **Input**: 224×224×3 RGB images
- **Architecture**:
  - Conv1: 7×7, 64 channels, stride 2
  - MaxPool: 3×3, stride 2
  - Layer1: 3 bottleneck blocks, 256 channels
  - Layer2: 4 bottleneck blocks, 512 channels
  - Layer3: 6 bottleneck blocks, 1024 channels
  - Layer4: 3 bottleneck blocks, 2048 channels
  - Global Average Pooling
  - FC: 2048 → num_classes

- **Bottleneck Block**: 1×1 conv → 3×3 conv → 1×1 conv (expansion=4)
- **Initialization**: Kaiming (He) initialization
- **Zero-init Residual**: Last BatchNorm in each residual branch initialized to 0
- **Parameters**: ~23.7M (for 100 classes)

## Data Augmentation

### Training Transforms
1. RandomResizedCrop(224, scale=(0.08, 1.0))
2. RandomHorizontalFlip(p=0.5)
3. ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
4. ToTensor + ImageNet Normalization
5. RandomErasing(p=0.25)

### Validation Transforms
1. Resize(256)
2. CenterCrop(224)
3. ToTensor + ImageNet Normalization

## Hyperparameters

### Optimizer: SGD
- **Momentum**: 0.9 (Nesterov)
- **Weight Decay**: 1e-4
- **Bias/BN Weight Decay**: 0 (excluded)

### Learning Rate: OneCycleLR
- **Initial LR**: 0.07047 (from LR Finder)
- **Max LR**: 0.7047 (from LR Finder)
- **div_factor**: 25 (initial_lr = max_lr / 25)
- **final_div_factor**: 10000
- **pct_start**: 0.3 (30% warmup)
- **Anneal Strategy**: Cosine

### Loss: CrossEntropyLoss
- **Label Smoothing**: 0.1

### Regularization
- **Gradient Clipping**: max_norm=1.0
- **Weight Decay**: 1e-4
- **Label Smoothing**: 0.1
- **RandomErasing**: p=0.25

## Performance Tips

### For Faster Training
- Increase batch size (if memory allows)
- Use more workers (`--num_workers 8`)
- Enable benchmark mode (automatic in code)

### For Better Accuracy
- Train longer (120+ epochs)
- Use AutoAugment (`--auto_augment`)
- Experiment with different learning rates
- Try larger batch sizes with linear LR scaling

### Memory Optimization
- Reduce batch size
- Disable mixed precision (`--no_amp`)
- Reduce number of workers

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python scripts/train.py --batch_size 128 [...]

# Or disable mixed precision
python scripts/train.py --no_amp [...]
```

### Slow Data Loading
```bash
# Increase workers
python scripts/train.py --num_workers 8 [...]
```

### Training Divergence
```bash
# Reduce learning rate
python scripts/train.py --max_lr 0.5 [...]

# Or increase warmup
python scripts/train.py --pct_start 0.4 [...]
```

## Checkpointing

Checkpoints are saved every `save_freq` epochs and include:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training metrics
- Best metric tracking

**Checkpoint Files:**
- `checkpoint_epoch_XXX.pth`: Regular checkpoints
- `best_model.pth`: Best model based on validation accuracy
- `latest.pth`: Most recent checkpoint (for resuming)

## Citation

If you use this code, please cite the original ResNet paper:

```bibtex
@article{he2015deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={arXiv preprint arXiv:1512.03385},
  year={2015}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- ResNet implementation based on the original paper by He et al.
- LR Finder inspired by fastai and Leslie Smith's work
- OneCycleLR from "Super-Convergence" paper by Smith & Topin

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions

## Roadmap

- [ ] TensorBoard integration
- [ ] Multi-GPU training (DDP)
- [ ] EMA (Exponential Moving Average)
- [ ] More augmentation strategies (RandAugment, TrivialAugment)
- [ ] Model quantization
- [ ] ONNX export
