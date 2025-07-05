# Land Vacancy Detection

A computer vision system for detecting vacant lots in aerial and satellite imagery using deep learning techniques.

## Project Overview

This project implements both object detection (bounding box) and semantic segmentation approaches to identify vacant lots in aerial/satellite images. The system is designed for competition submissions and provides automated generation of submission files.

## Features

- **Dual Detection**: Both bounding box detection and pixel-level segmentation
- **Modern Architecture**: U-Net with ResNet backbone for robust performance
- **Automated Pipeline**: End-to-end workflow from data preprocessing to submission
- **Comprehensive Evaluation**: Multiple metrics including IoU, Dice score, and mAP
- **Visualization Tools**: Built-in visualization for predictions and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/example/vacant-lot-detection.git
cd vacant-lot-detection

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Data Preparation

Place your raw data in the `data/raw/` directory:
- `train_bbox_images.zip`
- `train_bbox_annotations.json`
- `train_segmentation_images.zip`
- `train_segmentation_annotations.json`
- `evaluation_bbox_images.zip`
- `evaluation_segmentation_images.zip`

### 2. Preprocessing

```bash
python src/data_pipeline/preprocess.py --config configs/config.yaml --mode both
```

### 3. Training

```bash
python src/train.py --config configs/config.yaml
```

### 4. Evaluation

```bash
python src/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pth
```

### 5. Inference

```bash
python src/inference.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pth --output_dir data/submissions/
```

## Configuration

All settings are managed through `configs/config.yaml`. Key parameters include:

- **Model**: Architecture type, backbone, number of classes
- **Training**: Learning rate, batch size, epochs, augmentations
- **Inference**: Score thresholds, NMS parameters
- **Paths**: Data directories and file locations

## Directory Structure

```
vacant-lot-detection/
├── data/
│   ├── raw/                    # Raw competition data
│   ├── processed/              # Preprocessed images and masks
│   └── submissions/            # Final submission files
├── src/
│   ├── data_pipeline/          # Data loading and preprocessing
│   ├── models/                 # Model architectures
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── inference.py           # Inference and submission generation
│   └── utils.py               # Common utilities
├── configs/
│   └── config.yaml            # Configuration file
├── outputs/
│   ├── logs/                  # TensorBoard logs
│   └── checkpoints/           # Model checkpoints
└── notebooks/                 # Jupyter notebooks for analysis
```

## Model Architecture

The system uses a U-Net architecture with ResNet34 backbone:

- **Encoder**: Pre-trained ResNet34 for feature extraction
- **Decoder**: U-Net decoder for pixel-wise predictions
- **Loss Function**: Combined Dice + Focal loss for better performance
- **Augmentations**: Horizontal/vertical flip, rotation, brightness/contrast

## Evaluation Metrics

- **Segmentation**: IoU, Dice score, Precision, Recall, F1-score
- **Detection**: mAP (mean Average Precision) at multiple IoU thresholds
- **Threshold Analysis**: Performance across different confidence thresholds

## Submission Format

The system generates two JSON files:

1. **bbox.json**: Bounding box predictions
2. **segmentation.json**: Polygon segmentation predictions

Both files follow the competition format requirements.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
isort src/
flake8 src/
```

### Monitoring Training

```bash
tensorboard --logdir outputs/logs/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.