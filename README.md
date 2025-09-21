# ZoeDepth-3D: Advanced Monocular Depth Estimation & 3D Reconstruction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive computer vision pipeline for monocular depth estimation and 3D point cloud generation using state-of-the-art ZoeDepth models. This project enables high-quality 3D reconstruction from single RGB images for both indoor and outdoor scenes.

## 🌟 Features

- **Advanced Depth Estimation**: Utilizes ZoeDepth with DepthAnything backbone for accurate monocular depth prediction
- **Camera Calibration**: Complete checkerboard-based camera calibration pipeline
- **3D Point Cloud Generation**: Convert depth maps to colored 3D point clouds with proper camera projection
- **Multi-Scene Support**: Optimized for both indoor (NYU) and outdoor (KITTI) environments  
- **Format Conversion**: HEIC to PNG conversion utilities for mobile device images
- **Flexible Inference**: Support for various pretrained models and custom datasets
- **Professional Pipeline**: End-to-end workflow from image capture to 3D visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Camera for calibration (optional)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnirudhDattu/ZoeDepth-3D.git
   cd ZoeDepth-3D
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pretrained models:**
   ```bash
   # For indoor scenes
   wget -O depth_anything_metric_depth_indoor.pt [MODEL_URL]
   
   # For outdoor scenes  
   wget -O depth_anything_metric_depth_outdoor.pt [MODEL_URL]
   ```

## 📚 Usage Guide

### 1. Camera Calibration

First, calibrate your camera using checkerboard patterns:

```bash
python calibration-camera.py
```

**Requirements:**
- Print a 9×6 checkerboard pattern with 24mm squares
- Capture 10-20 images from different angles
- Place images in `chessboard_calibration/` directory

### 2. Depth Estimation & Point Cloud Generation

#### Basic Usage

```bash
# Place input images in ./test/input/
python depth_to_pointcloud.py
```

#### Advanced Usage with Custom Parameters

```python
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# Configure for your use case
DATASET = 'nyu'  # For indoor scenes
# DATASET = 'kitti'  # For outdoor scenes

config = get_config('zoedepth', "eval", DATASET)
model = build_model(config)
```

### 3. Image Format Conversion

Convert HEIC images to PNG format:

```bash
python heic2png.py
```

## 🏗️ Architecture Overview

### ZoeDepth Model Structure

```
Input RGB Image → DINOv2 Feature Extractor → ZoeDepth Decoder → Metric Depth Map
                                                    ↓
Camera Calibration Parameters → 3D Point Projection → Colored Point Cloud
```

### Key Components

- **DepthAnything Core**: Advanced encoder using DINOv2 vision transformer
- **ZoeDepth Architecture**: Metric depth prediction with attractor layers
- **Camera Projection**: Proper 3D reconstruction using calibrated intrinsics
- **Point Cloud Export**: PLY format output for 3D visualization

## 📊 Supported Datasets & Models

### Pretrained Models
- **Indoor Scenes**: Trained on NYU Depth V2 dataset
- **Outdoor Scenes**: Trained on KITTI depth dataset
- **General Purpose**: Mixed training for versatile performance

### Dataset Compatibility
- NYU Depth V2 (Indoor)
- KITTI (Outdoor) 
- DIML Outdoor
- DDAD
- Virtual KITTI
- Custom datasets

## 🔧 Configuration

### Model Parameters

```python
# Key configuration options
CHECKERBOARD = (9, 6)  # Calibration pattern size
SQUARE_SIZE = 0.024    # Physical square size in meters
DATASET = 'nyu'        # Model type selection
NYU_DATA = False       # Focal length computation mode
```

### Directory Structure

```
ZoeDepth-3D/
├── test/
│   ├── input/          # Input RGB images
│   └── output/         # Generated point clouds (.ply)
├── chessboard_calibration/  # Calibration images
├── zoedepth/          # Core model implementation
├── checkpoints/       # Pretrained model weights
└── torchhub/         # DINOv2 backbone
```

## 🎯 Applications

- **3D Reconstruction**: Create 3D models from single photos
- **Augmented Reality**: Depth-aware AR applications
- **Robotics**: Navigation and obstacle detection
- **Architecture**: Building and interior modeling
- **Film/Gaming**: 3D asset generation from reference photos

## 🔬 Technical Details

### Depth Estimation Pipeline

1. **Image Preprocessing**: Resize and normalize input images
2. **Feature Extraction**: DINOv2 ViT-Large encoder extracts rich visual features
3. **Depth Prediction**: ZoeDepth decoder outputs metric depth values
4. **3D Projection**: Camera intrinsics project pixels to 3D space
5. **Point Cloud Export**: Colored vertices saved in PLY format

### Camera Calibration Mathematics

The calibration process estimates:
- **Intrinsic Matrix K**: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
- **Distortion Coefficients**: Radial and tangential distortion parameters
- **Focal Lengths**: fx, fy for accurate depth-to-3D conversion

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ZoeDepth**: Based on the excellent work by Shariq Farooq Bhat et al.
- **DepthAnything**: Utilizes the DepthAnything model architecture
- **DINOv2**: Meta's DINOv2 vision transformer as feature backbone
- **Open3D**: 3D data processing and visualization
- **PyTorch**: Deep learning framework

## 📖 Citations

```bibtex
@article{bhat2023zoedepth,
  title={ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth},
  author={Bhat, Shariq Farooq and Alhashim, Ibraheem and Wonka, Peter},
  journal={arXiv preprint arXiv:2302.12288},
  year={2023}
}
```

## 🐛 Issues & Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/AnirudhDattu/ZoeDepth-3D/issues) page
2. Search existing discussions
3. Create a new issue with detailed information

---

**Made with ❤️ by the ZoeDepth-3D team**