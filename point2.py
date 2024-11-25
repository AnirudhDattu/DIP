import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import pandas as pd  # Required for PyntCloud

# Load the saved calibration parameters
calibration_data = np.load("CalibrationMatrix_college_cpt.npz")
camera_matrix = calibration_data['Camera_matrix']
FX = camera_matrix[0, 0]  # Focal length in the x direction
FY = camera_matrix[1, 1]  # Focal length in the y direction
FL = (FX + FY) / 2  # Average focal length

NYU_DATA = False
INPUT_DIR = './test/input'
OUTPUT_DIR = './test/output'
DATASET = 'nyu'  # For INDOOR

def write_ply(points, colors, filename):
    """Writes points and colors to a PLY file."""
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        # Write points
        for point, color in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")

def process_images(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.png')) + glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # Load image
            color_image = Image.open(image_path).convert('RGB')
            original_width, original_height = color_image.size
            FINAL_HEIGHT, FINAL_WIDTH = original_height, original_width
            image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to(
                'cuda' if torch.cuda.is_available() else 'cpu')

            # Get depth prediction
            pred = model(image_tensor, dataset=DATASET)
            if isinstance(pred, dict):
                pred = pred.get('metric_depth', pred.get('out'))
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]
            pred = pred.squeeze().detach().cpu().numpy()

            # Resize color image and depth to final size
            resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
            resized_pred = np.array(Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST))

            # Generate 3D points
            focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
            x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
            x = (x - FINAL_WIDTH / 2) / focal_length_x
            y = (y - FINAL_HEIGHT / 2) / focal_length_y
            z = resized_pred
            points = np.stack((x * z, y * z, z), axis=-1).reshape(-1, 3)

            # Normalize colors
            colors = np.array(resized_color_image).reshape(-1, 3)

            # Write to PLY
            output_file = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".ply")
            write_ply(points, colors, output_file)
            print(f"Point cloud saved to {output_file}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main(model_name, pretrained_resource):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    process_images(model)

if __name__ == '__main__':
    model = 'zoedepth'
    pretrained_resource = 'local::./depth_anything_metric_depth_indoor.pt'
    main(model, pretrained_resource)
