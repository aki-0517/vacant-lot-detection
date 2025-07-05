import os
import json
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, List
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image

from data_pipeline.dataset import (
    EvaluationDataset,
    get_val_transforms,
    collate_fn
)
from models.build_model import build_model, load_checkpoint
from utils import setup_logging, mask_to_polygons


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_optimal_device():
    """Get the best available device"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_image_info_safe(image_path: str) -> Dict[str, Any]:
    """Safely get image dimensions"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return {"width": width, "height": height}
    except:
        return {"width": 500, "height": 375}


def create_optimized_data_loaders(config: Dict[str, Any], device: torch.device):
    """Create optimized data loaders"""
    val_transforms = get_val_transforms()
    
    # Bbox evaluation dataset
    bbox_eval_dir = os.path.join(config['data']['bbox_dir'], 'eval_images')
    bbox_dataset = EvaluationDataset(
        images_dir=bbox_eval_dir,
        transforms=val_transforms
    )
    
    # Optimized settings for speed
    batch_size = 16
    num_workers = 0  # Avoid multiprocessing issues
    
    bbox_loader = DataLoader(
        bbox_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    # Segmentation evaluation dataset
    seg_eval_dir = os.path.join(config['data']['segmentation_dir'], 'eval_images')
    seg_dataset = EvaluationDataset(
        images_dir=seg_eval_dir,
        transforms=val_transforms
    )
    
    seg_loader = DataLoader(
        seg_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    return bbox_loader, seg_loader


def predict_optimized_segmentations(model: nn.Module,
                                  data_loader: DataLoader,
                                  device: torch.device,
                                  config: Dict[str, Any]) -> List[Dict]:
    """Optimized segmentation prediction"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Optimized segmentation prediction')
        
        for batch in pbar:
            images = batch['images'].to(device)
            image_paths = batch['image_paths']
            image_names = batch['image_names']
            
            # Forward pass
            outputs = model(images)
            
            # Process each image in batch
            for i in range(images.size(0)):
                image_name = image_names[i]
                image_path = image_paths[i]
                
                # Get segmentation output
                if isinstance(outputs, dict):
                    seg_output = outputs['segmentation'][i]
                else:
                    seg_output = outputs[i]
                
                # Convert to probability mask
                prob_mask = torch.sigmoid(seg_output).cpu().numpy()
                
                # Single threshold for speed
                threshold = 0.4
                binary_mask = (prob_mask > threshold).astype(np.uint8)
                
                if binary_mask.ndim > 2:
                    binary_mask = binary_mask[0]
                
                # Convert mask to polygons (simplified)
                polygons = mask_to_polygons(binary_mask)
                
                # Take best polygons
                valid_polygons = []
                for polygon in polygons:
                    if len(polygon) >= 6:  # At least 3 points
                        # Calculate area quickly
                        polygon_array = np.array(polygon).reshape(-1, 2)
                        area = cv2.contourArea(polygon_array)
                        if area > 30:  # Minimum area
                            valid_polygons.append(polygon)
                
                # Limit to top polygons
                valid_polygons = valid_polygons[:3]
                
                # Create submission entries
                for polygon in valid_polygons:
                    predictions.append({
                        'image_name': image_name,
                        'segmentation': polygon,
                        'category_id': 1
                    })
                
                # If no polygons found, add default
                if not valid_polygons:
                    img_info = get_image_info_safe(image_path)
                    center_x, center_y = img_info['width'] // 2, img_info['height'] // 2
                    size = min(img_info['width'], img_info['height']) // 6
                    
                    default_polygon = [
                        center_x - size, center_y - size,
                        center_x + size, center_y - size,
                        center_x + size, center_y + size,
                        center_x - size, center_y + size
                    ]
                    
                    predictions.append({
                        'image_name': image_name,
                        'segmentation': default_polygon,
                        'category_id': 1
                    })
    
    return predictions


def predict_optimized_bboxes(model: nn.Module,
                           data_loader: DataLoader,
                           device: torch.device,
                           config: Dict[str, Any]) -> List[Dict]:
    """Optimized bbox prediction"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Optimized bbox prediction')
        
        for batch in pbar:
            images = batch['images'].to(device)
            image_paths = batch['image_paths']
            image_names = batch['image_names']
            
            # Forward pass
            outputs = model(images)
            
            # Process each image in batch
            for i in range(images.size(0)):
                image_name = image_names[i]
                image_path = image_paths[i]
                
                # Get segmentation output
                if isinstance(outputs, dict):
                    seg_output = outputs['segmentation'][i]
                else:
                    seg_output = outputs[i]
                
                # Convert to probability mask
                prob_mask = torch.sigmoid(seg_output).cpu().numpy()
                
                # Single threshold for speed
                threshold = 0.4
                binary_mask = (prob_mask > threshold).astype(np.uint8)
                
                if binary_mask.ndim > 2:
                    binary_mask = binary_mask[0]
                
                # Find connected components
                num_labels, labels = cv2.connectedComponents(binary_mask)
                
                boxes = []
                for label in range(1, num_labels):
                    component_mask = (labels == label)
                    
                    # Calculate score and area
                    score = np.mean(prob_mask[0][component_mask])
                    area = np.sum(component_mask)
                    
                    if score > 0.3 and area > 50:
                        # Find bounding box
                        y_indices, x_indices = np.where(component_mask)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            x1, y1 = int(np.min(x_indices)), int(np.min(y_indices))
                            x2, y2 = int(np.max(x_indices)), int(np.max(y_indices))
                            w, h = x2 - x1, y2 - y1
                            
                            if w > 5 and h > 5:
                                boxes.append({
                                    'bbox': [x1, y1, w, h],
                                    'score': float(score)
                                })
                
                # Sort by score and take top boxes
                boxes.sort(key=lambda x: x['score'], reverse=True)
                boxes = boxes[:2]  # Limit per image
                
                # Create submission entries
                for box in boxes:
                    predictions.append({
                        'image_name': image_name,
                        'bbox': box['bbox'],
                        'score': box['score'],
                        'category_id': 1
                    })
                
                # Ensure at least one prediction
                if not boxes:
                    img_info = get_image_info_safe(image_path)
                    center_x = img_info['width'] // 2
                    center_y = img_info['height'] // 2
                    size = min(img_info['width'], img_info['height']) // 6
                    
                    predictions.append({
                        'image_name': image_name,
                        'bbox': [center_x - size//2, center_y - size//2, size, size],
                        'score': 0.5,
                        'category_id': 1
                    })
    
    return predictions


def create_optimized_submissions(bbox_predictions: List[Dict],
                               seg_predictions: List[Dict],
                               bbox_eval_dir: str,
                               seg_eval_dir: str,
                               output_dir: Path) -> None:
    """Create optimized submission files"""
    
    # Group bbox predictions by image
    bbox_images = {}
    for pred in bbox_predictions:
        image_name = pred['image_name']
        if image_name not in bbox_images:
            img_path = os.path.join(bbox_eval_dir, image_name)
            img_info = get_image_info_safe(img_path)
            bbox_images[image_name] = {
                'file_name': image_name,
                'width': img_info['width'],
                'height': img_info['height'],
                'annotations': []
            }
        
        bbox_images[image_name]['annotations'].append({
            'class': 'vacant_lot',
            'bbox': pred['bbox']
        })
    
    bbox_submission = {'images': list(bbox_images.values())}
    
    # Group segmentation predictions by image
    seg_images = {}
    for pred in seg_predictions:
        image_name = pred['image_name']
        if image_name not in seg_images:
            img_path = os.path.join(seg_eval_dir, image_name)
            img_info = get_image_info_safe(img_path)
            seg_images[image_name] = {
                'file_name': image_name,
                'width': img_info['width'],
                'height': img_info['height'],
                'annotations': []
            }
        
        seg_images[image_name]['annotations'].append({
            'class': 'vacant_lot',
            'segmentation': pred['segmentation']
        })
    
    seg_submission = {'images': list(seg_images.values())}
    
    # Save submissions
    bbox_path = output_dir / 'bbox.json'
    seg_path = output_dir / 'segmentation.json'
    
    with open(bbox_path, 'w', encoding='utf-8') as f:
        json.dump(bbox_submission, f, indent=2, ensure_ascii=False)
    
    with open(seg_path, 'w', encoding='utf-8') as f:
        json.dump(seg_submission, f, indent=2, ensure_ascii=False)
    
    logging.info(f'Optimized bbox predictions: {bbox_path}')
    logging.info(f'Optimized segmentation predictions: {seg_path}')


def main():
    parser = argparse.ArgumentParser(description='Optimized inference for land vacancy detection')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/submissions')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir / 'optimized_inference.log')
    
    # Load config
    config = load_config(args.config)
    
    # Get device
    device = get_optimal_device()
    print(f"Using device: {device}")
    logging.info(f'Using device: {device}')
    
    # Data loaders
    bbox_loader, seg_loader = create_optimized_data_loaders(config, device)
    logging.info(f'Bbox evaluation samples: {len(bbox_loader.dataset)}')
    logging.info(f'Segmentation evaluation samples: {len(seg_loader.dataset)}')
    
    # Model
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    logging.info(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')
    
    # Set model to evaluation mode
    model.eval()
    
    # Optimized predictions
    logging.info('Starting optimized bbox prediction...')
    bbox_predictions = predict_optimized_bboxes(model, bbox_loader, device, config)
    logging.info(f'Generated {len(bbox_predictions)} bbox predictions')
    
    logging.info('Starting optimized segmentation prediction...')
    seg_predictions = predict_optimized_segmentations(model, seg_loader, device, config)
    logging.info(f'Generated {len(seg_predictions)} segmentation predictions')
    
    # Create submission files
    bbox_eval_dir = os.path.join(config['data']['bbox_dir'], 'eval_images')
    seg_eval_dir = os.path.join(config['data']['segmentation_dir'], 'eval_images')
    
    create_optimized_submissions(
        bbox_predictions, seg_predictions,
        bbox_eval_dir, seg_eval_dir, output_dir
    )
    
    logging.info('Optimized inference completed successfully!')


if __name__ == '__main__':
    main()