import os
import json
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, List, Tuple
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
    """Get the best available device for inference"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print(f"Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU device")
    
    return device


def get_image_info_safe(image_path: str) -> Dict[str, Any]:
    """Safely get image dimensions"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return {"width": width, "height": height}
    except Exception as e:
        logging.warning(f"Could not read image {image_path}: {e}")
        return {"width": 500, "height": 375}  # Default dimensions


def create_high_accuracy_data_loaders(config: Dict[str, Any], device: torch.device) -> Tuple[DataLoader, DataLoader]:
    """Create optimized data loaders for high accuracy inference"""
    
    # Use the existing validation transforms
    val_transforms = get_val_transforms()
    
    # Bbox evaluation dataset
    bbox_eval_dir = os.path.join(config['data']['bbox_dir'], 'eval_images')
    bbox_dataset = EvaluationDataset(
        images_dir=bbox_eval_dir,
        transforms=val_transforms
    )
    
    # Optimize batch size based on device
    batch_size = 4 if device.type == 'cpu' else 8
    num_workers = 2 if device.type == 'cpu' else 4
    
    bbox_loader = DataLoader(
        bbox_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type != 'cpu'),
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
        pin_memory=(device.type != 'cpu'),
        collate_fn=collate_fn
    )
    
    return bbox_loader, seg_loader


def test_time_augmentation(model: nn.Module, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply test-time augmentation for better accuracy"""
    predictions = []
    
    # Original image
    with torch.no_grad():
        pred = model(image)
        if isinstance(pred, dict):
            pred = pred['segmentation']
        predictions.append(torch.sigmoid(pred))
    
    # Horizontal flip
    image_hflip = torch.flip(image, dims=[3])
    with torch.no_grad():
        pred = model(image_hflip)
        if isinstance(pred, dict):
            pred = pred['segmentation']
        pred_hflip = torch.flip(torch.sigmoid(pred), dims=[3])
        predictions.append(pred_hflip)
    
    # Vertical flip
    image_vflip = torch.flip(image, dims=[2])
    with torch.no_grad():
        pred = model(image_vflip)
        if isinstance(pred, dict):
            pred = pred['segmentation']
        pred_vflip = torch.flip(torch.sigmoid(pred), dims=[2])
        predictions.append(pred_vflip)
    
    # Average all predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred


def predict_high_accuracy_segmentations(model: nn.Module,
                                       data_loader: DataLoader,
                                       device: torch.device,
                                       config: Dict[str, Any],
                                       use_tta: bool = True) -> List[Dict]:
    """High accuracy segmentation prediction with TTA"""
    model.eval()
    predictions = []
    
    # Lower threshold for better recall
    mask_threshold = 0.3
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='High-accuracy segmentation prediction')
        
        for batch in pbar:
            images = batch['images'].to(device)
            image_paths = batch['image_paths']
            image_names = batch['image_names']
            
            # Process each image in batch
            for i in range(images.size(0)):
                image_name = image_names[i]
                image_path = image_paths[i]
                single_image = images[i:i+1]  # Keep batch dimension
                
                # Apply test-time augmentation
                if use_tta:
                    prob_mask = test_time_augmentation(model, single_image, device)[0]
                else:
                    outputs = model(single_image)
                    if isinstance(outputs, dict):
                        seg_output = outputs['segmentation'][0]
                    else:
                        seg_output = outputs[0]
                    prob_mask = torch.sigmoid(seg_output)
                
                prob_mask_np = prob_mask.cpu().numpy()
                
                # Multi-threshold approach for better accuracy
                thresholds = [0.2, 0.3, 0.4, 0.5]
                all_polygons = []
                
                for threshold in thresholds:
                    binary_mask = (prob_mask_np > threshold).astype(np.uint8)
                    
                    if binary_mask.ndim > 2:
                        binary_mask = binary_mask[0]
                    
                    # Convert mask to polygons
                    polygons = mask_to_polygons(binary_mask)
                    
                    # Filter by area and score
                    for polygon in polygons:
                        if len(polygon) >= 6:  # At least 3 points
                            # Calculate polygon area
                            polygon_array = np.array(polygon).reshape(-1, 2)
                            area = cv2.contourArea(polygon_array)
                            
                            # Filter small polygons
                            if area > 50:  # Minimum area threshold
                                all_polygons.append(polygon)
                
                # Remove duplicate polygons and create submission entries
                unique_polygons = []
                for polygon in all_polygons:
                    # Simple deduplication based on first few points
                    is_duplicate = False
                    for existing in unique_polygons:
                        if len(polygon) >= 4 and len(existing) >= 4:
                            if (abs(polygon[0] - existing[0]) < 5 and 
                                abs(polygon[1] - existing[1]) < 5):
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        unique_polygons.append(polygon)
                
                # Create submission entries
                for polygon in unique_polygons[:5]:  # Limit to top 5 per image
                    predictions.append({
                        'image_name': image_name,
                        'segmentation': polygon,
                        'category_id': 1
                    })
                
                # If no good polygons found, add a default one
                if not unique_polygons:
                    img_info = get_image_info_safe(image_path)
                    center_x, center_y = img_info['width'] // 2, img_info['height'] // 2
                    size = min(img_info['width'], img_info['height']) // 4
                    
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


def predict_high_accuracy_bboxes(model: nn.Module,
                                data_loader: DataLoader,
                                device: torch.device,
                                config: Dict[str, Any],
                                use_tta: bool = True) -> List[Dict]:
    """High accuracy bbox prediction from segmentation"""
    model.eval()
    predictions = []
    
    score_threshold = 0.3  # Lower threshold for better recall
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='High-accuracy bbox prediction')
        
        for batch in pbar:
            images = batch['images'].to(device)
            image_paths = batch['image_paths']
            image_names = batch['image_names']
            
            # Process each image in batch
            for i in range(images.size(0)):
                image_name = image_names[i]
                image_path = image_paths[i]
                single_image = images[i:i+1]
                
                # Apply test-time augmentation
                if use_tta:
                    prob_mask = test_time_augmentation(model, single_image, device)[0]
                else:
                    outputs = model(single_image)
                    if isinstance(outputs, dict):
                        seg_output = outputs['segmentation'][0]
                    else:
                        seg_output = outputs[0]
                    prob_mask = torch.sigmoid(seg_output)
                
                prob_mask_np = prob_mask.cpu().numpy()
                
                # Multi-threshold approach
                all_boxes = []
                thresholds = [0.2, 0.3, 0.4, 0.5]
                
                for threshold in thresholds:
                    binary_mask = (prob_mask_np > threshold).astype(np.uint8)
                    
                    if binary_mask.ndim > 2:
                        binary_mask = binary_mask[0]
                    
                    # Find connected components
                    num_labels, labels = cv2.connectedComponents(binary_mask)
                    
                    for label in range(1, num_labels):
                        component_mask = (labels == label)
                        
                        # Calculate score and area
                        score = np.mean(prob_mask_np[0][component_mask])
                        area = np.sum(component_mask)
                        
                        if score > score_threshold and area > 100:  # Minimum area
                            # Find bounding box
                            y_indices, x_indices = np.where(component_mask)
                            if len(y_indices) > 0 and len(x_indices) > 0:
                                x1, y1 = int(np.min(x_indices)), int(np.min(y_indices))
                                x2, y2 = int(np.max(x_indices)), int(np.max(y_indices))
                                w, h = x2 - x1, y2 - y1
                                
                                if w > 10 and h > 10:  # Minimum size
                                    all_boxes.append({
                                        'bbox': [x1, y1, w, h],
                                        'score': float(score),
                                        'area': area
                                    })
                
                # Sort by score and remove duplicates
                all_boxes.sort(key=lambda x: x['score'], reverse=True)
                
                # Remove overlapping boxes (simple NMS)
                final_boxes = []
                for box in all_boxes:
                    is_overlap = False
                    for existing in final_boxes:
                        # Simple overlap check
                        if (abs(box['bbox'][0] - existing['bbox'][0]) < 20 and
                            abs(box['bbox'][1] - existing['bbox'][1]) < 20):
                            is_overlap = True
                            break
                    
                    if not is_overlap:
                        final_boxes.append(box)
                        if len(final_boxes) >= 3:  # Limit per image
                            break
                
                # Create submission entries
                for box in final_boxes:
                    predictions.append({
                        'image_name': image_name,
                        'bbox': box['bbox'],
                        'score': box['score'],
                        'category_id': 1
                    })
                
                # Ensure at least one prediction per image
                if not final_boxes:
                    img_info = get_image_info_safe(image_path)
                    center_x = img_info['width'] // 2
                    center_y = img_info['height'] // 2
                    default_size = min(img_info['width'], img_info['height']) // 4
                    
                    predictions.append({
                        'image_name': image_name,
                        'bbox': [center_x - default_size//2, center_y - default_size//2, 
                               default_size, default_size],
                        'score': 0.5,
                        'category_id': 1
                    })
    
    return predictions


def create_high_accuracy_submissions(bbox_predictions: List[Dict],
                                   seg_predictions: List[Dict],
                                   bbox_eval_dir: str,
                                   seg_eval_dir: str,
                                   output_dir: Path) -> None:
    """Create high-accuracy submission files"""
    
    # Create bbox submission
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
    
    # Create segmentation submission
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
    
    logging.info(f'High-accuracy bbox predictions: {bbox_path}')
    logging.info(f'High-accuracy segmentation predictions: {seg_path}')


def main():
    parser = argparse.ArgumentParser(description='High-accuracy inference for land vacancy detection')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/submissions')
    parser.add_argument('--use_tta', action='store_true', default=True,
                       help='Use test-time augmentation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir / 'high_accuracy_inference.log')
    
    # Load config
    config = load_config(args.config)
    
    # Get optimal device
    device = get_optimal_device()
    
    # Data loaders
    bbox_loader, seg_loader = create_high_accuracy_data_loaders(config, device)
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
    
    # High-accuracy predictions
    logging.info('Starting high-accuracy bbox prediction...')
    bbox_predictions = predict_high_accuracy_bboxes(
        model, bbox_loader, device, config, args.use_tta
    )
    logging.info(f'Generated {len(bbox_predictions)} bbox predictions')
    
    logging.info('Starting high-accuracy segmentation prediction...')
    seg_predictions = predict_high_accuracy_segmentations(
        model, seg_loader, device, config, args.use_tta
    )
    logging.info(f'Generated {len(seg_predictions)} segmentation predictions')
    
    # Create submission files
    bbox_eval_dir = os.path.join(config['data']['bbox_dir'], 'eval_images')
    seg_eval_dir = os.path.join(config['data']['segmentation_dir'], 'eval_images')
    
    create_high_accuracy_submissions(
        bbox_predictions, seg_predictions,
        bbox_eval_dir, seg_eval_dir, output_dir
    )
    
    logging.info('High-accuracy inference completed successfully!')


if __name__ == '__main__':
    main()