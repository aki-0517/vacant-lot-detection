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
from utils import setup_logging, apply_nms, mask_to_polygons


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    val_transforms = get_val_transforms()
    
    # Bbox evaluation dataset
    bbox_eval_dir = os.path.join(config['data']['bbox_dir'], 'eval_images')
    bbox_dataset = EvaluationDataset(
        images_dir=bbox_eval_dir,
        transforms=val_transforms
    )
    
    bbox_loader = DataLoader(
        bbox_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
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
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return bbox_loader, seg_loader


def predict_bboxes(model: nn.Module,
                   data_loader: DataLoader,
                   device: torch.device,
                   config: Dict[str, Any]) -> List[Dict]:
    model.eval()
    predictions = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Predicting bboxes')
        
        for batch in pbar:
            images = batch['images'].to(device)
            image_paths = batch['image_paths']
            image_names = batch['image_names']
            
            # Forward pass
            outputs = model(images)
            
            # Process each image in batch
            for i in range(images.size(0)):
                image_name = image_names[i]
                
                # If model returns dict (combined model)
                if isinstance(outputs, dict):
                    if 'bbox' in outputs:
                        bbox_output = outputs['bbox']
                        cls_scores = torch.sigmoid(bbox_output['classification'][i])
                        bbox_coords = bbox_output['regression'][i]
                    else:
                        # Use segmentation output to generate bboxes
                        seg_output = outputs['segmentation'][i]
                        cls_scores, bbox_coords = segmentation_to_bbox(seg_output, config)
                else:
                    # Single segmentation model - convert to bboxes
                    seg_output = outputs[i]
                    cls_scores, bbox_coords = segmentation_to_bbox(seg_output, config)
                
                # Filter by score threshold
                valid_indices = cls_scores > config['inference']['score_threshold']
                
                if valid_indices.any():
                    valid_scores = cls_scores[valid_indices]
                    valid_boxes = bbox_coords[valid_indices]
                    
                    # Apply NMS
                    keep_indices = apply_nms(valid_boxes, valid_scores, config['inference']['nms_iou_threshold'])
                    
                    final_boxes = valid_boxes[keep_indices]
                    final_scores = valid_scores[keep_indices]
                    
                    # Convert to submission format
                    for box, score in zip(final_boxes, final_scores):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        predictions.append({
                            'image_name': image_name,
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # [x, y, w, h]
                            'score': float(score),
                            'category_id': 1
                        })
    
    return predictions


def predict_segmentations(model: nn.Module,
                         data_loader: DataLoader,
                         device: torch.device,
                         config: Dict[str, Any]) -> List[Dict]:
    model.eval()
    predictions = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Predicting segmentations')
        
        for batch in pbar:
            images = batch['images'].to(device)
            image_paths = batch['image_paths']
            image_names = batch['image_names']
            
            # Forward pass
            outputs = model(images)
            
            # Process each image in batch
            for i in range(images.size(0)):
                image_name = image_names[i]
                
                # Get segmentation output
                if isinstance(outputs, dict):
                    seg_output = outputs['segmentation'][i]
                else:
                    seg_output = outputs[i]
                
                # Convert to probability mask
                prob_mask = torch.sigmoid(seg_output).cpu().numpy()
                
                # Threshold to binary mask
                binary_mask = (prob_mask > config['inference']['mask_threshold']).astype(np.uint8)
                
                # Convert mask to polygons
                polygons = mask_to_polygons(binary_mask[0])  # Remove channel dimension
                
                # Create submission entries
                for polygon in polygons:
                    if len(polygon) >= 6:  # At least 3 points
                        predictions.append({
                            'image_name': image_name,
                            'segmentation': polygon,
                            'category_id': 1
                        })
    
    return predictions


def segmentation_to_bbox(seg_output: torch.Tensor, config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    prob_mask = torch.sigmoid(seg_output)
    binary_mask = (prob_mask > config['inference']['mask_threshold']).cpu().numpy()
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask[0].astype(np.uint8))
    
    scores = []
    boxes = []
    
    for label in range(1, num_labels):  # Skip background (label 0)
        component_mask = (labels == label)
        
        # Calculate score as mean probability in the component
        component_prob = prob_mask[0].cpu().numpy() * component_mask
        score = np.mean(component_prob[component_mask])
        
        # Find bounding box
        y_indices, x_indices = np.where(component_mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x1, y1 = np.min(x_indices), np.min(y_indices)
            x2, y2 = np.max(x_indices), np.max(y_indices)
            
            scores.append(score)
            boxes.append([x1, y1, x2, y2])
    
    if len(scores) == 0:
        return torch.tensor([]), torch.tensor([]).reshape(0, 4)
    
    return torch.tensor(scores), torch.tensor(boxes, dtype=torch.float32)


def create_submission_files(bbox_predictions: List[Dict],
                           seg_predictions: List[Dict],
                           output_dir: Path) -> None:
    # Create bbox submission
    bbox_submission = []
    for i, pred in enumerate(bbox_predictions):
        bbox_submission.append({
            'id': i,
            'image_name': pred['image_name'],
            'bbox': pred['bbox'],
            'score': pred['score'],
            'category_id': pred['category_id']
        })
    
    # Create segmentation submission
    seg_submission = []
    for i, pred in enumerate(seg_predictions):
        seg_submission.append({
            'id': i,
            'image_name': pred['image_name'],
            'segmentation': pred['segmentation'],
            'category_id': pred['category_id']
        })
    
    # Save submissions
    bbox_path = output_dir / 'bbox.json'
    seg_path = output_dir / 'segmentation.json'
    
    with open(bbox_path, 'w') as f:
        json.dump(bbox_submission, f, indent=2)
    
    with open(seg_path, 'w') as f:
        json.dump(seg_submission, f, indent=2)
    
    logging.info(f'Saved bbox predictions: {bbox_path}')
    logging.info(f'Saved segmentation predictions: {seg_path}')


def visualize_predictions(model: nn.Module,
                         data_loader: DataLoader,
                         device: torch.device,
                         config: Dict[str, Any],
                         output_dir: Path,
                         num_samples: int = 10) -> None:
    model.eval()
    
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= num_samples:
                break
            
            images = batch['images'].to(device)
            image_paths = batch['image_paths']
            image_names = batch['image_names']
            
            # Forward pass
            outputs = model(images)
            
            # Process each image in batch
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break
                
                image_name = image_names[i]
                
                # Load original image
                orig_image = cv2.imread(image_paths[i])
                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                
                # Get prediction
                if isinstance(outputs, dict):
                    if 'segmentation' in outputs:
                        seg_output = outputs['segmentation'][i]
                        prob_mask = torch.sigmoid(seg_output).cpu().numpy()
                        
                        # Create visualization
                        plt.figure(figsize=(15, 5))
                        
                        # Original image
                        plt.subplot(1, 3, 1)
                        plt.imshow(orig_image)
                        plt.title('Original Image')
                        plt.axis('off')
                        
                        # Probability mask
                        plt.subplot(1, 3, 2)
                        plt.imshow(prob_mask[0], cmap='hot', alpha=0.7)
                        plt.title('Probability Mask')
                        plt.axis('off')
                        
                        # Overlay
                        plt.subplot(1, 3, 3)
                        plt.imshow(orig_image)
                        plt.imshow(prob_mask[0], cmap='hot', alpha=0.5)
                        plt.title('Overlay')
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(vis_dir / f'{image_name}_prediction.png', dpi=150, bbox_inches='tight')
                        plt.close()
                
                sample_count += 1


def main():
    parser = argparse.ArgumentParser(description='Inference for land vacancy detection')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='data/submissions',
                       help='Output directory for submission files')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization of predictions')
    parser.add_argument('--num_vis', type=int, default=10,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir / 'inference.log')
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Data loaders
    bbox_loader, seg_loader = setup_data_loaders(config)
    logging.info(f'Bbox evaluation samples: {len(bbox_loader.dataset)}')
    logging.info(f'Segmentation evaluation samples: {len(seg_loader.dataset)}')
    
    # Model
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    logging.info(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')
    
    # Predict bboxes
    logging.info('Predicting bounding boxes...')
    bbox_predictions = predict_bboxes(model, bbox_loader, device, config)
    logging.info(f'Generated {len(bbox_predictions)} bbox predictions')
    
    # Predict segmentations
    logging.info('Predicting segmentations...')
    seg_predictions = predict_segmentations(model, seg_loader, device, config)
    logging.info(f'Generated {len(seg_predictions)} segmentation predictions')
    
    # Create submission files
    create_submission_files(bbox_predictions, seg_predictions, output_dir)
    
    # Visualize predictions
    if args.visualize:
        logging.info('Generating visualizations...')
        visualize_predictions(model, seg_loader, device, config, output_dir, args.num_vis)
    
    logging.info('Inference completed!')


if __name__ == '__main__':
    main()