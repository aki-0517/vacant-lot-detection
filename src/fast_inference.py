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

from data_pipeline.dataset import (
    EvaluationDataset,
    get_val_transforms,
    collate_fn
)
from models.build_model import build_model, load_checkpoint
from utils import setup_logging, mask_to_polygons


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
        num_workers=8,
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
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return bbox_loader, seg_loader


def predict_segmentations_fast(model: nn.Module,
                               data_loader: DataLoader,
                               device: torch.device,
                               config: Dict[str, Any]) -> List[Dict]:
    model.eval()
    predictions = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Predicting segmentations')
        
        for batch in pbar:
            images = batch['images'].to(device)
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


def predict_bboxes_fast(model: nn.Module,
                        data_loader: DataLoader,
                        device: torch.device,
                        config: Dict[str, Any]) -> List[Dict]:
    model.eval()
    predictions = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Predicting bboxes')
        
        for batch in pbar:
            images = batch['images'].to(device)
            image_names = batch['image_names']
            
            # Forward pass
            outputs = model(images)
            
            # Process each image in batch
            for i in range(images.size(0)):
                image_name = image_names[i]
                
                # Get segmentation output and convert to bbox
                if isinstance(outputs, dict):
                    seg_output = outputs['segmentation'][i]
                else:
                    seg_output = outputs[i]
                
                # Convert to probability mask
                prob_mask = torch.sigmoid(seg_output).cpu().numpy()
                binary_mask = (prob_mask > config['inference']['mask_threshold']).astype(np.uint8)
                
                # Find connected components
                num_labels, labels = cv2.connectedComponents(binary_mask[0])
                
                for label in range(1, num_labels):  # Skip background
                    component_mask = (labels == label)
                    
                    # Calculate score as mean probability
                    score = np.mean(prob_mask[0][component_mask])
                    
                    if score > config['inference']['score_threshold']:
                        # Find bounding box
                        y_indices, x_indices = np.where(component_mask)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            x1, y1 = np.min(x_indices), np.min(y_indices)
                            x2, y2 = np.max(x_indices), np.max(y_indices)
                            w, h = x2 - x1, y2 - y1
                            
                            predictions.append({
                                'image_name': image_name,
                                'bbox': [float(x1), float(y1), float(w), float(h)],
                                'score': float(score),
                                'category_id': 1
                            })
    
    return predictions


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


def main():
    parser = argparse.ArgumentParser(description='Fast inference for land vacancy detection')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='data/submissions',
                       help='Output directory for submission files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir / 'fast_inference.log')
    
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
    bbox_predictions = predict_bboxes_fast(model, bbox_loader, device, config)
    logging.info(f'Generated {len(bbox_predictions)} bbox predictions')
    
    # Predict segmentations
    logging.info('Predicting segmentations...')
    seg_predictions = predict_segmentations_fast(model, seg_loader, device, config)
    logging.info(f'Generated {len(seg_predictions)} segmentation predictions')
    
    # Create submission files
    create_submission_files(bbox_predictions, seg_predictions, output_dir)
    
    logging.info('Fast inference completed!')


if __name__ == '__main__':
    main()