import os
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
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline.dataset import (
    SegmentationDataset, 
    get_val_transforms,
    collate_fn
)
from models.build_model import build_model, load_checkpoint
from utils import setup_logging, calculate_iou, calculate_dice, calculate_precision_recall


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_data_loader(config: Dict[str, Any]) -> DataLoader:
    images_dir = os.path.join(config['data']['segmentation_dir'], 'images')
    masks_dir = os.path.join(config['data']['segmentation_dir'], 'masks')
    annotations_path = os.path.join(config['data']['segmentation_dir'], 'annotations.json')
    
    val_transforms = get_val_transforms()
    
    dataset = SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        annotations_path=annotations_path,
        transforms=val_transforms
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return data_loader


def evaluate_segmentation(model: nn.Module,
                         data_loader: DataLoader,
                         device: torch.device,
                         threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    
    all_ious = []
    all_dices = []
    all_precisions = []
    all_recalls = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluating')
        
        for batch in pbar:
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            
            # Forward pass
            outputs = model(images)
            pred_masks = torch.sigmoid(outputs) > threshold
            
            # Calculate metrics for each sample in batch
            for i in range(images.size(0)):
                pred_mask = pred_masks[i]
                true_mask = masks[i]
                
                iou = calculate_iou(pred_mask.unsqueeze(0), true_mask.unsqueeze(0))
                dice = calculate_dice(pred_mask.unsqueeze(0), true_mask.unsqueeze(0))
                precision, recall = calculate_precision_recall(pred_mask.unsqueeze(0), true_mask.unsqueeze(0))
                
                all_ious.append(iou)
                all_dices.append(dice)
                all_precisions.append(precision)
                all_recalls.append(recall)
            
            # Update progress bar
            pbar.set_postfix({
                'IoU': np.mean(all_ious),
                'Dice': np.mean(all_dices)
            })
    
    # Calculate final metrics
    metrics = {
        'iou_mean': np.mean(all_ious),
        'iou_std': np.std(all_ious),
        'dice_mean': np.mean(all_dices),
        'dice_std': np.std(all_dices),
        'precision_mean': np.mean(all_precisions),
        'precision_std': np.std(all_precisions),
        'recall_mean': np.mean(all_recalls),
        'recall_std': np.std(all_recalls),
        'f1_mean': np.mean([2 * p * r / (p + r) if (p + r) > 0 else 0 
                           for p, r in zip(all_precisions, all_recalls)])
    }
    
    return metrics, all_ious, all_dices


def evaluate_multiple_thresholds(model: nn.Module,
                                data_loader: DataLoader,
                                device: torch.device,
                                thresholds: List[float]) -> Dict[str, List[float]]:
    model.eval()
    
    results = {
        'thresholds': thresholds,
        'ious': [],
        'dices': [],
        'precisions': [],
        'recalls': [],
        'f1s': []
    }
    
    with torch.no_grad():
        # Collect all predictions and ground truth
        all_outputs = []
        all_masks = []
        
        pbar = tqdm(data_loader, desc='Collecting predictions')
        for batch in pbar:
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            
            all_outputs.append(outputs.cpu())
            all_masks.append(masks.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # Evaluate at different thresholds
        for threshold in tqdm(thresholds, desc='Evaluating thresholds'):
            pred_masks = all_outputs > threshold
            
            iou = calculate_iou(pred_masks, all_masks)
            dice = calculate_dice(pred_masks, all_masks)
            precision, recall = calculate_precision_recall(pred_masks, all_masks)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results['ious'].append(iou)
            results['dices'].append(dice)
            results['precisions'].append(precision)
            results['recalls'].append(recall)
            results['f1s'].append(f1)
    
    return results


def plot_metrics(results: Dict[str, List[float]], save_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # IoU vs Threshold
    axes[0, 0].plot(results['thresholds'], results['ious'], 'b-', marker='o')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('IoU')
    axes[0, 0].set_title('IoU vs Threshold')
    axes[0, 0].grid(True)
    
    # Dice vs Threshold
    axes[0, 1].plot(results['thresholds'], results['dices'], 'g-', marker='o')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Dice Score vs Threshold')
    axes[0, 1].grid(True)
    
    # Precision-Recall
    axes[1, 0].plot(results['recalls'], results['precisions'], 'r-', marker='o')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].grid(True)
    
    # F1 vs Threshold
    axes[1, 1].plot(results['thresholds'], results['f1s'], 'm-', marker='o')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score vs Threshold')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_distribution(all_ious: List[float], all_dices: List[float], save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # IoU distribution
    axes[0].hist(all_ious, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(all_ious), color='red', linestyle='--', label=f'Mean: {np.mean(all_ious):.3f}')
    axes[0].set_xlabel('IoU')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('IoU Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice distribution
    axes[1].hist(all_dices, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(np.mean(all_dices), color='red', linestyle='--', label=f'Mean: {np.mean(all_dices):.3f}')
    axes[1].set_xlabel('Dice Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Dice Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate land vacancy detection model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--thresholds', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                       help='Comma-separated list of thresholds to evaluate')
    
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [float(t) for t in args.thresholds.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir / 'evaluation.log')
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Data loader
    data_loader = setup_data_loader(config)
    logging.info(f'Evaluation samples: {len(data_loader.dataset)}')
    
    # Model
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    logging.info(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')
    
    # Single threshold evaluation
    logging.info(f'Evaluating at threshold {config["inference"]["mask_threshold"]}')
    metrics, all_ious, all_dices = evaluate_segmentation(
        model, data_loader, device, config['inference']['mask_threshold']
    )
    
    # Log results
    logging.info('Evaluation Results:')
    logging.info(f'IoU: {metrics["iou_mean"]:.4f} ± {metrics["iou_std"]:.4f}')
    logging.info(f'Dice: {metrics["dice_mean"]:.4f} ± {metrics["dice_std"]:.4f}')
    logging.info(f'Precision: {metrics["precision_mean"]:.4f} ± {metrics["precision_std"]:.4f}')
    logging.info(f'Recall: {metrics["recall_mean"]:.4f} ± {metrics["recall_std"]:.4f}')
    logging.info(f'F1: {metrics["f1_mean"]:.4f}')
    
    # Save single threshold results
    import json
    with open(output_dir / 'single_threshold_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot metric distributions
    plot_metric_distribution(all_ious, all_dices, output_dir / 'metric_distributions.png')
    
    # Multiple threshold evaluation
    logging.info('Evaluating at multiple thresholds...')
    threshold_results = evaluate_multiple_thresholds(model, data_loader, device, thresholds)
    
    # Save threshold results
    with open(output_dir / 'threshold_results.json', 'w') as f:
        json.dump(threshold_results, f, indent=2)
    
    # Plot threshold results
    plot_metrics(threshold_results, output_dir / 'threshold_analysis.png')
    
    # Find best threshold
    best_f1_idx = np.argmax(threshold_results['f1s'])
    best_threshold = threshold_results['thresholds'][best_f1_idx]
    best_f1 = threshold_results['f1s'][best_f1_idx]
    
    logging.info(f'Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})')
    
    # Summary
    summary = {
        'single_threshold': {
            'threshold': config['inference']['mask_threshold'],
            'metrics': metrics
        },
        'best_threshold': {
            'threshold': best_threshold,
            'f1_score': best_f1,
            'iou': threshold_results['ious'][best_f1_idx],
            'dice': threshold_results['dices'][best_f1_idx]
        }
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info('Evaluation completed!')


if __name__ == '__main__':
    main()