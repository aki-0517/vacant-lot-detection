import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Tuple
import logging
from pathlib import Path

from data_pipeline.dataset import (
    SegmentationDataset, 
    get_train_transforms, 
    get_val_transforms,
    collate_fn
)
from models.build_model import (
    build_model,
    build_loss_function,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    count_parameters
)
from utils import setup_logging, calculate_iou, EarlyStopping


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    # Dataset paths
    images_dir = os.path.join(config['data']['segmentation_dir'], 'images')
    masks_dir = os.path.join(config['data']['segmentation_dir'], 'masks')
    annotations_path = os.path.join(config['data']['segmentation_dir'], 'annotations.json')
    
    # Transforms
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms()
    
    # Full dataset
    full_dataset = SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        annotations_path=annotations_path,
        transforms=None
    )
    
    # Train/validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
    train_dataset.dataset.transforms = train_transforms
    val_dataset.dataset.transforms = val_transforms
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int,
                writer: SummaryWriter,
                config: Dict[str, Any]) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        masks = batch['masks'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            pred_masks = torch.sigmoid(outputs) > 0.5
            iou = calculate_iou(pred_masks, masks)
        
        total_loss += loss.item()
        total_iou += iou
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'iou': iou,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Log to tensorboard
        if batch_idx % config['logging']['log_interval'] == 0:
            step = epoch * num_batches + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), step)
            writer.add_scalar('Train/IoU', iou, step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], step)
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return {
        'loss': avg_loss,
        'iou': avg_iou
    }


def validate_epoch(model: nn.Module,
                   val_loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device,
                   epoch: int,
                   writer: SummaryWriter,
                   config: Dict[str, Any]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            pred_masks = torch.sigmoid(outputs) > 0.5
            iou = calculate_iou(pred_masks, masks)
            
            total_loss += loss.item()
            total_iou += iou
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'iou': iou
            })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/IoU', avg_iou, epoch)
    
    return {
        'loss': avg_loss,
        'iou': avg_iou
    }


def main():
    parser = argparse.ArgumentParser(description='Train land vacancy detection model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    log_dir = Path(config['logging']['tensorboard_logdir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / 'train.log')
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Data loaders
    train_loader, val_loader = setup_data_loaders(config)
    logging.info(f'Train samples: {len(train_loader.dataset)}')
    logging.info(f'Val samples: {len(val_loader.dataset)}')
    
    # Model
    model = build_model(config)
    model = model.to(device)
    
    # Log model info
    num_params = count_parameters(model)
    logging.info(f'Model parameters: {num_params:,}')
    
    # Loss function
    criterion = build_loss_function(config)
    
    # Optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        verbose=True,
        delta=1e-6
    )
    
    # TensorBoard
    writer = SummaryWriter(config['logging']['tensorboard_logdir'])
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config['training']['epochs']):
        logging.info(f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, config
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer, config
        )
        
        # Step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics['loss'])
        else:
            scheduler.step()
        
        # Log epoch metrics
        logging.info(f'Train - Loss: {train_metrics["loss"]:.4f}, IoU: {train_metrics["iou"]:.4f}')
        logging.info(f'Val - Loss: {val_metrics["loss"]:.4f}, IoU: {val_metrics["iou"]:.4f}')
        
        # Save checkpoints
        if config['checkpoint']['save_last']:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics['loss'], val_metrics,
                checkpoint_dir / 'last.pth'
            )
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            if config['checkpoint']['save_best']:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_metrics['loss'], val_metrics,
                    checkpoint_dir / 'best.pth'
                )
                logging.info(f'New best model saved with val_loss: {best_val_loss:.4f}')
        
        # Early stopping
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            logging.info('Early stopping triggered')
            break
    
    writer.close()
    logging.info('Training completed')


if __name__ == '__main__':
    main()