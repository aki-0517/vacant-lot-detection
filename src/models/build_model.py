import torch
import torch.nn as nn
from typing import Dict, Any
from .unet import UNet, UNetWithBBoxHead, DeepLabV3Plus, FPN, CombinedLoss


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_type = config['model']['type'].lower()
    
    model_params = {
        'encoder_name': config['model']['backbone'],
        'encoder_weights': 'imagenet' if config['model']['pretrained'] else None,
        'num_classes': config['model']['num_classes']
    }
    
    if model_type == 'unet':
        model = UNet(**model_params)
    elif model_type == 'unet_with_bbox':
        model = UNetWithBBoxHead(**model_params)
    elif model_type == 'deeplabv3plus':
        model = DeepLabV3Plus(**model_params)
    elif model_type == 'fpn':
        model = FPN(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def build_loss_function(config: Dict[str, Any]) -> nn.Module:
    loss_type = config.get('loss', {}).get('type', 'combined')
    
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        from .unet import dice_loss
        return dice_loss
    elif loss_type == 'focal':
        from .unet import focal_loss
        return focal_loss
    elif loss_type == 'combined':
        return CombinedLoss(
            dice_weight=config.get('loss', {}).get('dice_weight', 0.5),
            focal_weight=config.get('loss', {}).get('focal_weight', 0.5)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_type = config.get('optimizer', {}).get('type', 'adam')
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        momentum = config.get('optimizer', {}).get('momentum', 0.9)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler_type = config['training'].get('scheduler', 'cosine')
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['lr'] * 0.01
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['epochs'] // 3,
            gamma=0.1
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config['training'].get('early_stopping_patience', 10) // 2,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return checkpoint


def save_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   epoch: int,
                   loss: float,
                   metrics: Dict[str, float],
                   filepath: str) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_encoder(model: nn.Module) -> None:
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        for param in model.model.encoder.parameters():
            param.requires_grad = False


def unfreeze_encoder(model: nn.Module) -> None:
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = True
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        for param in model.model.encoder.parameters():
            param.requires_grad = True