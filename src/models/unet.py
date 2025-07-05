import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import segmentation_models_pytorch as smp


class UNet(nn.Module):
    def __init__(self, 
                 encoder_name: str = 'resnet34',
                 encoder_weights: str = 'imagenet',
                 num_classes: int = 1,
                 activation: str = None):
        super(UNet, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetWithBBoxHead(nn.Module):
    def __init__(self, 
                 encoder_name: str = 'resnet34',
                 encoder_weights: str = 'imagenet',
                 num_classes: int = 1,
                 num_bbox_classes: int = 1):
        super(UNetWithBBoxHead, self).__init__()
        
        # Segmentation backbone
        self.segmentation_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation='sigmoid'
        )
        
        # Get encoder output channels
        encoder = smp.encoders.get_encoder(encoder_name, encoder_weights)
        self.encoder_channels = encoder.out_channels
        
        # Bbox detection head
        self.bbox_head = BBoxHead(
            in_channels=self.encoder_channels[-1],
            num_classes=num_bbox_classes
        )
        
        # Share encoder between segmentation and bbox detection
        self.encoder = encoder
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get encoder features
        features = self.encoder(x)
        
        # Segmentation output
        seg_output = self.segmentation_model.decoder(*features)
        seg_output = self.segmentation_model.segmentation_head(seg_output)
        
        # Bbox detection output
        bbox_output = self.bbox_head(features[-1])
        
        return {
            'segmentation': seg_output,
            'bbox': bbox_output
        }


class BBoxHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(BBoxHead, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, 3, padding=1)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification and regression heads
        self.cls_head = nn.Linear(128, num_classes)
        self.reg_head = nn.Linear(128, 4)  # x1, y1, x2, y2
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        
        # Classification and regression
        cls_output = self.cls_head(x)
        reg_output = self.reg_head(x)
        
        return {
            'classification': cls_output,
            'regression': reg_output
        }


class DeepLabV3Plus(nn.Module):
    def __init__(self, 
                 encoder_name: str = 'resnet34',
                 encoder_weights: str = 'imagenet',
                 num_classes: int = 1,
                 activation: str = None):
        super(DeepLabV3Plus, self).__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FPN(nn.Module):
    def __init__(self, 
                 encoder_name: str = 'resnet34',
                 encoder_weights: str = 'imagenet',
                 num_classes: int = 1,
                 activation: str = None):
        super(FPN, self).__init__()
        
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + 
                                                target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def focal_loss(pred: torch.Tensor, target: torch.Tensor, 
               alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, 
                 dice_weight: float = 0.5,
                 focal_weight: float = 0.5,
                 smooth: float = 1.0,
                 alpha: float = 1.0,
                 gamma: float = 2.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = dice_loss(torch.sigmoid(pred), target, self.smooth)
        focal = focal_loss(pred, target, self.alpha, self.gamma)
        
        return self.dice_weight * dice + self.focal_weight * focal