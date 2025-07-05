import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BBoxDataset(Dataset):
    def __init__(self, 
                 images_dir: str, 
                 annotations_path: str,
                 transforms: Optional[A.Compose] = None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = {img['id']: img for img in self.annotations['images']}
        self.image_ids = list(self.images.keys())
        
        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        annotations = self.annotations_by_image.get(img_id, [])
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            # Convert to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
        
        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': img_id
        }


class SegmentationDataset(Dataset):
    def __init__(self, 
                 images_dir: str, 
                 masks_dir: str,
                 annotations_path: str,
                 transforms: Optional[A.Compose] = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = {img['id']: img for img in self.annotations['images']}
        self.image_ids = list(self.images.keys())
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_filename = f"{img_info['file_name'].split('.')[0]}_mask.png"
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
        else:
            # Create empty mask if not found
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            'image_id': img_id
        }


class EvaluationDataset(Dataset):
    def __init__(self, 
                 images_dir: str,
                 transforms: Optional[A.Compose] = None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']:
            self.image_files.extend(Path(images_dir).glob(ext))
        
        self.image_files = sorted([str(f) for f in self.image_files])
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_files[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'image_path': img_path,
            'image_name': os.path.basename(img_path)
        }


def get_train_transforms(config: Dict) -> A.Compose:
    transforms = [
        A.HorizontalFlip(p=config['augmentation']['horizontal_flip']),
        A.VerticalFlip(p=config['augmentation']['vertical_flip']),
        A.Rotate(limit=config['augmentation']['rotation_limit'], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=config['augmentation']['brightness_limit'],
            contrast_limit=config['augmentation']['contrast_limit'],
            p=0.5
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def get_val_transforms() -> A.Compose:
    transforms = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def get_bbox_transforms(config: Dict) -> A.Compose:
    transforms = [
        A.HorizontalFlip(p=config['augmentation']['horizontal_flip']),
        A.VerticalFlip(p=config['augmentation']['vertical_flip']),
        A.Rotate(limit=config['augmentation']['rotation_limit'], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=config['augmentation']['brightness_limit'],
            contrast_limit=config['augmentation']['contrast_limit'],
            p=0.5
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    return A.Compose(transforms, bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))


def collate_fn(batch: List[Dict]) -> Dict:
    images = torch.stack([item['image'] for item in batch])
    
    if 'mask' in batch[0]:
        masks = torch.stack([item['mask'] for item in batch])
        return {
            'images': images,
            'masks': masks,
            'image_ids': [item['image_id'] for item in batch]
        }
    elif 'boxes' in batch[0]:
        return {
            'images': images,
            'boxes': [item['boxes'] for item in batch],
            'labels': [item['labels'] for item in batch],
            'image_ids': [item['image_id'] for item in batch]
        }
    else:
        return {
            'images': images,
            'image_paths': [item['image_path'] for item in batch],
            'image_names': [item['image_name'] for item in batch]
        }