import os
import json
import zipfile
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_zip(zip_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def load_annotations(annotation_path: str) -> Dict:
    with open(annotation_path, 'r') as f:
        return json.load(f)


def polygon_to_mask(polygon: List[List[float]], image_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


def process_bbox_data(config: Dict) -> None:
    print("Processing bbox data...")
    
    raw_dir = config['data']['raw_dir']
    bbox_dir = config['data']['bbox_dir']
    
    # Extract images
    bbox_images_zip = os.path.join(raw_dir, config['paths']['train_bbox_images'])
    bbox_images_extract = os.path.join(bbox_dir, 'images')
    os.makedirs(bbox_images_extract, exist_ok=True)
    
    if os.path.exists(bbox_images_zip):
        extract_zip(bbox_images_zip, bbox_images_extract)
    
    # Process annotations
    bbox_annotations_path = os.path.join(raw_dir, config['paths']['train_bbox_annotations'])
    if os.path.exists(bbox_annotations_path):
        annotations = load_annotations(bbox_annotations_path)
        
        # Save processed annotations
        output_annotations_path = os.path.join(bbox_dir, 'annotations.json')
        with open(output_annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    print(f"Bbox data processed and saved to {bbox_dir}")


def process_segmentation_data(config: Dict) -> None:
    print("Processing segmentation data...")
    
    raw_dir = config['data']['raw_dir']
    seg_dir = config['data']['segmentation_dir']
    
    # Extract images
    seg_images_zip = os.path.join(raw_dir, config['paths']['train_segmentation_images'])
    seg_images_extract = os.path.join(seg_dir, 'images')
    os.makedirs(seg_images_extract, exist_ok=True)
    
    if os.path.exists(seg_images_zip):
        extract_zip(seg_images_zip, seg_images_extract)
    
    # Process annotations and create masks
    seg_annotations_path = os.path.join(raw_dir, config['paths']['train_segmentation_annotations'])
    if os.path.exists(seg_annotations_path):
        annotations = load_annotations(seg_annotations_path)
        
        # Create masks directory
        masks_dir = os.path.join(seg_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        # Process each image and create corresponding mask
        for img_idx, img_info in enumerate(tqdm(annotations.get('images', []), desc="Creating masks")):
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Get annotations directly from image info
            image_annotations = img_info.get('annotations', [])
            
            if image_annotations:
                # Create combined mask for all annotations in this image
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                for ann in image_annotations:
                    if 'segmentation' in ann:
                        polygon = ann['segmentation']
                        # Convert polygon to mask
                        poly_mask = polygon_to_mask(
                            np.array(polygon).reshape(-1, 2), 
                            (img_height, img_width)
                        )
                        mask = np.maximum(mask, poly_mask)
                
                # Save mask
                mask_filename = f"{img_info['file_name'].split('.')[0]}_mask.png"
                mask_path = os.path.join(masks_dir, mask_filename)
                cv2.imwrite(mask_path, mask * 255)
        
        # Save processed annotations
        output_annotations_path = os.path.join(seg_dir, 'annotations.json')
        with open(output_annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    print(f"Segmentation data processed and saved to {seg_dir}")


def process_evaluation_data(config: Dict) -> None:
    print("Processing evaluation data...")
    
    raw_dir = config['data']['raw_dir']
    
    # Extract evaluation bbox images
    eval_bbox_zip = os.path.join(raw_dir, config['paths']['evaluation_bbox_images'])
    eval_bbox_dir = os.path.join(config['data']['bbox_dir'], 'eval_images')
    os.makedirs(eval_bbox_dir, exist_ok=True)
    
    if os.path.exists(eval_bbox_zip):
        extract_zip(eval_bbox_zip, eval_bbox_dir)
    
    # Extract evaluation segmentation images
    eval_seg_zip = os.path.join(raw_dir, config['paths']['evaluation_segmentation_images'])
    eval_seg_dir = os.path.join(config['data']['segmentation_dir'], 'eval_images')
    os.makedirs(eval_seg_dir, exist_ok=True)
    
    if os.path.exists(eval_seg_zip):
        extract_zip(eval_seg_zip, eval_seg_dir)
    
    print("Evaluation data processed")


def main():
    parser = argparse.ArgumentParser(description='Preprocess land vacancy detection data')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['bbox', 'segmentation', 'both'],
                       default='both', help='Processing mode')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.mode in ['bbox', 'both']:
        process_bbox_data(config)
    
    if args.mode in ['segmentation', 'both']:
        process_segmentation_data(config)
    
    process_evaluation_data(config)
    
    print("Preprocessing completed!")


if __name__ == "__main__":
    main()