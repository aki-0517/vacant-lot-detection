import os
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_image_info(image_path: str) -> Dict[str, Any]:
    """Get image dimensions"""
    with Image.open(image_path) as img:
        width, height = img.size
    return {"width": width, "height": height}


def create_bbox_submission_correct(bbox_eval_dir: str, output_path: str):
    """Create bbox submission with correct Solafune format"""
    image_files = [f for f in os.listdir(bbox_eval_dir) if f.endswith('.tif')]
    image_files.sort()
    
    submission = {
        "images": []
    }
    
    for image_file in image_files:
        image_path = os.path.join(bbox_eval_dir, image_file)
        img_info = get_image_info(image_path)
        
        # Create image entry with annotations
        image_entry = {
            "file_name": image_file,
            "width": img_info["width"],
            "height": img_info["height"],
            "annotations": [
                {
                    "class": "vacant_lot",
                    "bbox": [50, 50, 100, 100]  # [x, y, w, h]
                }
            ]
        }
        
        submission["images"].append(image_entry)
    
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=4)
    
    print(f'Created bbox submission with {len(submission["images"])} images: {output_path}')


def create_segmentation_submission_correct(seg_eval_dir: str, output_path: str):
    """Create segmentation submission with correct Solafune format"""
    image_files = [f for f in os.listdir(seg_eval_dir) if f.endswith('.tif')]
    image_files.sort()
    
    submission = {
        "images": []
    }
    
    for image_file in image_files:
        image_path = os.path.join(seg_eval_dir, image_file)
        img_info = get_image_info(image_path)
        
        # Create image entry with annotations
        image_entry = {
            "file_name": image_file,
            "width": img_info["width"],
            "height": img_info["height"],
            "annotations": [
                {
                    "class": "vacant_lot",
                    "segmentation": [100, 100, 200, 100, 200, 200, 100, 200]  # polygon points
                }
            ]
        }
        
        submission["images"].append(image_entry)
    
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=4)
    
    print(f'Created segmentation submission with {len(submission["images"])} images: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Create correct Solafune submission files')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--output_dir', type=str, default='data/submissions')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get evaluation directories
    bbox_eval_dir = os.path.join(config['data']['bbox_dir'], 'eval_images')
    seg_eval_dir = os.path.join(config['data']['segmentation_dir'], 'eval_images')
    
    # Create submission files
    bbox_path = output_dir / 'bbox.json'
    seg_path = output_dir / 'segmentation.json'
    
    create_bbox_submission_correct(bbox_eval_dir, bbox_path)
    create_segmentation_submission_correct(seg_eval_dir, seg_path)
    
    print(f'Correct submission files created in: {output_dir}')


if __name__ == '__main__':
    main()