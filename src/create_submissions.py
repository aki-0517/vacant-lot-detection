import os
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_image_names(eval_dir: str) -> List[str]:
    """Get all image names from evaluation directory"""
    image_names = []
    for filename in os.listdir(eval_dir):
        if filename.endswith('.tif'):
            image_names.append(filename)
    return sorted(image_names)


def create_bbox_submission(bbox_eval_dir: str, output_path: str):
    """Create bbox submission with placeholder predictions"""
    image_names = get_image_names(bbox_eval_dir)
    
    submission = []
    for i, image_name in enumerate(image_names):
        # Create a minimal bbox prediction for each image
        submission.append({
            'id': i,
            'image_name': image_name,
            'bbox': [50.0, 50.0, 100.0, 100.0],  # [x, y, w, h]
            'score': 0.8,
            'category_id': 1
        })
    
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f'Created bbox submission with {len(submission)} predictions: {output_path}')


def create_segmentation_submission(seg_eval_dir: str, output_path: str):
    """Create segmentation submission with placeholder predictions"""
    image_names = get_image_names(seg_eval_dir)
    
    submission = []
    for i, image_name in enumerate(image_names):
        # Create a minimal segmentation prediction for each image
        submission.append({
            'id': i,
            'image_name': image_name,
            'segmentation': [100, 100, 200, 100, 200, 200, 100, 200],  # polygon points
            'category_id': 1
        })
    
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f'Created segmentation submission with {len(submission)} predictions: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Create submission files')
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
    
    create_bbox_submission(bbox_eval_dir, bbox_path)
    create_segmentation_submission(seg_eval_dir, seg_path)
    
    print(f'Submission files created in: {output_dir}')


if __name__ == '__main__':
    main()