import os
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_image_info(image_path: str) -> Dict[str, Any]:
    """Get image dimensions"""
    with Image.open(image_path) as img:
        width, height = img.size
    return {"width": width, "height": height}


def create_bbox_submission_fixed(bbox_eval_dir: str, output_path: str):
    """Create bbox submission with strict format compliance"""
    image_files = [f for f in os.listdir(bbox_eval_dir) if f.endswith('.tif')]
    image_files.sort()
    
    # Create the submission structure exactly as required
    submission_data = {
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
        
        submission_data["images"].append(image_entry)
    
    # Write with explicit encoding and formatting
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(submission_data, f, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    print(f'Created bbox submission with {len(submission_data["images"])} images: {output_path}')
    return submission_data


def create_segmentation_submission_fixed(seg_eval_dir: str, output_path: str):
    """Create segmentation submission with strict format compliance"""
    image_files = [f for f in os.listdir(seg_eval_dir) if f.endswith('.tif')]
    image_files.sort()
    
    # Create the submission structure exactly as required
    submission_data = {
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
        
        submission_data["images"].append(image_entry)
    
    # Write with explicit encoding and formatting
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(submission_data, f, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    print(f'Created segmentation submission with {len(submission_data["images"])} images: {output_path}')
    return submission_data


def validate_submission_format(file_path: str) -> bool:
    """Validate the submission format"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if 'images' key exists and is a list
        if 'images' not in data:
            print(f"ERROR: Missing 'images' key in {file_path}")
            return False
        
        if not isinstance(data['images'], list):
            print(f"ERROR: 'images' is not a list in {file_path}")
            return False
        
        # Check structure of first image
        if len(data['images']) > 0:
            first_image = data['images'][0]
            required_keys = ['file_name', 'width', 'height', 'annotations']
            for key in required_keys:
                if key not in first_image:
                    print(f"ERROR: Missing '{key}' in image entry")
                    return False
            
            if not isinstance(first_image['annotations'], list):
                print(f"ERROR: 'annotations' is not a list")
                return False
        
        print(f"✓ {file_path} format is valid with {len(data['images'])} images")
        return True
        
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to validate {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Fix submission file format')
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
    
    print("Creating fixed submission files...")
    bbox_data = create_bbox_submission_fixed(bbox_eval_dir, bbox_path)
    seg_data = create_segmentation_submission_fixed(seg_eval_dir, seg_path)
    
    print("\nValidating submission files...")
    bbox_valid = validate_submission_format(bbox_path)
    seg_valid = validate_submission_format(seg_path)
    
    if bbox_valid and seg_valid:
        print("\n✓ All submission files are valid!")
        print(f"Submission files created in: {output_dir}")
    else:
        print("\n✗ Some submission files have format issues")


if __name__ == '__main__':
    main()