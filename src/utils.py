import os
import logging
import json
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def setup_logging(log_file: str = None) -> None:
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )


def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    
    intersection = (pred_mask & true_mask).float().sum()
    union = (pred_mask | true_mask).float().sum()
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    
    intersection = (pred_mask & true_mask).float().sum()
    total = pred_mask.float().sum() + true_mask.float().sum()
    
    dice = (2 * intersection + smooth) / (total + smooth)
    return dice.item()


def calculate_precision_recall(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> Tuple[float, float]:
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    
    tp = (pred_mask & true_mask).float().sum()
    fp = (pred_mask & ~true_mask).float().sum()
    fn = (~pred_mask & true_mask).float().sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return precision.item(), recall.item()


def apply_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Convert to numpy for easier manipulation
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    # Calculate areas
    areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
    
    # Sort by scores
    order = scores_np.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        # Pick the box with highest score
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(boxes_np[i, 0], boxes_np[order[1:], 0])
        yy1 = np.maximum(boxes_np[i, 1], boxes_np[order[1:], 1])
        xx2 = np.minimum(boxes_np[i, 2], boxes_np[order[1:], 2])
        yy2 = np.minimum(boxes_np[i, 3], boxes_np[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU less than threshold
        indices = np.where(iou <= iou_threshold)[0]
        order = order[indices + 1]
    
    return torch.tensor(keep, dtype=torch.long)


def mask_to_polygons(mask: np.ndarray, min_area: int = 100) -> List[List[float]]:
    polygons = []
    
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Approximate contour
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to polygon format
        if len(approx) >= 3:
            polygon = approx.reshape(-1, 2).flatten().tolist()
            polygons.append(polygon)
    
    return polygons


def visualize_predictions(image: np.ndarray, 
                         pred_mask: np.ndarray, 
                         true_mask: np.ndarray = None,
                         save_path: str = None) -> None:
    if true_mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(pred_mask, cmap='hot')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(pred_mask, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # True mask if provided
    if true_mask is not None:
        axes[3].imshow(true_mask, cmap='hot')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_json(data: Any, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Any:
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir(directory: str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)


def count_files(directory: str, extension: str = None) -> int:
    path = Path(directory)
    if extension:
        return len(list(path.glob(f"*.{extension}")))
    else:
        return len(list(path.iterdir()))


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0, 
                 path: str = 'checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss: float, model: torch.nn.Module = None) -> None:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if model is not None:
                self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class MetricTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_map(pred_boxes: List[np.ndarray], 
                  pred_scores: List[np.ndarray],
                  true_boxes: List[np.ndarray],
                  iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]) -> float:
    """Calculate mean Average Precision (mAP) for object detection."""
    
    def calculate_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix between two sets of boxes."""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        xx1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
        yy1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
        xx2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
        yy2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        return iou
    
    aps = []
    
    for iou_threshold in iou_thresholds:
        tp_fp = []
        
        for pred_box, pred_score, true_box in zip(pred_boxes, pred_scores, true_boxes):
            if len(pred_box) == 0:
                continue
            
            # Sort predictions by score
            order = np.argsort(pred_score)[::-1]
            pred_box = pred_box[order]
            pred_score = pred_score[order]
            
            if len(true_box) == 0:
                # All predictions are false positives
                tp_fp.extend([(score, 0) for score in pred_score])
                continue
            
            # Calculate IoU matrix
            iou_matrix = calculate_iou_matrix(pred_box, true_box)
            
            # Match predictions to ground truth
            matched_gt = set()
            for i, (box, score) in enumerate(zip(pred_box, pred_score)):
                if len(true_box) == 0:
                    tp_fp.append((score, 0))  # False positive
                    continue
                
                # Find best matching ground truth
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx in range(len(true_box)):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = iou_matrix[i, gt_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    tp_fp.append((score, 1))  # True positive
                    matched_gt.add(best_gt_idx)
                else:
                    tp_fp.append((score, 0))  # False positive
        
        if len(tp_fp) == 0:
            aps.append(0.0)
            continue
        
        # Sort by score
        tp_fp.sort(key=lambda x: x[0], reverse=True)
        
        # Calculate precision and recall
        tp = np.cumsum([x[1] for x in tp_fp])
        fp = np.cumsum([1 - x[1] for x in tp_fp])
        
        total_gt = sum(len(boxes) for boxes in true_boxes)
        
        precision = tp / (tp + fp)
        recall = tp / total_gt if total_gt > 0 else np.zeros_like(tp)
        
        # Calculate AP using precision-recall curve
        ap = average_precision_score(np.ones_like(recall), precision) if len(precision) > 0 else 0.0
        aps.append(ap)
    
    return np.mean(aps)