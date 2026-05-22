# ============================================================
# utils/inference_utils.py
# Shared ROI contour detection + classification pipeline.
# Used by webcam_test_all_classes.py AND all ROS2 nodes.
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms


def get_inference_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------
# ROI detector — returns list of (x, y, w, h) bounding boxes
# ------------------------------------------------------------------

def detect_rois(frame_bgr, blur_kernel=7, canny_low=30,
                canny_high=100, dilate_iters=3, min_area=4000,
                max_area_ratio=0.85, padding=10):
    """
    Detect candidate object regions using Canny edges + contours.
    Returns list of (x, y, w, h) tuples clipped to frame bounds.
    NOTE: contour-based detection, not a trained object detector.
    """
    h, w  = frame_bgr.shape[:2]
    frame_area = h * w

    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges   = cv2.Canny(blurred, canny_low, canny_high)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=dilate_iters)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > frame_area * max_area_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)
        rois.append((x1, y1, x2 - x1, y2 - y1))

    return _nms(rois)


def _nms(rois, overlap_thresh=0.4):
    if not rois:
        return rois
    rois_sorted = sorted(rois, key=lambda r: r[2]*r[3], reverse=True)
    kept = []
    for box in rois_sorted:
        x1, y1, w1, h1 = box
        dominated = False
        for kx, ky, kw, kh in kept:
            ix1 = max(x1, kx); iy1 = max(y1, ky)
            ix2 = min(x1+w1, kx+kw); iy2 = min(y1+h1, ky+kh)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            union = w1*h1 + kw*kh - inter
            if union > 0 and inter/union > overlap_thresh:
                dominated = True; break
        if not dominated:
            kept.append(box)
    return kept


# ------------------------------------------------------------------
# Classify a single crop
# ------------------------------------------------------------------

def classify_crop(crop_bgr, model, idx_to_class, transform, device):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = PILImage.fromarray(crop_rgb)
    tensor   = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
    return idx_to_class[pred_idx.item()], conf.item()


# ------------------------------------------------------------------
# Full pipeline: detect + classify all ROIs
# ------------------------------------------------------------------

def detect_and_classify(frame_bgr, model, idx_to_class,
                         transform, device,
                         confidence_threshold=0.70,
                         blur_kernel=7, canny_low=30,
                         canny_high=100, dilate_iters=3,
                         min_area=4000, max_area_ratio=0.85,
                         padding=10):
    """
    Returns list of dicts:
      { bbox, label, confidence, above_threshold }
    """
    rois = detect_rois(frame_bgr, blur_kernel, canny_low,
                       canny_high, dilate_iters, min_area,
                       max_area_ratio, padding)
    detections = []
    for (x, y, w, h) in rois:
        crop = frame_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        label, conf = classify_crop(
            crop, model, idx_to_class, transform, device)
        detections.append({
            'bbox':            (x, y, w, h),
            'label':           label,
            'confidence':      conf,
            'above_threshold': conf >= confidence_threshold,
        })
    return detections


# ------------------------------------------------------------------
# Draw detections on frame
# ------------------------------------------------------------------

def draw_detections(frame_bgr, detections,
                    confidence_threshold=0.70):
    """
    Green box  = above threshold
    Orange box = below threshold
    """
    out = frame_bgr.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        label  = det['label']
        conf   = det['confidence']
        above  = det['above_threshold']
        color  = (0, 200, 0) if above else (0, 140, 255)
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        text = f"{label}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(text, font, 0.55, 1)
        ly = max(y - 5, th + 5)
        cv2.rectangle(out,
                      (x, ly - th - bl - 4),
                      (x + tw + 4, ly + bl - 4),
                      color, cv2.FILLED)
        cv2.putText(out, text, (x+2, ly - bl - 2),
                    font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out
