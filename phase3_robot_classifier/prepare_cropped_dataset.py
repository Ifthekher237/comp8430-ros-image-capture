#!/usr/bin/env python3
# ============================================================
# prepare_cropped_dataset.py
#
# Runs YOLOv8n on every image in your clean_dataset.
# Saves ONLY the cropped can region from each image.
# Organises crops into the same class folder structure.
#
# This means your classifier trains on:
#   → just the can pixels
#   → no hands, no background, no person
#   → exactly matching what YOLO feeds it at inference time
#
# Usage:
#   python prepare_cropped_dataset.py \
#     --source-dir ../dataset/clean_dataset \
#     --output-dir ../dataset/cropped_dataset
#
# Then retrain:
#   python train_all_classes.py \
#     --data-dir ../dataset/cropped_dataset \
#     --epochs 20
# ============================================================

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# YOLO classes that correspond to drink cans
CAN_CLASSES = {
    'bottle',
    'cup',
    'wine glass',
    'vase',
    'sports ball',
    'bowl',
}

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Minimum crop size in pixels — crops smaller than this are skipped
# (too blurry / too far away to be useful for training)
MIN_CROP_SIZE = 64

# Padding added around each detected can crop
CROP_PADDING = 10


def parse_args():
    p = argparse.ArgumentParser(
        description='Crop cans from training images using YOLOv8')
    p.add_argument('--source-dir', default='../dataset/clean_dataset',
                   help='Original dataset with one folder per class')
    p.add_argument('--output-dir', default='../dataset/cropped_dataset',
                   help='Output folder — same structure, crops only')
    p.add_argument('--yolo-conf',  type=float, default=0.15,
                   help='YOLO confidence threshold (lower = find more cans)')
    p.add_argument('--min-size',   type=int,   default=MIN_CROP_SIZE,
                   help='Skip crops smaller than this (px)')
    p.add_argument('--padding',    type=int,   default=CROP_PADDING,
                   help='Pixels of padding around each crop')
    p.add_argument('--fallback-full', action='store_true',
                   help='If YOLO finds nothing, save full image as fallback')
    return p.parse_args()


def load_yolo():
    try:
        from ultralytics import YOLO
        print('Loading YOLOv8n ...')
        model = YOLO('yolov8n.pt')
        print('  YOLOv8n ready.')
        return model
    except ImportError:
        print('ERROR: ultralytics not installed.')
        print('Run: pip install ultralytics')
        sys.exit(1)


def get_can_crops(image_bgr, yolo_model, conf_threshold,
                  padding, min_size, fallback_full):
    """
    Run YOLO on one image.
    Returns list of cropped BGR numpy arrays — one per detected can.
    If nothing found and fallback_full=True, returns the full image.
    """
    h, w = image_bgr.shape[:2]
    results = yolo_model(image_bgr, verbose=False, conf=conf_threshold)[0]

    crops = []
    for box in results.boxes:
        cls_name = results.names[int(box.cls[0])]
        if cls_name not in CAN_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Add padding — clipped to image bounds
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)

        bw = x2 - x1
        bh = y2 - y1

        # Skip tiny crops — too blurry to be useful
        if bw < min_size or bh < min_size:
            continue

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)

    # Fallback — if YOLO found nothing, use full image
    # This preserves images that YOLO cannot detect but are valid
    if len(crops) == 0 and fallback_full:
        crops.append(image_bgr)

    return crops


def process_class(class_dir, output_dir, yolo_model,
                  conf_threshold, padding, min_size,
                  fallback_full):
    """
    Process all images in one class folder.
    Returns (processed, saved, skipped) counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [f for f in class_dir.iterdir()
              if f.suffix.lower() in IMG_EXTS]

    processed = 0
    saved     = 0
    skipped   = 0
    crop_idx  = 0

    for img_path in tqdm(images, desc=f'  {class_dir.name}',
                         leave=False):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            skipped += 1
            continue

        crops = get_can_crops(
            image_bgr, yolo_model,
            conf_threshold, padding,
            min_size, fallback_full)

        processed += 1

        if len(crops) == 0:
            skipped += 1
            continue

        for crop in crops:
            # Save crop with original stem + crop index
            out_name = f"{img_path.stem}_crop{crop_idx:04d}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop_idx += 1
            saved    += 1

    return processed, saved, skipped


def main():
    args = parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        print(f'ERROR: Source directory not found: {source_dir}')
        sys.exit(1)

    print(f'\nSource  : {source_dir}')
    print(f'Output  : {output_dir}')
    print(f'YOLO conf threshold : {args.yolo_conf}')
    print(f'Min crop size       : {args.min_size}px')
    print(f'Padding             : {args.padding}px')
    print(f'Fallback full image : {args.fallback_full}')
    print()

    yolo_model = load_yolo()

    # Get all class folders
    class_dirs = sorted([d for d in source_dir.iterdir()
                         if d.is_dir()])
    if not class_dirs:
        print(f'ERROR: No class folders found in {source_dir}')
        sys.exit(1)

    print(f'Found {len(class_dirs)} classes.\n')

    # Summary tracking
    total_processed = 0
    total_saved     = 0
    total_skipped   = 0
    class_summary   = []

    for class_dir in class_dirs:
        out_class_dir = output_dir / class_dir.name
        processed, saved, skipped = process_class(
            class_dir, out_class_dir, yolo_model,
            args.yolo_conf, args.padding,
            args.min_size, args.fallback_full)

        total_processed += processed
        total_saved     += saved
        total_skipped   += skipped
        class_summary.append((class_dir.name, processed, saved, skipped))

    # Print summary table
    print('\n' + '='*60)
    print(f'{"Class":<25} {"Images":>8} {"Crops":>8} {"Skipped":>8}')
    print('-'*60)
    for cls_name, proc, sav, skip in class_summary:
        print(f'{cls_name:<25} {proc:>8} {sav:>8} {skip:>8}')
    print('='*60)
    print(f'{"TOTAL":<25} {total_processed:>8} {total_saved:>8} {total_skipped:>8}')
    print()

    if total_saved == 0:
        print('WARNING: No crops were saved!')
        print('  YOLO could not detect cans in any image.')
        print('  Try: --yolo-conf 0.05')
        print('  Or:  --fallback-full  (uses full image when YOLO finds nothing)')
    else:
        skip_pct = (total_skipped / total_processed * 100
                    if total_processed > 0 else 0)
        print(f'Cropped dataset ready at: {output_dir}')
        print(f'  {total_saved} crops saved from {total_processed} images')
        print(f'  {total_skipped} images skipped ({skip_pct:.1f}%) — YOLO found no can')
        print()
        if skip_pct > 30:
            print('TIP: Many images were skipped. Consider running with:')
            print('  --fallback-full   to keep images where YOLO finds nothing')
            print('  --yolo-conf 0.10  to detect more cans at lower confidence')

    print('\nNext step — retrain on cropped dataset:')
    print(f'  python train_all_classes.py \\')
    print(f'    --data-dir {output_dir} \\')
    print(f'    --epochs 20 \\')
    print(f'    --batch-size 16')


if __name__ == '__main__':
    main()
