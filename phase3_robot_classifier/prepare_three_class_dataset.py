#!/usr/bin/env python3
# ============================================================
# prepare_three_class_dataset.py
# Copy exactly 3 selected classes from clean_dataset into a
# new folder with train/val/test splits ready for fine-tuning.
#
# Usage:
#   python prepare_three_class_dataset.py \
#     --source-dir ../dataset/clean_dataset \
#     --output-dir ../dataset/phase3_three_classes \
#     --classes Redbull_Classic Redbull_zero Oxyshred_Passion
# ============================================================

import os
import sys
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset_utils import split_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare 3-class subset for robot deployment fine-tuning")
    parser.add_argument('--source-dir', default='../dataset/clean_dataset',
                        help='Path to full clean_dataset')
    parser.add_argument('--output-dir', default='../dataset/phase3_three_classes',
                        help='Output folder for 3-class split dataset')
    parser.add_argument('--classes', nargs='+',
                        default=['Redbull_Classic', 'Redbull_zero', 'Oxyshred_Passion'],
                        help='Exactly 3 class folder names from clean_dataset')
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio',   type=float, default=0.15)
    parser.add_argument('--test-ratio',  type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    print(f"\nSource dataset : {source_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Selected classes: {args.classes}\n")

    # Validate that all requested class folders exist
    for cls in args.classes:
        cls_path = source_dir / cls
        if not cls_path.exists():
            print(f"ERROR: Class folder not found: {cls_path}")
            print("Available folders:")
            for d in sorted(source_dir.iterdir()):
                if d.is_dir():
                    print(f"  {d.name}")
            sys.exit(1)

    # Create a temporary folder with only the 3 selected classes
    temp_dir = output_dir.parent / '_temp_three_class_source'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    print("Copying selected class folders...")
    for cls in args.classes:
        src = source_dir / cls
        dst = temp_dir / cls
        shutil.copytree(src, dst)
        n_images = sum(1 for f in dst.rglob('*')
                       if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'})
        print(f"  {cls}: {n_images} images")

    # Remove existing output if present
    if output_dir.exists():
        print(f"\nRemoving existing output dir: {output_dir}")
        shutil.rmtree(output_dir)

    # Split into train/val/test
    print(f"\nSplitting into train/val/test ...")
    split_dataset(
        source_dir=str(temp_dir),
        output_dir=str(output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Cleanup temp
    shutil.rmtree(temp_dir)

    # Summary
    print("\n--- Dataset Summary ---")
    for split in ('train', 'val', 'test'):
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        print(f"\n  {split}/")
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                n = sum(1 for f in cls_dir.iterdir()
                        if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'})
                print(f"    {cls_dir.name}: {n} images")

    print(f"\nDone! 3-class dataset ready at: {output_dir}")


if __name__ == '__main__':
    main()
