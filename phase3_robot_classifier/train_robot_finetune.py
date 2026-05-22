#!/usr/bin/env python3
# ============================================================
# train_robot_finetune.py
# STEP 3 — Fine-tune the all-class model on robot-captured
# images. All 20 classes. This adapts the model to the robot
# camera's image characteristics.
#
# Usage:
#   python train_robot_finetune.py \
#     --data-dir ../dataset/robot_split \
#     --base-model outputs/models/all_class_model.pth \
#     --epochs 20 --batch-size 16
# ============================================================

import os, sys, argparse, time, yaml
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset_utils import make_dataloaders, save_class_map
from utils.model_utils   import (build_model, get_device, save_checkpoint,
                                  freeze_backbone, unfreeze_all, load_checkpoint)
from utils.eval_utils    import save_training_curves


def parse_args():
    p = argparse.ArgumentParser(
        description='Fine-tune on robot-captured images — all 20 classes')
    p.add_argument('--data-dir',    default='../dataset/robot_split')
    p.add_argument('--base-model',  default='outputs/models/all_class_model.pth',
                   help='Start from this checkpoint (Phase 2 trained model)')
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--batch-size',  type=int,   default=16)
    p.add_argument('--lr',          type=float, default=0.001)
    p.add_argument('--arch',        default='mobilenet_v3_large')
    p.add_argument('--workers',     type=int,   default=2)
    p.add_argument('--config',      default='config.yaml')
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader,
                desc='  train' if train else '  val  ', leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if train:
                loss.backward(); optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += outputs.max(1)[1].eq(labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}

    for d in ['outputs/models','outputs/plots']:
        os.makedirs(d, exist_ok=True)

    device = get_device()

    print('\n[1/3] Building DataLoaders ...')
    loaders, class_to_idx, idx_to_class = make_dataloaders(
        args.data_dir, cfg.get('image_size', 224),
        args.batch_size, args.workers)
    num_classes = len(class_to_idx)
    print(f'  Classes: {num_classes}')

    # Save updated class map (robot images may have same classes)
    map_path = f"outputs/models/{cfg.get('robot_finetuned_map_name','robot_finetuned_class_to_idx.json')}"
    save_class_map(class_to_idx, map_path)

    print(f'\n[2/3] Loading base model: {args.base_model}')
    model = build_model(num_classes, args.arch, pretrained=True,
                        dropout=cfg.get('dropout', 0.3))

    if os.path.exists(args.base_model):
        ckpt  = torch.load(args.base_model, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        m_state = model.state_dict()
        # Load matching backbone layers only
        matched = {k: v for k, v in state.items()
                   if k in m_state and 'classifier' not in k
                   and m_state[k].shape == v.shape}
        m_state.update(matched)
        model.load_state_dict(m_state)
        print(f'  Loaded {len(matched)} backbone layers from base model.')
    else:
        print('  Base model not found — training from ImageNet weights.')

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    save_path = f"outputs/models/{cfg.get('robot_finetuned_model_name','robot_finetuned_model.pth')}"
    history   = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    best_acc  = 0.0

    print(f'\n[3/3] Two-phase fine-tuning ({args.epochs} epochs) ...')

    # Phase A — head only
    head_epochs = min(5, args.epochs // 3)
    freeze_backbone(model, args.arch)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=cfg.get('weight_decay', 0.0001))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f'  Phase A — head only ({head_epochs} epochs)')
    for epoch in range(1, head_epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, loaders['train'], criterion, optimizer, device, True)
        vl, va = run_epoch(model, loaders['val'],   criterion, None,      device, False)
        scheduler.step()
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_acc'].append(ta);  history['val_acc'].append(va)
        print(f'  Epoch {epoch:3d}/{args.epochs} | train_acc={ta:.4f} val_acc={va:.4f} | {time.time()-t0:.1f}s')
        if va > best_acc:
            best_acc = va
            save_checkpoint(model, class_to_idx, save_path, args.arch,
                            {'epoch': epoch, 'val_acc': va})
            print(f'    ✓ Best saved (val_acc={va:.4f})')

    # Phase B — full fine-tune
    remaining = args.epochs - head_epochs
    if remaining > 0:
        print(f'\n  Phase B — full fine-tune ({remaining} epochs)')
        unfreeze_all(model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1,
                               weight_decay=cfg.get('weight_decay', 0.0001))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=1e-7)
        for epoch in range(head_epochs + 1, args.epochs + 1):
            t0 = time.time()
            tl, ta = run_epoch(model, loaders['train'], criterion, optimizer, device, True)
            vl, va = run_epoch(model, loaders['val'],   criterion, None,      device, False)
            scheduler.step()
            history['train_loss'].append(tl); history['val_loss'].append(vl)
            history['train_acc'].append(ta);  history['val_acc'].append(va)
            print(f'  Epoch {epoch:3d}/{args.epochs} | train_acc={ta:.4f} val_acc={va:.4f} | {time.time()-t0:.1f}s')
            if va > best_acc:
                best_acc = va
                save_checkpoint(model, class_to_idx, save_path, args.arch,
                                {'epoch': epoch, 'val_acc': va})
                print(f'    ✓ Best saved (val_acc={va:.4f})')

    save_training_curves(history, 'outputs/plots', 'robot_finetune')
    print(f'\nDone. Best val acc: {best_acc:.4f}')
    print(f'Model → {save_path}')


if __name__ == '__main__':
    main()
