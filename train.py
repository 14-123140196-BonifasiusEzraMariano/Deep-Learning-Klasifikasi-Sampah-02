"""Training ResNet-50 untuk klasifikasi sampah (5 kelas).

Contoh:
    # dataset satu folder (otomatis dibagi 80/20)
    python train.py --data ./dataset

    # dataset sudah punya train/ dan val/
    python train.py --data ./dataset/train --val-data ./dataset/val

Selalu mulai dari awal (tidak ada resume). Model terbaik (Macro-F1 val tertinggi)
disimpan ke submission/model.pth.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission"))
from model import NUM_CLASSES, get_model, get_transform  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    train_ds = datasets.ImageFolder(args.data, transform=get_transform(train=True))
    if len(train_ds.classes) != NUM_CLASSES:
        raise ValueError(f"Ditemukan {len(train_ds.classes)} kelas, model.py mengharapkan {NUM_CLASSES}.")

    if args.val_data:
        val_ds = datasets.ImageFolder(args.val_data, transform=get_transform(train=False))
        if val_ds.classes != train_ds.classes:
            raise ValueError("Nama/urutan kelas train dan val berbeda.")
        train_set = train_ds
    else:
        # split stratified per kelas, pakai dua dataset agar val tanpa augmentasi
        full_val = datasets.ImageFolder(args.data, transform=get_transform(train=False))
        targets = np.array(train_ds.targets)
        rng = np.random.RandomState(args.seed)
        tr_idx, va_idx = [], []
        for c in range(NUM_CLASSES):
            idx = np.where(targets == c)[0]
            rng.shuffle(idx)
            n_val = max(1, int(len(idx) * args.val_split))
            va_idx += idx[:n_val].tolist()
            tr_idx += idx[n_val:].tolist()
        train_set = Subset(train_ds, tr_idx)
        val_ds = Subset(full_val, va_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    train_targets = np.array(train_ds.targets)
    if not args.val_data:
        train_targets = train_targets[tr_idx]
    counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    return train_loader, val_loader, train_ds.classes, counts


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses, y_true, y_pred = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            y_true += y.cpu().tolist()
            y_pred += out.argmax(1).cpu().tolist()
    return float(np.mean(losses)), f1_score(y_true, y_pred, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--freeze-epochs", type=int, default=2, help="epoch awal: hanya latih head")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="LR head")
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="submission/model.pth")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, classes, counts = build_loaders(args)
    print("Kelas:", classes, "| jumlah train per kelas:", counts.tolist())

    # class weights (membantu Macro-F1 kalau kelas tidak seimbang)
    weights = torch.tensor(counts.sum() / (NUM_CLASSES * np.maximum(counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    model = get_model(pretrained=True).to(device)  # selalu model baru -> mulai dari awal
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_f1 = -1.0

    for epoch in range(args.epochs):
        freeze = epoch < args.freeze_epochs
        for p in backbone_params:
            p.requires_grad = not freeze

        tr_loss, tr_f1 = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_f1 = run_epoch(model, val_loader, device, criterion)
        scheduler.step()

        mark = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), args.out)
            mark = "  <- model terbaik disimpan"
        print(f"Epoch {epoch + 1:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} F1 {tr_f1:.4f} | "
              f"val loss {va_loss:.4f} F1 {va_f1:.4f}{mark}")

    print(f"\nSelesai. Best val Macro-F1 = {best_f1:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
