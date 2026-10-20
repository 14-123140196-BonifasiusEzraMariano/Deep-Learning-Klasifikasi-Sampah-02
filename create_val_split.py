"""Buat folder my_val/ berisi gambar val split yang SAMA persis seperti yang dipakai train.py.

Pemakaian:
    python create_val_split.py --data ./dataset --out ./my_val
"""
import argparse
import os
import shutil

import numpy as np
from torchvision import datasets

NUM_CLASSES = 5
SEED = 42
VAL_SPLIT = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset")
    ap.add_argument("--out", default="./my_val")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-split", type=float, default=VAL_SPLIT)
    args = ap.parse_args()

    ds = datasets.ImageFolder(args.data)
    targets = np.array(ds.targets)
    rng = np.random.RandomState(args.seed)

    va_idx = []
    for c in range(NUM_CLASSES):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * args.val_split))
        va_idx += idx[:n_val].tolist()

    print(f"Total val images: {len(va_idx)}")

    for i in va_idx:
        src_path, label_idx = ds.imgs[i]
        class_name = ds.classes[label_idx]
        dst_dir = os.path.join(args.out, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_path, os.path.join(dst_dir, os.path.basename(src_path)))

    print(f"Val split disimpan ke: {args.out}")

    # Tampilkan jumlah per kelas
    for c in ds.classes:
        n = len(os.listdir(os.path.join(args.out, c)))
        print(f"  {c}: {n} gambar")


if __name__ == "__main__":
    main()
