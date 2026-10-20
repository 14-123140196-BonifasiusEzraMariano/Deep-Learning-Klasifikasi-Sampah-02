"""Evaluasi model klasifikasi sampah.

Pemakaian:
    python evaluate.py --submission ./submission --data ./my_val

Struktur --data (format ImageFolder, satu folder per kelas):
    my_val/
      kelas_a/ img1.jpg ...
      kelas_b/ ...
"""
import argparse
import importlib.util
import os

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets


def load_submission(sub_dir: str, device):
    model_py = os.path.join(sub_dir, "model.py")
    model_pth = os.path.join(sub_dir, "model.pth")
    for p in (model_py, model_pth):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File tidak ditemukan: {p}")

    spec = importlib.util.spec_from_file_location("submission_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.get_model()
    state = torch.load(model_pth, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, mod.get_transform()


@torch.no_grad()
def predict(model, loader, device):
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    return y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="./submission")
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf = load_submission(args.submission, device)

    ds = datasets.ImageFolder(args.data, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    y_true, y_pred = predict(model, loader, device)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    print(f"Device    : {device}")
    print(f"Jumlah    : {len(ds)} gambar, {len(ds.classes)} kelas -> {ds.classes}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro-F1  : {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=ds.classes, digits=4, zero_division=0))
    print("Confusion matrix (baris = asli, kolom = prediksi):")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
