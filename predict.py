"""Inferensi model klasifikasi sampah.

Contoh pemakaian:
    # Satu gambar
    python predict.py --image foto_sampah.jpg

    # Semua gambar dalam folder
    python predict.py --folder ./foto_baru

    # Tampilkan top-3 prediksi
    python predict.py --image foto_sampah.jpg --top 3
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission"))
try:
    from submission.model import get_model, get_transform  # noqa: E402
except ImportError:
    from model import get_model, get_transform  # noqa: E402

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(submission_dir: str, device):
    model_py  = os.path.join(submission_dir, "model.py")
    model_pth = os.path.join(submission_dir, "model.pth")
    for p in (model_py, model_pth):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File tidak ditemukan: {p}")

    model = get_model(pretrained=False)
    state = torch.load(model_pth, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_image(image_path: str, model, transform, device, top_k: int = 1):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(top_k)
    results = [
        (CLASSES[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]
    return results


def process_image(image_path: str, model, transform, device, top_k: int):
    results = predict_image(image_path, model, transform, device, top_k)
    label, conf = results[0]
    print(f"\n📷 {os.path.basename(image_path)}")
    print(f"   ➜  Prediksi : {label} ({conf * 100:.1f}%)")
    if top_k > 1:
        print("   Top prediksi:")
        for rank, (lbl, prob) in enumerate(results, 1):
            bar = "█" * int(prob * 20)
            print(f"   {rank}. {lbl:<20} {prob * 100:5.1f}%  {bar}")


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path ke satu gambar")
    group.add_argument("--folder", help="Path ke folder berisi gambar")
    ap.add_argument("--submission", default="./submission")
    ap.add_argument("--top", type=int, default=1, help="Tampilkan top-N prediksi")
    args = ap.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(args.submission, device)
    transform = get_transform(train=False)

    print(f"Device: {device} | Model: {args.submission}/model.pth")
    print(f"Kelas : {CLASSES}")

    if args.image:
        process_image(args.image, model, transform, device, args.top)

    elif args.folder:
        images = [
            os.path.join(args.folder, f)
            for f in sorted(os.listdir(args.folder))
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
        if not images:
            print(f"Tidak ada gambar di folder: {args.folder}")
            return
        print(f"\nMemproses {len(images)} gambar dari {args.folder}...")
        for img_path in images:
            process_image(img_path, model, transform, device, args.top)

    print()


if __name__ == "__main__":
    main()
