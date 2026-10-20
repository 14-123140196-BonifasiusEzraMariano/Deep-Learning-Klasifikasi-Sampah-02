"""Definisi model klasifikasi sampah (ResNet-50, 5 kelas).

File ini di-import oleh evaluate.py:
    get_model()      -> nn.Module (arsitektur; bobot di-load dari model.pth)
    get_transform()  -> transform untuk preprocessing gambar
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_model(pretrained: bool = False) -> nn.Module:
    """Bangun ResNet-50 dengan head 5 kelas.

    pretrained=True  -> pakai bobot ImageNet (dipakai saat training).
    pretrained=False -> arsitektur saja (dipakai saat evaluasi, bobot dari model.pth).
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    return model


def get_transform(train: bool = False):
    """Transform gambar. train=True memakai augmentasi, selain itu deterministik."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
