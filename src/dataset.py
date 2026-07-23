"""Data loading and image transforms for the classification model."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def build_transforms(img_size: int = config.IMG_SIZE):
    """Return (train_tf, eval_tf).

    Training uses light augmentation (flip / rotation / colour jitter) to reduce
    over-fitting; evaluation just resizes and normalises. Normalisation uses the
    ImageNet statistics expected by the pretrained backbone.
    """
    from torchvision import transforms

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_tf, eval_tf


def build_dataloaders(img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
                      num_workers=config.NUM_WORKERS):
    """Create train/val/test dataloaders from data/classification/<split>/<class>/.

    Returns (loaders_dict, class_names).
    """
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    train_tf, eval_tf = build_transforms(img_size)
    root = config.CLASSIFICATION_DIR

    splits = {}
    for split, tf in (("train", train_tf), ("val", eval_tf), ("test", eval_tf)):
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing '{split_dir}'. Run download_data.py then "
                f"prepare_classification.py first."
            )
        splits[split] = ImageFolder(str(split_dir), transform=tf)

    class_names = splits["train"].classes

    loaders = {
        "train": DataLoader(splits["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        "val": DataLoader(splits["val"], batch_size=batch_size, shuffle=False,
                          num_workers=num_workers),
        "test": DataLoader(splits["test"], batch_size=batch_size, shuffle=False,
                           num_workers=num_workers),
    }
    return loaders, class_names
