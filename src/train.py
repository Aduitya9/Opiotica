"""One training loop, used by every label source without exception.

The whole study rests on holding everything constant except the label, so the schedule lives
here rather than in a notebook where it could drift between runs. Changing anything in this
file after the gate has been run is an amendment to the pre-registration, not a tweak.

Needs `torch` and `segmentation_models_pytorch`; both are installed in the Colab notebooks.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np


# --------------------------------------------------------------------------- fixed schedule
@dataclass(frozen=True)
class Config:
    encoder: str = "efficientnet-b4"
    encoder_weights: str = "imagenet"
    crop_size: int = 512
    batch_size: int = 8
    epochs: int = 60
    patience: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    num_workers: int = 2


CONFIG = Config()


def set_seed(seed: int) -> None:
    """Everything a run can be seeded with. The seed is the only thing that varies per repeat."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- data
class CropDataset:
    """Frozen crops on disk, one .npz per eye. `label_source` picks which masks are the target.

    The `image` array inside the .npz is the SAME bytes for every label source - that is the
    whole point of the frozen box, and it is what makes the convention contrast exact.
    """

    def __init__(self, manifest, label_source: str, root: str, size: int = CONFIG.crop_size,
                 augment: bool = False):
        self.rows = manifest.reset_index(drop=True)
        self.label_source = label_source
        self.root = root
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import torch

        row = self.rows.iloc[i]
        with np.load(os.path.join(self.root, row["npz"])) as rec:
            image = rec["image"].astype(np.float32) / 255.0
            cup = rec[f"cup_{self.label_source}"].astype(bool)
            disc = rec[f"disc_{self.label_source}"].astype(bool)
        if self.augment:
            if random.random() < 0.5:  # horizontal flip; the vCDR is a vertical measure
                image, cup, disc = image[:, ::-1], cup[:, ::-1], disc[:, ::-1]
            if random.random() < 0.5:  # mild photometric jitter, geometry untouched
                image = np.clip(image * random.uniform(0.85, 1.15), 0.0, 1.0)
        image = torch.from_numpy(np.ascontiguousarray(image).transpose(2, 0, 1)).float()
        target = torch.from_numpy(
            np.ascontiguousarray(np.stack([cup, disc])).astype(np.float32)
        )
        return image, target


# --------------------------------------------------------------------------- model and losses
def build_model(classes: int = 2, config: Config = CONFIG):
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=classes,
    )


def dice_bce(logits, target, eps: float = 1.0):
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    dice = 1.0 - ((2 * inter + eps) / (probs.sum(dims) + target.sum(dims) + eps)).mean()
    return bce + dice


# --------------------------------------------------------------------------- the loop
def train_one(train_set, val_set, seed: int, out_dir: str, config: Config = CONFIG,
              device: str = "cuda", log_every: int = 1):
    """Train one model. Identical for every label source; only `seed` and the target differ."""
    import torch
    from torch.utils.data import DataLoader

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    model = build_model(config=config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)

    loaders = {
        "train": DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers, drop_last=True, pin_memory=True),
        "val": DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                          num_workers=config.num_workers, pin_memory=True),
    }

    best, best_epoch, history = float("inf"), -1, []
    weights_path = os.path.join(out_dir, "best.pt")
    for epoch in range(config.epochs):
        row = {"epoch": epoch}
        for phase in ("train", "val"):
            model.train(phase == "train")
            total, count = 0.0, 0
            with torch.set_grad_enabled(phase == "train"):
                for images, targets in loaders[phase]:
                    images, targets = images.to(device), targets.to(device)
                    with torch.amp.autocast("cuda", enabled=config.amp):
                        loss = dice_bce(model(images), targets)
                    if phase == "train":
                        optimiser.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(optimiser)
                        scaler.update()
                    total += loss.item() * images.size(0)
                    count += images.size(0)
            row[f"{phase}_loss"] = total / max(count, 1)
        scheduler.step()
        history.append(row)
        if row["val_loss"] < best:
            best, best_epoch = row["val_loss"], epoch
            torch.save(model.state_dict(), weights_path)
        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:3d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}"
                  f"{'  *' if best_epoch == epoch else ''}")
        if epoch - best_epoch >= config.patience:
            print(f"  early stop at epoch {epoch} (best {best_epoch}, val {best:.4f})")
            break

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": asdict(config), "seed": seed, "best_epoch": best_epoch,
                   "best_val_loss": best, "history": history}, fh, indent=2)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model, {"seed": seed, "best_epoch": best_epoch, "best_val_loss": best}


# --------------------------------------------------------------------------- inference
def predict_vcdr(model, dataset, device: str = "cuda", batch_size: int = 8,
                 threshold: float = 0.5):
    """Predicted vCDR per eye. This single number is what the whole paper compares."""
    import torch
    from torch.utils.data import DataLoader

    from . import vcdr as vcdr_mod

    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=CONFIG.num_workers)
    out = []
    with torch.no_grad():
        for images, _ in loader:
            probs = torch.sigmoid(model(images.to(device))).float().cpu().numpy()
            for p in probs:
                out.append(vcdr_mod.vcdr(p[0] > threshold, p[1] > threshold))
    return np.asarray(out, dtype=float)
