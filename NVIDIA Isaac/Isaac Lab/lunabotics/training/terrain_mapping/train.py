"""Training script for terrain mapping model.

Usage:
    python -m training.terrain_mapping.train \
        --data_dir /path/to/data \
        --output_dir /path/to/checkpoints \
        --epochs 100 --batch_size 32
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from dataset import TerrainDataset
from losses import TotalLoss
from model import TerrainMappingModel


def collate_fn(batch):
    bevs, heights, semantics, objects, occupancies = zip(*batch)
    return (
        torch.stack(bevs),
        torch.stack(heights),
        torch.stack(semantics),
        list(objects),
        torch.stack(occupancies),
    )


def train(data_dir: str, output_dir: str, epochs: int, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = TerrainDataset(data_dir)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4,
                            collate_fn=collate_fn, pin_memory=True)

    model = TerrainMappingModel().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = TotalLoss()
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bev, height_gt, semantic_gt, objects_gt, occupancy in train_loader:
            bev, height_gt, semantic_gt, occupancy = (
                bev.to(device), height_gt.to(device), semantic_gt.to(device), occupancy.to(device)
            )
            optimizer.zero_grad()
            loss = criterion(model(bev), (height_gt, semantic_gt, objects_gt, occupancy))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bev, height_gt, semantic_gt, objects_gt, occupancy in val_loader:
                bev, height_gt, semantic_gt, occupancy = (
                    bev.to(device), height_gt.to(device), semantic_gt.to(device), occupancy.to(device)
                )
                val_loss += criterion(model(bev), (height_gt, semantic_gt, objects_gt, occupancy)).item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out / "best_model.pt")
            print(f"  → saved best (val={best_val:.4f})")

    torch.save(model.state_dict(), out / "final_model.pt")
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="checkpoints/terrain_mapping")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    train(args.data_dir, args.output_dir, args.epochs, args.batch_size)
