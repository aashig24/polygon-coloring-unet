
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import wandb
from dataset import PolygonColorDataset
from model import UNet

def train():
    wandb.init(project="polygon-coloring-unet")

    train_dataset = PolygonColorDataset("dataset", split="training")
    val_dataset = PolygonColorDataset("dataset", split="validation")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = UNet().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(75):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        wandb.log({"epoch": epoch, "train_loss": total_loss / len(train_loader)})

        # Simple validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                preds = model(x)
                val_loss += criterion(preds, y).item()
        wandb.log({"val_loss": val_loss / len(val_loader)})

    torch.save(model.state_dict(), "unet_polygon.pth")

if __name__ == "__main__":
    train()

