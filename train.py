import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 导入你的模块
from dataset import VaihingenDataset
from model import PEEN

# === 全局参数 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LEARNING_RATE = 2e-4  # 保持学习率
EPOCHS = 200
NUM_CLASSES = 6

# === 修改点 1: 定义类别权重 (Class Weights) ===
# 针对 Car (index 4) 和 Low Veg (index 2) 进行加权
# [Imp_Surf, Building, Low_Veg, Tree, Car, Clutter]
# Car 设为 3.0，Low Veg 设为 1.5，其他 1.0
CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 1.5, 1.0, 3.0, 1.0]).to(DEVICE)


# === Dice Loss (用于边缘) ===
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)


# === 训练函数 ===
def train_one_epoch(model, loader, optimizer, seg_criterion, edge_criterion, epoch_idx):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch_idx}/{EPOCHS}")

    for images, masks, edges in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        edges = edges.to(DEVICE)

        seg_out, edge_outs = model(images)

        # Loss 计算 (应用类别权重)
        loss_seg = seg_criterion(seg_out, masks)

        loss_edge_total = 0
        for edge_pred in edge_outs:
            loss_edge_total += edge_criterion(edge_pred, edges)

        # === 修改点 2: 提高边缘 Loss 的权重 ===
        # 从 0.5 改为 1.0，让模型更重视边缘细节
        total_loss = loss_seg + 1.0 * loss_edge_total

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        loop.set_postfix(loss=total_loss.item())

    return running_loss / len(loader)


def main():
    root_dir = 'processed_data'
    train_dataset = VaihingenDataset(root_dir, split='train')
    # Windows 必须 num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"设备: {DEVICE} | 训练集: {len(train_dataset)} 张")
    print(f"类别权重已启用: {CLASS_WEIGHTS.cpu().numpy()}")

    model = PEEN(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # === 将权重传入 CrossEntropy ===
    seg_criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    edge_criterion = DiceLoss()

    os.makedirs('checkpoints', exist_ok=True)
    best_loss = float('inf')

    # 如果你想在旧权重基础上继续训练，取消下面两行的注释
    # if os.path.exists("checkpoints/peen_best.pth"):
    #     model.load_state_dict(torch.load("checkpoints/peen_best.pth"))

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, seg_criterion, edge_criterion, epoch)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # 保存 Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/peen_best.pth")
            print(">>> 发现新低 Loss，已保存 peen_best.pth")

        # 兜底保存
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"checkpoints/peen_epoch_{epoch}.pth")

    print(f"训练结束！最佳模型 Loss 为: {best_loss:.4f}")


if __name__ == '__main__':
    main()