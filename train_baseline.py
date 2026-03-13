print(">>> [1/5] 脚本启动...")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from dataset import VaihingenDataset

print(">>> [2/5] 库导入成功，正在配置参数...")

# === 全局参数配置 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
EPOCHS = 100
NUM_CLASSES = 6


def train_one_epoch(model, loader, optimizer, criterion, epoch_idx):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch_idx}/{EPOCHS} [Baseline]")

    for images, masks, _ in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)


def main():
    print(f">>> [3/5] 开始准备数据 (Device: {DEVICE})...")

    root_dir = 'processed_data'
    if not os.path.exists(root_dir):
        print(f"错误: 找不到数据文件夹 {root_dir}")
        return

    train_dataset = VaihingenDataset(root_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f">>> [4/5] 数据加载完成，训练集数量: {len(train_dataset)}")
    print(">>> [5/5] 初始化 UNet 模型...")

    # === 初始化模型 ===
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    os.makedirs('checkpoints', exist_ok=True)
    best_loss = float('inf')

    print(">>> 一切就绪，开始训练循环！")

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        scheduler.step()

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/unet_best.pth")
            print(f">>> 发现新低 Loss，已保存 unet_best.pth")

    print("对比实验结束！")


# === 必须顶格写，不能缩进 ===
if __name__ == '__main__':
    main()