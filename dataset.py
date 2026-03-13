import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms


class VaihingenDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.split = split
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        self.edge_dir = os.path.join(root_dir, split, 'edges')

        self.files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        print(f"[{split.upper()}] 加载了 {len(self.files)} 张图片")

        # 仅定义标准化，因为几何变换我们用 numpy 手写
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]

        img_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)
        edge_path = os.path.join(self.edge_dir, file_name)

        # 读取数据
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)

        # === 核心修改：数据增强 (仅对训练集) ===
        if self.split == 'train':
            # 1. 随机水平翻转
            if random.random() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
                edge = np.fliplr(edge)

            # 2. 随机垂直翻转
            if random.random() > 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)
                edge = np.flipud(edge)

            # 3. 随机 90度 旋转 (0, 90, 180, 270)
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
                edge = np.rot90(edge, k)

            # === 重要：解决 Numpy 负步长问题 ===
            # 翻转操作可能会产生负步长，PyTorch 不支持，必须 copy 一次
            image = image.copy()
            mask = mask.copy()
            edge = edge.copy()

        # === 转换为 Tensor ===
        image = self.transform(image)
        mask = torch.from_numpy(mask).long()
        edge = torch.from_numpy(edge).float() / 255.0
        edge = edge.unsqueeze(0)

        return image, mask, edge