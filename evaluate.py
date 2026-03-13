import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 导入你的模型
from model import PEEN

# === 核心配置 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
CROP_SIZE = 256
CHECKPOINT_PATH = 'checkpoints/peen_best.pth'  # 确保使用最佳权重

# === 提分关键点 1: 更密集的步长 ===
# 建议设为 CROP_SIZE // 3 (约85) 或 // 4 (64)
# 越小越准，但速度越慢。85 是一个很好的平衡点。
STRIDE = 85

# === 提分关键点 2: 多尺度推理 (Multi-Scale Testing) ===
# 论文中为了刷 SOTA 通常会用 [0.5, 0.75, 1.0, 1.25, 1.5]
# 如果显存不够，可以减少为 [0.75, 1.0, 1.25]
SCALES = [0.75, 1.0, 1.25]

# === 提分关键点 3: 翻转集成 (Flip Testing) ===
ENABLE_FLIP = True

RAW_IMAGE_DIR = 'raw_data/top'
RAW_LABEL_DIR = 'raw_data/gts'
TEST_IDS = ['1', '4', '7', '10', '13', '16', '17', '22', '26', '28', '31', '33']

COLOR_MAP = {
    (255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2,
    (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5
}
CLASS_NAMES = ['Imp. Surf.', 'Building', 'Low Veg.', 'Tree', 'Car', 'Clutter']


def rgb_to_mask(img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for rgb, idx in COLOR_MAP.items():
        condition = (img[:, :, 0] == rgb[0]) & (img[:, :, 1] == rgb[1]) & (img[:, :, 2] == rgb[2])
        mask[condition] = idx
    return mask


class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        return count.reshape(self.num_classes, self.num_classes)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def evaluate(self):
        IoUs = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix) + 1e-6
        )
        mIoU_paper = np.nanmean(IoUs[:5])  # 只算前5类
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-6)
        return IoUs, OA, mIoU_paper


def predict_patch(model, patch_tensor):
    """单张 Patch 的预测，包含 Flip TTA"""
    # 1. 原图预测
    seg_out, _ = model(patch_tensor)
    prob = F.softmax(seg_out, dim=1)  # [1, C, H, W]

    if ENABLE_FLIP:
        # 2. 水平翻转
        patch_flip = torch.flip(patch_tensor, [3])
        seg_out_flip, _ = model(patch_flip)
        prob_flip = F.softmax(seg_out_flip, dim=1)
        prob += torch.flip(prob_flip, [3])

        # 3. 垂直翻转
        patch_vflip = torch.flip(patch_tensor, [2])
        seg_out_vflip, _ = model(patch_vflip)
        prob_vflip = F.softmax(seg_out_vflip, dim=1)
        prob += torch.flip(prob_vflip, [2])

        prob /= 3.0  # 取平均

    return prob


def predict_whole_image(model, image, num_classes=6, crop_size=256, stride=128):
    """支持多尺度的全图推理"""
    h_orig, w_orig, _ = image.shape

    # 初始化最终概率图 (累加所有尺度的结果)
    final_probs = torch.zeros((num_classes, h_orig, w_orig), device=DEVICE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model.eval()

    for scale in SCALES:
        # 1. Resize 图片
        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 初始化当前尺度的概率图和计数图
        scale_probs = torch.zeros((num_classes, new_h, new_w), device=DEVICE)
        count_map = torch.zeros((1, new_h, new_w), device=DEVICE)

        # 2. 滑动窗口
        rows = [r for r in range(0, new_h - crop_size + 1, stride)]
        if (new_h - crop_size) % stride != 0: rows.append(new_h - crop_size)
        cols = [c for c in range(0, new_w - crop_size + 1, stride)]
        if (new_w - crop_size) % stride != 0: cols.append(new_w - crop_size)

        with torch.no_grad():
            for y in rows:
                for x in cols:
                    patch = image_scaled[y:y + crop_size, x:x + crop_size]
                    patch_pil = Image.fromarray(patch)
                    patch_tensor = transform(patch_pil).unsqueeze(0).to(DEVICE)

                    # 预测 (含 Flip TTA)
                    prob = predict_patch(model, patch_tensor).squeeze(0)

                    scale_probs[:, y:y + crop_size, x:x + crop_size] += prob
                    count_map[:, y:y + crop_size, x:x + crop_size] += 1.0

        # 平均当前尺度结果
        scale_probs /= count_map

        # 3. Resize 回原图大小 (双线性插值)
        # interpolate 需要 [B, C, H, W]
        scale_probs = F.interpolate(scale_probs.unsqueeze(0),
                                    size=(h_orig, w_orig),
                                    mode='bilinear',
                                    align_corners=False).squeeze(0)

        final_probs += scale_probs  # 累加到总结果

    # 取所有尺度的 argmax
    final_pred = torch.argmax(final_probs, dim=0).cpu().numpy().astype(np.uint8)
    return final_pred


def main():
    print(f"启动终极评估: Stride={STRIDE}, Scales={SCALES}, Flip={ENABLE_FLIP}")

    model = PEEN(num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print(f"错误：找不到权重 {CHECKPOINT_PATH}")
        return

    evaluator = Evaluator(NUM_CLASSES)

    # 自动匹配文件逻辑优化
    files = os.listdir(RAW_IMAGE_DIR)

    for area_id in tqdm(TEST_IDS, desc="Evaluating"):
        # 模糊匹配文件名
        img_filename = next((f for f in files if f"area{area_id}." in f or f"area{area_id}_" in f), None)

        if img_filename is None:
            continue  # 找不到就算了

        img_path = os.path.join(RAW_IMAGE_DIR, img_filename)
        # 尝试匹配标签
        gt_filename = img_filename  # 假设同名
        if not os.path.exists(os.path.join(RAW_LABEL_DIR, gt_filename)):
            gt_filename = img_filename.replace(".tif", "_class.tif")  # 尝试 _class 后缀

        gt_path = os.path.join(RAW_LABEL_DIR, gt_filename)

        if not os.path.exists(gt_path):
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_img = np.array(Image.open(gt_path))
        gt_mask = rgb_to_mask(gt_img)

        if image.shape[:2] != gt_mask.shape:
            image = cv2.resize(image, (gt_mask.shape[1], gt_mask.shape[0]))

        # === 终极预测 ===
        pred_mask = predict_whole_image(model, image,
                                        num_classes=NUM_CLASSES,
                                        crop_size=CROP_SIZE,
                                        stride=STRIDE)

        evaluator.add_batch(gt_mask, pred_mask)

    IoUs, OA, mIoU = evaluator.evaluate()

    print("\n" + "=" * 45)
    print(f"{'Class':<15} | {'IoU (Final)':<15}")
    print("-" * 45)
    for i in range(5):
        print(f"{CLASS_NAMES[i]:<15} | {IoUs[i] * 100:.2f}%")
    print("-" * 45)
    print(f"Overall Accuracy (OA): {OA * 100:.2f}%")
    print(f"Paper mIoU (Top-5):    {mIoU * 100:.2f}%")
    print("=" * 45)


if __name__ == '__main__':
    main()