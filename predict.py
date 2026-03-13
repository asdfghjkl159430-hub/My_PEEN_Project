import torch
import cv2
import os
import numpy as np
from model import PEEN
from torchvision import transforms
from PIL import Image

# === 配置 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
# 这里填训练好的权重路径，等会生成了 peen_epoch_10.pth 就可以填进去试
CHECKPOINT_PATH = 'checkpoints/peen_epoch_200.pth'
TEST_IMG_DIR = 'processed_data/test/images'  # 测试集图片路径
OUTPUT_DIR = 'results'  # 结果保存路径

# 颜色映射 (用于把 0-5 的预测结果变成彩色图，方便肉眼观察)
# 对应: 不透水面(白), 建筑(蓝), 低植被(青), 树(绿), 车(黄), 背景(红)
COLOR_MAP = [
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0]
]


def label_to_rgb(mask):
    """把单通道 Mask (0-5) 转成 RGB 彩色图"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(COLOR_MAP):
        rgb[mask == i] = color
    return rgb


# 修改 predict.py 中的 predict 函数，增加 GT 的读取和展示
def predict():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型
    print(f"正在加载模型: {CHECKPOINT_PATH} ...")
    model = PEEN(num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("错误：找不到权重文件！")
        return
    model.eval()

    # 2. 预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 读取测试集列表
    test_files = os.listdir(TEST_IMG_DIR)[11:20]  # 多测几张，比如前10张

    with torch.no_grad():
        for file_name in test_files:
            img_path = os.path.join(TEST_IMG_DIR, file_name)
            # 对应的 GT 路径 (假设在 masks 文件夹下)
            gt_path = os.path.join('processed_data/test/masks', file_name)

            # 读取原图和 GT
            original_img = cv2.imread(img_path)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            # 预处理输入
            img_pil = Image.open(img_path).convert('RGB')
            input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

            # 推理
            seg_out, edge_outs = model(input_tensor)

            # 处理预测结果
            pred_mask = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy()
            pred_edge = edge_outs[-1].squeeze().cpu().numpy()
            pred_edge = (pred_edge * 255).astype(np.uint8)

            # === 转换颜色 ===
            pred_rgb = label_to_rgb(pred_mask)  # 预测图转彩色
            gt_rgb = label_to_rgb(gt_mask)  # 真实标签转彩色 (新增!)
            pred_edge_bgr = cv2.cvtColor(pred_edge, cv2.COLOR_GRAY2BGR)  # 边缘转 BGR
            pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)  # 预测转 BGR
            gt_bgr = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)  # 真实标签转 BGR

            # === 拼接 4 张图: 原图 | GT | 预测 | 边缘 ===
            combined = np.hstack([original_img, gt_bgr, pred_bgr, pred_edge_bgr])

            save_path = os.path.join(OUTPUT_DIR, f"result_{file_name}")
            cv2.imwrite(save_path, combined)
            print(f"保存结果: {save_path}")


if __name__ == '__main__':
    predict()