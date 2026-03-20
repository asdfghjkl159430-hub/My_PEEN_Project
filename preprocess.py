import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# === 1. 配置路径 (请确保这里指向正确的文件夹) ===
RAW_IMAGE_DIR = 'raw_data/top'  # 存放原始影像
RAW_LABEL_DIR = 'raw_data/gts'  # 存放原始标签 (文件名与影像完全一致)
OUTPUT_DIR = 'processed_data'
CROP_SIZE = 256
STRIDE = 128  # 训练集重叠切割(步长128)，测试集不重叠(步长256)

# === 2. 论文规定的测试集 ID ===
# 根据文件名中的数字区分
TEST_IDS = ['1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '33']

# === 3. 颜色映射 (RGB -> 类别 ID) ===
COLOR_MAP = {
    (255, 0, 0): 0,      # Clutter/Background (红) -> 严格设为 0
    (255, 255, 255): 1,  # Impervious surfaces (白)
    (0, 0, 255): 2,      # Building (蓝)
    (0, 255, 255): 3,    # Low vegetation (青)
    (0, 255, 0): 4,      # Tree (绿)
    (255, 255, 0): 5     # Car (黄)
}


def rgb_to_mask(img):
    """将RGB标签转换为单通道类别索引 (0-5)"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for rgb, idx in COLOR_MAP.items():
        # 找到图像中符合当前颜色的像素点
        condition = (img[:, :, 0] == rgb[0]) & (img[:, :, 1] == rgb[1]) & (img[:, :, 2] == rgb[2])
        mask[condition] = idx
    return mask


def generate_edge(mask):
    """使用论文指定的 Canny 算子生成单像素精度的细边缘"""
    # 将类别 mask (0-5) 拉伸到 0-255，确保不同类别的交界处有明显像素差
    mask_255 = (mask * 50).astype(np.uint8)
    # 使用 Canny 算子提取边缘 (阈值设为 10, 100 即可检出所有类别跳变)
    edge = cv2.Canny(mask_255, 10, 100)
    return edge


def crop_and_save(img_path, label_path, mode='train'):
    # 读取图像
    image = cv2.imread(img_path)  # BGR格式
    # 读取标签 (使用PIL读取以保证RGB通道顺序正确)
    label_img = Image.open(label_path)
    label_rgb = np.array(label_img)

    # 检查读取是否成功
    if image is None:
        print(f"无法读取影像: {img_path}")
        return

    # 确保大小一致 (ISPRS数据中有时会出现1-2像素的误差)
    if image.shape[:2] != label_rgb.shape[:2]:
        # print(f"校正尺寸: {os.path.basename(img_path)}")
        image = cv2.resize(image, (label_rgb.shape[1], label_rgb.shape[0]))

    # 转为 Mask (0-5)
    label_mask = rgb_to_mask(label_rgb)

    # 生成 Edge (0, 255)
    label_edge = generate_edge(label_mask)

    h, w, _ = image.shape

    # 设置步长：测试集不重叠(256)，训练集重叠(128)以增广数据
    step = CROP_SIZE if mode == 'test' else STRIDE

    # 获取文件名 ID 部分用于保存
    base_name = os.path.basename(img_path).split('.')[0]

    # 滑动窗口切图
    for i in range(0, h - CROP_SIZE + 1, step):
        for j in range(0, w - CROP_SIZE + 1, step):
            img_crop = image[i:i + CROP_SIZE, j:j + CROP_SIZE]
            mask_crop = label_mask[i:i + CROP_SIZE, j:j + CROP_SIZE]
            edge_crop = label_edge[i:i + CROP_SIZE, j:j + CROP_SIZE]

            # 保存文件名: area1_0_0.png
            save_name = f"{base_name}_{i}_{j}.png"

            # 保存路径
            save_root = os.path.join(OUTPUT_DIR, mode)
            cv2.imwrite(os.path.join(save_root, 'images', save_name), img_crop)
            cv2.imwrite(os.path.join(save_root, 'masks', save_name), mask_crop)
            cv2.imwrite(os.path.join(save_root, 'edges', save_name), edge_crop)


def main():
    # 创建文件夹结构
    for mode in ['train', 'test']:
        for sub in ['images', 'masks', 'edges']:
            os.makedirs(os.path.join(OUTPUT_DIR, mode, sub), exist_ok=True)

    # 获取 raw_data/top 下的所有文件
    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"错误: 找不到文件夹 {RAW_IMAGE_DIR}，请检查路径。")
        return

    files = sorted(os.listdir(RAW_IMAGE_DIR))
    print(f"在 {RAW_IMAGE_DIR} 中找到 {len(files)} 个文件。开始处理...")

    for f in tqdm(files):
        # 1. 修复语法错误：增加空格
        if not f.endswith('.tif'):
            continue

        # 2. 解析 Area ID (文件名格式: top_mosaic_09cm_area1.tif)
        try:
            # 以 'area' 分割，取最后一部分 '1.tif'，再以 '.' 分割取 '1'
            area_id = f.split('area')[-1].split('.')[0]
        except:
            print(f"文件名解析ID失败: {f}，跳过")
            continue

        # 3. 构建路径 (既然文件名完全一致，直接用 f)
        img_path = os.path.join(RAW_IMAGE_DIR, f)
        label_path = os.path.join(RAW_LABEL_DIR, f)

        # 4. 检查标签文件是否存在
        if not os.path.exists(label_path):
            print(f"警告：在 {RAW_LABEL_DIR} 中找不到对应的标签文件: {f}，跳过此图。")
            continue

        # 5. 划分数据集并处理
        if area_id in TEST_IDS:
            crop_and_save(img_path, label_path, mode='test')
        else:
            crop_and_save(img_path, label_path, mode='train')

    print("处理完成！请检查 processed_data 文件夹。")


if __name__ == '__main__':
    main()