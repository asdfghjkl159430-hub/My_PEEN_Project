import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import time

# 导入你自己的 PEEN 模型
from model import PEEN

# ===========================
# 1. 全局配置与常量定义
# ===========================
st.set_page_config(
    page_title="遥感分割算法对比平台",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 6
# 请确保路径正确
PEEN_WEIGHT = 'checkpoints/peen_best.pth'
UNET_WEIGHT = 'checkpoints/unet_best.pth'

# ISPRS Vaihingen 标准颜色映射
COLOR_MAP = [
    [255, 255, 255],  # 0: 不透水面 (White)
    [0, 0, 255],  # 1: 建筑物 (Blue)
    [0, 255, 255],  # 2: 低矮植被 (Cyan)
    [0, 255, 0],  # 3: 树木 (Green)
    [255, 255, 0],  # 4: 车辆 (Yellow)
    [255, 0, 0]  # 5: 杂物 (Red)
]

# 预处理转换
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ===========================
# 2. 工具函数定义
# ===========================
def label_to_rgb(mask):
    """将 0-5 的标签图转换为 RGB 彩色图"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(COLOR_MAP):
        rgb[mask == i] = color
    return rgb


@st.cache_resource
def load_models():
    """加载所有模型并缓存，避免重复加载"""
    models = {}
    # 1. 加载 PEEN (Ours)
    try:
        peen = PEEN(num_classes=NUM_CLASSES).to(DEVICE)
        peen.load_state_dict(torch.load(PEEN_WEIGHT, map_location=DEVICE))
        peen.eval()
        models['✨ PEEN (本文算法)'] = peen
    except FileNotFoundError:
        st.toast(f"⚠️ 未找到 PEEN 权重: {PEEN_WEIGHT}", icon="⚠️")

    # 2. 加载 UNet (Baseline)
    try:
        unet = smp.Unet(encoder_name="resnet34", classes=NUM_CLASSES).to(DEVICE)
        unet.load_state_dict(torch.load(UNET_WEIGHT, map_location=DEVICE))
        unet.eval()
        models['🐢 UNet (基线对比)'] = unet
    except FileNotFoundError:
        pass

    return models


def run_inference(model, input_tensor):
    """执行单次推理"""
    with torch.no_grad():
        if isinstance(model, PEEN):
            seg_out, _ = model(input_tensor)
        else:
            seg_out = model(input_tensor)

        pred = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy()
        pred_rgb = label_to_rgb(pred)
        return pred_rgb


# ===========================
# 3. 界面布局逻辑
# ===========================

# --- 侧边栏 ---
with st.sidebar:
    st.header("🎛️ 控制面板")
    st.write("请上传一张 ISPRS Vaihingen 数据集的测试图片（IR-R-G 波段）。")
    uploaded_file = st.file_uploader("📁 点击上传图片", type=["png", "jpg", "tif"])

    st.divider()
    st.markdown("### 💡 关于项目")
    st.info(
        "本项目旨在对比改进算法 (PEEN) 与传统基线算法 (UNet) "
        "在遥感高分影像语义分割任务上的表现差异。\n\n"
        f"当前运行设备: **{DEVICE.upper()}**"
    )

# --- 主界面标题 ---
st.markdown("<h1 style='text-align: center;'>🛰️ 遥感图像语义分割算法对比平台</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>毕业设计成果展示 | 概率引导与边缘强化网络 (PEEN)</p>",
            unsafe_allow_html=True)
st.divider()

# 加载模型
with st.spinner("正在初始化 AI 模型库，请稍候..."):
    models_dict = load_models()

if uploaded_file is not None:
    # --- 数据准备 ---
    image_pil = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform_fn(image_pil).unsqueeze(0).to(DEVICE)

    # ===========================
    # 布局区域 1: 原始影像与图例
    # ===========================
    st.subheader("1️⃣ 原始数据与图例")
    col_input, col_legend = st.columns([3, 2], gap="medium")

    with col_input:
        with st.container(border=True):
            # === 修改点 1: 替换为 use_container_width=True ===
            st.image(image_pil, caption="原始影像 (近红外-红-绿 合成)", use_container_width=True)

    with col_legend:
        with st.container(border=True):
            st.markdown("#### 🎨 类别图例说明")
            st.markdown("不同的颜色代表地表不同的物体类别：")
            st.markdown("""
            | 颜色示例 | 类别名称 | 典型地物 |
            | :---: | :--- | :--- |
            | ⚪ 白色 | **不透水面** | 道路、水泥地、停车场 |
            | 🔵 蓝色 | **建筑物** | 房屋屋顶 |
            | 🧪 青色 | **低矮植被** | 草坪、灌木丛 |
            | 🟢 绿色 | **树木** | 高大的乔木 |
            | 🟡 黄色 | **车辆** | 汽车、卡车 |
            | 🔴 红色 | **杂物** | 背景、集装箱、水体等 |
            """)

    st.divider()

    # ===========================
    # 布局区域 2: 算法结果并排对比
    # ===========================
    st.subheader("2️⃣ 算法分割结果对比")

    if not models_dict:
        st.error("❌ 未加载到任何有效模型，请检查权重文件路径！")
    else:
        model_names = list(models_dict.keys())
        num_models = len(model_names)
        cols = st.columns(num_models, gap="large")

        for idx, name in enumerate(model_names):
            model = models_dict[name]
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"### {name}")

                    with st.spinner(f"正在进行推理..."):
                        pred_rgb = run_inference(model, input_tensor)

                    # === 修改点 2: 替换为 use_container_width=True ===
                    st.image(pred_rgb, caption=f"{name} 预测结果", use_container_width=True)

                    if 'PEEN' in name:
                        st.success("✅ 优势：边缘细节更平滑，小物体（如车辆）识别更准确。")
                    elif 'UNet' in name:
                        st.warning("⚠️ 不足：边缘存在锯齿状噪声，易丢失小目标。")

else:
    st.markdown(
        """
        <div style='text-align: center; padding: 50px; color: gray;'>
            <h2>👈 请在左侧侧边栏上传图片以开始测试</h2>
            <p>支持 PNG, JPG, TIF 格式的遥感影像</p>
        </div>
        """,
        unsafe_allow_html=True
    )