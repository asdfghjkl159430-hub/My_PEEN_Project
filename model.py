import torch
import torch.nn as nn
import torch.nn.functional as F


# === 1. 卷积自注意力模块 (Conv SA) ===
class ConvSA(nn.Module):
    def __init__(self, channels, kernel_size=15):
        super(ConvSA, self).__init__()
        # 严格按照论文 Eq 1-3 和 Fig 4(b)：首先进行 BN 操作
        self.bn = nn.BatchNorm2d(channels)

        # A 分支 (注意力权重): A = Sigmoid(DConv(W1 * X))
        self.W1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.DConv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

        # V 分支 (Value): V = W2 * X
        self.W2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_bn = self.bn(x)
        A = self.sigmoid(self.DConv(self.W1(x_bn)))
        V = self.W2(x_bn)
        return A * V


# === 2. 编码器块 (Encoder Block) ===
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        # 如果输入输出通道不一致，先用 1x1 卷积对齐通道数
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              bias=False) if in_channels != out_channels else nn.Identity()

        # 论文中每个阶段包含 ConvSA 模块进行特征提取
        self.conv_sa = ConvSA(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.relu(self.bn(self.conv_sa(x)))
        return x


# === 3. 解码器块 (Decoder Block) ===
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 2倍上采样
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # 将跳跃连接的特征通道对齐
        self.skip_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False)

        # 【重要提精修改】拼接后使用连续两个 3x3 卷积，大幅增强特征融合能力
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x_up = self.up(x)
        skip_proj = self.skip_conv(skip)

        # 拼接 (Concatenation)
        out = torch.cat([x_up, skip_proj], dim=1)
        return self.conv(out)


# === 4. 非对称卷积块 (用于 IPEP) ===
class AsymConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AsymConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# === 5. IPEP 模块 (迭代边缘预测) ===
class IPEP(nn.Module):
    def __init__(self, in_channels, iterations=5):
        super(IPEP, self).__init__()
        self.iterations = iterations
        self.layers = nn.ModuleList()
        self.predictors = nn.ModuleList()

        for i in range(iterations):
            input_dim = in_channels if i == 0 else in_channels * 2
            self.layers.append(AsymConvBlock(input_dim, in_channels))
            self.predictors.append(nn.Conv2d(in_channels, 1, kernel_size=1))

    def forward(self, f0):
        edge_preds = []
        f_prev = None

        for i in range(self.iterations):
            if i == 0:
                x = f0
            else:
                x = torch.cat([f0, f_prev], dim=1)

            f_curr = self.layers[i](x)
            p_curr = torch.sigmoid(self.predictors[i](f_curr))
            edge_preds.append(p_curr)
            f_prev = f_curr

        return edge_preds


# === 6. 完整的 PEEN 模型 ===
class PEEN(nn.Module):
    def __init__(self, num_classes=6):
        super(PEEN, self).__init__()

        # --- Encoder (编码器) ---
        self.enc1 = EncoderBlock(3, 64)  # C1 = 64
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = EncoderBlock(64, 256)  # C2 = 256
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = EncoderBlock(256, 512)  # C3 = 512
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = EncoderBlock(512, 1024)  # C4 = 1024
        self.pool4 = nn.MaxPool2d(2, 2)

        # 【重要提精修改】增加 Bottleneck (瓶颈层)
        self.bot = EncoderBlock(1024, 1024)

        # --- Decoder (解码器) ---
        self.dec4 = DecoderBlock(1024, 1024, 256)  # C5 = 256
        self.dec3 = DecoderBlock(256, 512, 128)  # C6 = 128
        self.dec2 = DecoderBlock(128, 256, 64)  # C7 = 64
        self.dec1 = DecoderBlock(64, 64, 16)  # C8 = 16

        # --- Heads ---
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.ipep = IPEP(in_channels=16, iterations=5)

    def forward(self, x):
        # --- Encoder ---
        s1 = self.enc1(x)  # [B, 64, 256, 256]
        p1 = self.pool1(s1)  # [B, 64, 128, 128]

        s2 = self.enc2(p1)  # [B, 256, 128, 128]
        p2 = self.pool2(s2)  # [B, 256, 64, 64]

        s3 = self.enc3(p2)  # [B, 512, 64, 64]
        p3 = self.pool3(s3)  # [B, 512, 32, 32]

        s4 = self.enc4(p3)  # [B, 1024, 32, 32]
        p4 = self.pool4(s4)  # [B, 1024, 16, 16]

        # Bottleneck
        bot = self.bot(p4)  # [B, 1024, 16, 16]

        # --- Decoder ---
        d4 = self.dec4(bot, s4)  # [B, 256, 32, 32]
        d3 = self.dec3(d4, s3)  # [B, 128, 64, 64]
        d2 = self.dec2(d3, s2)  # [B, 64, 128, 128]
        f0 = self.dec1(d2, s1)  # [B, 16, 256, 256]

        # --- Heads ---
        seg_out = self.seg_head(f0)
        edge_outs = self.ipep(f0)

        return seg_out, edge_outs


if __name__ == '__main__':
    inputs = torch.randn(2, 3, 256, 256)
    model = PEEN(num_classes=6)
    try:
        seg, edges = model(inputs)
        print(f"输入尺寸: {inputs.shape}")
        print(f"语义分割输出: {seg.shape}")
        print(f"边缘预测迭代次数: {len(edges)}")
        print(f"最后一次边缘预测: {edges[-1].shape}")
        print(">>> 模型测试通过！新架构已准备就绪。")
    except Exception as e:
        print(f"测试失败: {e}")