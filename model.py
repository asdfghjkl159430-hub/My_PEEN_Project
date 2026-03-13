import torch
import torch.nn as nn
import torch.nn.functional as F


# === 1. 卷积自注意力模块 (Conv SA) ===
class ConvSA(nn.Module):
    def __init__(self, in_channels, kernel_size=15):
        super(ConvSA, self).__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=in_channels, bias=False),
            nn.Sigmoid()
        )
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        A = self.conv_a(x)
        V = self.conv_v(x)
        return A * V

    # === 2. 非对称卷积块 (用于 IPEP) ===


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


# === 3. IPEP 模块 (迭代边缘预测) ===
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


# === 4. 完整的 PEEN 模型 (修正版) ===
class PEEN(nn.Module):
    def __init__(self, num_classes=6):
        super(PEEN, self).__init__()

        # --- Encoder ---
        # Stage 1: Input (3) -> s1 (3) -> Down (64) -> Pool (64, H/2)
        self.enc1 = nn.Sequential(ConvSA(3), nn.BatchNorm2d(3), nn.ReLU())
        self.down1 = nn.Conv2d(3, 64, kernel_size=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Stage 2: Input (64) -> s2 (64) -> Down (256) -> Pool (256, H/4)
        self.enc2 = nn.Sequential(ConvSA(64), nn.BatchNorm2d(64), nn.ReLU())
        self.down2 = nn.Conv2d(64, 256, kernel_size=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Stage 3: Input (256) -> s3 (256) -> Down (512) -> Pool (512, H/8)
        self.enc3 = nn.Sequential(ConvSA(256), nn.BatchNorm2d(256), nn.ReLU())
        self.down3 = nn.Conv2d(256, 512, kernel_size=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Stage 4: Input (512) -> s4 (512) -> Down (1024) -> Pool (1024, H/16)
        self.enc4 = nn.Sequential(ConvSA(512), nn.BatchNorm2d(512), nn.ReLU())
        self.down4 = nn.Conv2d(512, 1024, kernel_size=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # --- Skip Connections ---
        # 连接 Stage 2 (64ch) -> Decoder
        self.skip1 = nn.Conv2d(64, 64, 1)
        # 连接 Stage 3 (256ch) -> Decoder
        self.skip2 = nn.Conv2d(256, 64, 1)
        # 连接 Stage 4 (512ch) -> Decoder
        self.skip3 = nn.Conv2d(512, 128, 1)

        # --- Decoder ---
        # Up 4 -> 3 (16x16 -> 32x32)
        self.up4 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            # 输入: 256(上层) + 128(skip3, 来自s4)
            nn.Conv2d(256 + 128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU()
        )

        # Up 3 -> 2 (32x32 -> 64x64)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            # 输入: 128(上层) + 64(skip2, 来自s3)
            nn.Conv2d(128 + 64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )

        # Up 2 -> 1 (64x64 -> 128x128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            # 输入: 64(上层) + 64(skip1, 来自s2)
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        # Final Layer (128x128 -> 256x256)
        # 最后一层通常不接Skip，直接还原
        self.up1 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU()
        )

        # --- Heads ---
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.ipep = IPEP(in_channels=16, iterations=5)

    def forward(self, x):
        # --- Encoder ---
        # Stage 1
        s1 = self.enc1(x)  # [B, 3, 256, 256]
        d1 = self.down1(s1)  # [B, 64, 256, 256]
        p1 = self.pool1(d1)  # [B, 64, 128, 128]

        # Stage 2
        s2 = self.enc2(p1)  # [B, 64, 128, 128] -> 这里的 s2 是 64ch, 用于 skip1
        d2 = self.down2(s2)  # [B, 256, 128, 128]
        p2 = self.pool2(d2)  # [B, 256, 64, 64]

        # Stage 3
        s3 = self.enc3(p2)  # [B, 256, 64, 64] -> 这里的 s3 是 256ch, 用于 skip2
        d3 = self.down3(s3)  # [B, 512, 64, 64]
        p3 = self.pool3(d3)  # [B, 512, 32, 32]

        # Stage 4
        s4 = self.enc4(p3)  # [B, 512, 32, 32] -> 这里的 s4 是 512ch, 用于 skip3
        d4 = self.down4(s4)  # [B, 1024, 32, 32]
        p4 = self.pool4(d4)  # [B, 1024, 16, 16] (Bottom)

        # --- Decoder ---
        # 1. 恢复到 32x32，连接 s4 (32x32, 512ch)
        feat_up4 = self.up4(p4)  # [B, 256, 32, 32]
        sk3 = self.skip3(s4)  # [B, 128, 32, 32] (注意这里改成了 s4!)
        feat_dec4 = torch.cat([feat_up4, sk3], dim=1)
        feat_dec4 = self.dec4(feat_dec4)

        # 2. 恢复到 64x64，连接 s3 (64x64, 256ch)
        feat_up3 = self.up3(feat_dec4)  # [B, 128, 64, 64]
        sk2 = self.skip2(s3)  # [B, 64, 64, 64] (注意这里改成了 s3!)
        feat_dec3 = torch.cat([feat_up3, sk2], dim=1)
        feat_dec3 = self.dec3(feat_dec3)

        # 3. 恢复到 128x128，连接 s2 (128x128, 64ch)
        feat_up2 = self.up2(feat_dec3)  # [B, 64, 128, 128]
        sk1 = self.skip1(s2)  # [B, 64, 128, 128] (注意这里改成了 s2!)
        feat_dec2 = torch.cat([feat_up2, sk1], dim=1)
        feat_dec2 = self.dec2(feat_dec2)

        # 4. 恢复到 256x256 (不接skip，直接输出)
        f0 = self.up1(feat_dec2)  # [B, 16, 256, 256]
        f0 = self.final_conv(f0)

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
        print("模型测试通过！无报错。")
    except Exception as e:
        print(f"测试失败: {e}")