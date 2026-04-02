import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 基础卷积块 (ConvBlock) ---
class ConvBlock(nn.Module):
    """
    基础卷积块：Conv2d + BatchNorm2d + LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

# --- 2. 轻量化 Transformer 全局感知模块 (Neck) ---
class TransformerNeck(nn.Module):
    """
    接收 Backbone 输出的 [B, 1024, 18, 25] 特征图，
    进行通道降维、空间展开和全局自注意力增强。
    """
    def __init__(self, in_channels=1024, d_model=256, nhead=8, num_layers=1):
        super(TransformerNeck, self).__init__()
        
        # 降维：1x1 卷积压缩通道数 (1024 -> 256)
        self.reduce_conv = ConvBlock(in_channels, d_model, kernel_size=1)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: [B, 1024, 18, 25]
        B, C, H, W = x.shape
        
        # 1. 降维 [B, 256, 18, 25]
        x = self.reduce_conv(x)
        
        # 2. 序列化 [B, 450, 256] (18 * 25 = 450)
        # flatten(2) 将 H, W 展平，transpose(1, 2) 调整维度为 (Batch, Seq, Dim)
        x_seq = x.flatten(2).transpose(1, 2)
        
        # 3. 全局自注意力计算
        x_enhanced = self.transformer(x_seq)
        
        # 4. 重新恢复特征图空间维度 [B, 256, 18, 25]
        x_out = x_enhanced.transpose(1, 2).reshape(B, -1, H, W)
        
        return x_out

# --- 3. 串联式 CNN–Transformer 混合网络 (Darknet-19) ---
class Darknet19(nn.Module):
    """
    标准的 Darknet-19 Backbone (19个卷积层 + 5个池化层)
    """
    def __init__(self, num_outputs=22):
        super(Darknet19, self).__init__()
        
        # --- Backbone: Darknet-19 ---
        # 输入: [B, 3, 600, 800]
        
        # Block 1
        self.conv1 = ConvBlock(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # [B, 32, 300, 400]
        
        # Block 2
        self.conv2 = ConvBlock(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # [B, 64, 150, 200]
        
        # Block 3
        self.conv3 = ConvBlock(64, 128, 3, padding=1)
        self.conv4 = ConvBlock(128, 64, 1, padding=0)
        self.conv5 = ConvBlock(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # [B, 128, 75, 100]
        
        # Block 4
        self.conv6 = ConvBlock(128, 256, 3, padding=1)
        self.conv7 = ConvBlock(256, 128, 1, padding=0)
        self.conv8 = ConvBlock(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2) # [B, 256, 37, 50]
        
        # Block 5
        self.conv9 = ConvBlock(256, 512, 3, padding=1)
        self.conv10 = ConvBlock(512, 256, 1, padding=0)
        self.conv11 = ConvBlock(256, 512, 3, padding=1)
        self.conv12 = ConvBlock(512, 256, 1, padding=0)
        self.conv13 = ConvBlock(256, 512, 3, padding=1)
        # 注意: 这里根据输入尺寸 600x800, 下采样 16倍后是 37x50
        # 第 5 次池化后将下采样 32倍，变成 18x25
        self.pool5 = nn.MaxPool2d(2, 2) # [B, 512, 18, 25]
        
        # Block 6
        self.conv14 = ConvBlock(512, 1024, 3, padding=1)
        self.conv15 = ConvBlock(1024, 512, 1, padding=0)
        self.conv16 = ConvBlock(512, 1024, 3, padding=1)
        self.conv17 = ConvBlock(1024, 512, 1, padding=0)
        self.conv18 = ConvBlock(512, 1024, 3, padding=1) # [B, 1024, 18, 25]
        
        # --- Neck: Transformer 全局增强 ---
        self.neck = TransformerNeck(in_channels=1024, d_model=256)
        
        # --- Head: 22维回归预测 (Grid-based Head) ---
        # 输入: [B, 256, 18, 25]
        # 输出: [B, 22, 18, 25]
        # 使用 1x1 卷积代替全连接层
        self.head_conv = nn.Conv2d(256, num_outputs, kernel_size=1)

    def forward(self, x):
        # 1. Backbone 前向传播
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool3(self.conv5(x))
        
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool4(self.conv8(x))
        
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool5(self.conv13(x))
        
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x) # [B, 1024, 18, 25]
        
        # 2. Neck 全局感知增强
        x = self.neck(x) # [B, 256, 18, 25]
        
        # 3. Head 空间回归
        x = self.head_conv(x) # [B, 22, 18, 25]
        
        return x

# 为了兼容性，定义一个别名 Darknet
class Darknet(Darknet19):
    def __init__(self):
        super(Darknet, self).__init__(num_outputs=22)

# --- 4. 测试代码 ---
if __name__ == '__main__':
    # 模拟输入：BatchSize=2, 通道=3, 高=600, 宽=800
    dummy_input = torch.randn(2, 3, 600, 800)
    
    # 初始化模型
    model = Darknet()
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    
    # 前向传播测试
    print("开始测试前向传播...")
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输入张量维度: {dummy_input.shape}")
    print(f"输出张量维度: {output.shape} <-- 完美匹配 [B, 22, 18, 25]！")
    
    if output.shape == (2, 22, 18, 25):
        print("测试通过！网络架构符合 Grid-based Single-Shot 设计要求。")
    else:
        print("测试失败！请检查网络层参数配置。")
