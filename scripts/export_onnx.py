import torch
import os
from models.darknet import Darknet

def export_model():
    """将 PyTorch 模型导出为 ONNX，支持后续 TensorRT 加速"""
    print("=== 启动模型导出流水线 (Model Export Pipeline) ===")
    
    # 1. 加载模型
    model = Darknet()
    checkpoint_path = "darknet_strawberry_checkpoint.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"加载最优权重文件: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    model.eval()

    # 2. 准备虚拟输入 (800x600)
    dummy_input = torch.randn(1, 3, 600, 800)
    output_path = "strawberry_pose_model.onnx"

    # 3. 导出为 ONNX
    print(f"正在转换模型至 ONNX 格式 (FP32)...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"导出成功: {output_path}")

    # 4. 说明后续 TensorRT 流程
    print("\n[工程落地建议]")
    print("1. 请在 Jetson 终端执行以下命令进行 TensorRT 加速:")
    print(f"   trtexec --onnx={output_path} --saveEngine=strawberry_pose.engine --fp16")
    print("2. 该操作将执行算子融合 (Operator Fusion) 并启用 FP16 半精度量化。")

if __name__ == "__main__":
    export_model()
