import os
import time
from huggingface_hub import snapshot_download

# 1. 依然要设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_with_retry():
    # 这里设置你的下载路径
    local_dir = "D:/HELLO/huggingface/bishe/Straw6D_Raw"
    repo_id = "lishoulun/Straw6D"
    
    print(f"目标目录: {local_dir}")
    print("开始断点续传下载，如果中途断开会自动重连...")
    
    while True:
        try:
            # max_workers=4: 限制并发数，防止被服务器踢下线
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                resume_download=True,
                max_workers=4  
            )
            print("\n🎉 恭喜！所有文件下载完成！")
            break  # 下载完成后跳出循环
            
        except Exception as e:
            # 捕获报错，但不退出
            print(f"\n⚠️ 网络波动：{e}")
            print("⏳ 休息 5 秒后自动重试，接刚才的进度继续下...")
            time.sleep(5)
            continue

if __name__ == "__main__":
    download_with_retry()