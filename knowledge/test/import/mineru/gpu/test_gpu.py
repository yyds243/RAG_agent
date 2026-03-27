import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ 成功！当前使用的 CUDA 版本: {torch.version.cuda}")
    print(f"🚀 显卡名称: {torch.cuda.get_device_name(0)}")
else:
    print("❌ 遗憾：当前是 CPU 版本，或者无法检测到显卡。")

    print("💡 提示：这通常是因为安装了 cpu-only 的包，或者 CUDA 驱动不匹配。")

    # mineru -p D:\A-py\AI_project\smartku\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf -o D:\A-py\AI_project\smartku\knowledge\processor\import_process\output_temp_dir --source local