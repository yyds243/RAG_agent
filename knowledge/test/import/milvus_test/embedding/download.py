from transformers import AutoModel
from safetensors.torch import save_file
import os

model_path = r"D:\A_model\bge-m3"
save_path = r"D:\A_model\bge-m3-safetensors"

os.makedirs(save_path, exist_ok=True)

model = AutoModel.from_pretrained(model_path)

save_file(model.state_dict(), os.path.join(save_path, "model.safetensors"))

print("转换完成")