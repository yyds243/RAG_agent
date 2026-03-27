from typing import Optional
import torch
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from torch import device



bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=r'D:\A_model\bge-m3', # Specify the model name
    device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=True # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)

print(bge_m3_ef)

texts = ["Milvus contains a sparse vector field."]
embeddings = bge_m3_ef(texts)

# 打印出的就是你图片中对应的格式
print(embeddings)

print(embeddings["sparse"])

# bge_m3_ef : Optional[BGEM3EmbeddingFunction] = None
#
# def get_beg_m3_embedding_models():
#     global bge_m3_ef
#
#     if bge_m3_ef is None:
#         return bge_m3_ef
#     # 1. 获取参数
#     model_name = os.ge
#
#     bge_m3_ef = BGEM3EmbeddingFunction(
#         model_name=model_name,
#         devices=device,
#         use_fp16=use_fp16
#     )
#
#     return bge_m3_ef
# if __name__ == '__main__':
#     embedding_model = get_beg_m3_embedding_models()
#
#     query = ""
#     embedding_model.encode_queries([query])
