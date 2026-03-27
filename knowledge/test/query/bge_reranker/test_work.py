from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    model_name_or_path=r"D:\A_model\bge_reranker",
    device="cuda",      # GPU 加速
    use_fp16=True       # 半精度推理
)

my_query = "如何清洗羽绒服"
# 计算相关性得分
candidates = [
    "羽绒服可以用洗衣机随意搅拌清洗。",
    "羽绒服建议使用中性洗涤剂，手洗或滚筒洗衣机轻柔模式，不可拧干。",
    "今天天气不错，适合出去跑步。",  # 无关干扰项
    "清洗羽绒服前请查看洗涤标签，通常建议干洗或专用清洗液。",
    "Python 是一种编程语言。"       # 完全无关项
]

pairs = [(my_query, candidate) for candidate in candidates]
scores = reranker.compute_score(pairs)

print(scores)
#
results = list(zip(candidates, scores))
results.sort(key=lambda x: x[1], reverse=True)

print("\n--- 相关性排名 (从高到低) ---")
for i, (text, score) in enumerate(results, 1):
    print(f"Top {i}: [得分: {score:.4f}] -> {text}")
