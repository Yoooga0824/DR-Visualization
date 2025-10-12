import numpy as np
import os
import json
from pathlib import Path

# 路径配置
hands_dir = r'D:\Hand-DR-Project\data\raw\Hands\Hands'
anom_dir  = r'D:\Hand-DR-Project\data\raw\Anomalous_Hands'
results_dir = r'D:\Hand-DR-Project\results'
fail_path = r'D:\Hand-DR-Project\data\features\failed_images.txt'

normal_emb_path = r'D:\Hand-DR-Project\data\features\TSNE_embedding.npy'
anom_emb_path   = r'D:\Hand-DR-Project\data\features\anomalous_TSNE_embedding.npy'

# 1. 读取 failed_images
with open(fail_path, 'r', encoding='utf-8') as f:
    failed = set(line.strip() for line in f if line.strip())

# 2. 筛选未失败图片
all_normal_images = sorted(list(Path(hands_dir).glob("Hand_*.jpg")))
normal_images = [img for img in all_normal_images if img.name not in failed]

# 3. 加载 embedding
normal_embedding = np.load(normal_emb_path)
assert len(normal_images) == len(normal_embedding), (
    f"过滤后正常图片数({len(normal_images)})应和embedding数({len(normal_embedding)})一致！")

normal_mapping = []
for i, (img, emb) in enumerate(zip(normal_images, normal_embedding)):
    normal_mapping.append({
        "index": i,
        "embedding_coords": emb.tolist(),
        "image_path": str(img),
        "features": []
    })

# 4. 异常图片（如无失败图片，可全量用）
anom_embedding = np.load(anom_emb_path)
anom_images = sorted(list(Path(anom_dir).glob("*.jpg")))
anom_images = anom_images[:len(anom_embedding)]  # 若有异常失败图片可类似剔除
anom_mapping = []
for i, (img, emb) in enumerate(zip(anom_images, anom_embedding)):
    anom_mapping.append({
        "index": len(normal_mapping)+i,
        "embedding_coords": emb.tolist(),
        "image_path": str(img),
        "features": []
    })

# 合并并保存
mapping_data = normal_mapping + anom_mapping
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "embedding_to_image_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(mapping_data, f, indent=2, ensure_ascii=False)

print(f"embedding_to_image_mapping.json 已生成！共{len(mapping_data)}条。")

from collections import Counter

paths = [item['image_path'] for item in mapping_data]
counter = Counter(paths)
repeats = [(p, c) for p, c in counter.items() if c > 1]
print(f"总点数：{len(paths)}")
print(f"唯一图片数：{len(set(paths))}")
print(f"出现多次的图片及重复次数：{repeats[:10]}")