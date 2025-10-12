import os
import re
import subprocess
from pathlib import Path

FEATURES_DIR = r'D:\Hand-DR-Project\data\features'
CODE_DIR = r'D:\Hand-DR-Project\code'
RESULTS_DIR = r'D:\Hand-DR-Project\results'

# 1. 搜索所有 embedding 前缀
files = os.listdir(FEATURES_DIR)
pattern = re.compile(r'^(anomalous_)?([A-Za-z0-9_]+)_embedding\.npy$')
method_prefixes = set()
for fname in files:
    m = pattern.match(fname)
    if m and not fname.startswith('anomalous_'):
        method_prefixes.add(m.group(2))
method_prefixes = sorted(method_prefixes)

print(f"检测到降维方法: {method_prefixes}")

for prefix in method_prefixes:
    print(f"\n==== 处理方法: {prefix} ====")
    # 2. 生成 mapping
    mapping_cmd = [
        'python', os.path.join(CODE_DIR, 'make_embedding_mapping.py'),
        '--prefix', prefix
    ]
    print(' '.join(mapping_cmd))
    subprocess.run(mapping_cmd, check=True)

    # 3. 生成 HTML
    analysis_cmd = [
        'python', os.path.join(CODE_DIR, 'boundary_analysis.py'),
        '--prefix', prefix
    ]
    print(' '.join(analysis_cmd))
    subprocess.run(analysis_cmd, check=True)

print("\n全部方法处理完成！每个方法都已生成独立 mapping 和 HTML 文件。")
