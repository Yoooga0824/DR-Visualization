import os
import re
import subprocess
import argparse
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

def choose_method(method_prefixes):
    print("\n可用降维方法：")
    for i, p in enumerate(method_prefixes):
        print(f"  [{i+1}] {p}")
    while True:
        sel = input("请输入要生成的降维方法编号（回车全选，支持多选如1,3）：").strip()
        if not sel:
            return method_prefixes  # 全选
        try:
            idxs = [int(x)-1 for x in sel.split(',') if x.strip().isdigit()]
            chosen = [method_prefixes[i] for i in idxs if 0 <= i < len(method_prefixes)]
            if chosen:
                return chosen
        except Exception:
            pass
        print("输入有误，请重新输入！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, help='只处理指定降维方法前缀')
    args = parser.parse_args()

    # 新增：交互式选择
    if args.prefix:
        method_prefixes = [args.prefix]
    else:
        method_prefixes = choose_method(method_prefixes)

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
