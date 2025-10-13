from flask import Flask, send_file, abort, request
from flask_cors import CORS
import os
from PIL import Image
from io import BytesIO
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)


@app.route('/api/hand_thumb/<prefix>/<int:index>')
def get_hand_thumb(prefix, index):
    # 支持递归查找所有 results/*-results/embedding_to_image_mapping_{prefix}.json
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    mapping_candidates = []
    dataset_filter = request.args.get('dataset')  # 可选：指定数据集，缩小匹配范围
    # 先查找所有 results/*-results/embedding_to_image_mapping_{prefix}.json
    for root, dirs, files in os.walk(results_root):
        for fname in files:
            if fname == f'embedding_to_image_mapping_{prefix}.json':
                full = os.path.join(root, fname)
                if dataset_filter:
                    # 约定路径中包含 <dataset>-results
                    if f"{dataset_filter}-results" in os.path.normpath(full):
                        mapping_candidates.append(full)
                else:
                    mapping_candidates.append(full)
    # fallback: 兼容老路径
    mapping_candidates.append(os.path.join(results_root, f'embedding_to_image_mapping_{prefix}.json'))

    mapping_path = None
    for candidate in mapping_candidates:
        if os.path.exists(candidate):
            mapping_path = candidate
            break
    if not mapping_path:
        print(f"Mapping file not found for prefix {prefix}, tried: {mapping_candidates}")
        abort(404)
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    if index < 0 or index >= len(mapping_data):
        abort(404)
    img_path = mapping_data[index]['image_path']
    # 如果映射文件中遗留旧的绝对路径（例如包含 Hand-DR-Project），尝试替换为当前工作区根路径
    try:
        if isinstance(img_path, str) and ('Hand-DR-Project' in img_path or 'DR-Visualization' in img_path):
            p = Path(img_path)
            # 用当前仓库根替换旧根（假设相对 data 结构未变）
            repo_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            # 找到从 data/ 之后的相对路径
            parts = p.parts
            if 'data' in parts:
                data_idx = parts.index('data')
                rel_from_data = Path(*parts[data_idx+1:]) if data_idx+1 <= len(parts)-1 else Path('')
                candidate = repo_root / 'data' / rel_from_data
                if candidate.exists():
                    img_path = str(candidate)
    except Exception:
        pass
    print(f"Trying to access: {img_path}")
    if not os.path.exists(img_path):
        print("Image not found!")
        abort(404)
    try:
        img = Image.open(img_path)
        img.thumbnail((200, 200))
        # 为兼容 PNG 等模式（如 RGBA/P/LA），统一转换为可写 JPEG 的模式
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error opening image: {e}")
        abort(500)

# 兼容老接口，默认用TSNE
@app.route('/api/hand_thumb/<int:index>')
def get_hand_thumb_default(index):
    return get_hand_thumb('TSNE', index)



# 主页返回纯文本
@app.route('/')
def index():
    return 'Image server is running!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678, debug=True)