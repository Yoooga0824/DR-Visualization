from flask import Flask, send_file, abort, request
from flask_cors import CORS
import os
from PIL import Image
from io import BytesIO
import json

app = Flask(__name__)
CORS(app)

@app.route('/api/hand_thumb/<prefix>/<int:index>')
def get_hand_thumb(prefix, index):
    mapping_path = rf'D:\Hand-DR-Project\results\embedding_to_image_mapping_{prefix}.json'
    if not os.path.exists(mapping_path):
        print(f"Mapping file not found: {mapping_path}")
        abort(404)
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    if index < 0 or index >= len(mapping_data):
        abort(404)
    img_path = mapping_data[index]['image_path']
    print(f"Trying to access: {img_path}")
    if not os.path.exists(img_path):
        print("Image not found!")
        abort(404)
    try:
        img = Image.open(img_path)
        img.thumbnail((200, 200))
        buf = BytesIO()
        img.save(buf, format='JPEG')
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