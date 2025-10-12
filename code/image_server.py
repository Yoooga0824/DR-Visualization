from flask import Flask, send_file, abort
from flask_cors import CORS
import os
from PIL import Image
from io import BytesIO
import json

app = Flask(__name__)
CORS(app)

MAPPING_PATH = r'D:\Hand-DR-Project\results\embedding_to_image_mapping.json'

@app.route('/api/hand_thumb/<int:index>')
def get_hand_thumb(index):
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
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

@app.route('/')
def index():
    return 'Image server is running!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678, debug=True)