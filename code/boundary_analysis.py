import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from detection.BDLLE import bd_lle
import sys
import os
import time
from tqdm import tqdm
import json
from pathlib import Path

# 添加自定义模块路径（如果需要可以保留）
sys.path.append(os.path.abspath('./code/'))

# 导入tool_functions中的wasserstein_loss函数
from tool_functions import wasserstein_loss

print("成功导入wasserstein_loss函数")

# 获取和code同级的results目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_normalize_data():
    print("加载并归一化数据...")

    # 数据路径保持和项目根目录对应
    normal_embedding = np.load(os.path.join(PROJECT_ROOT, 'data', 'features', 'TSNE_embedding.npy'))
    anomalous_embedding = np.load(os.path.join(PROJECT_ROOT, 'data', 'features', 'anomalous_TSNE_embedding.npy'))

    # 合并数据
    combined_embedding = np.vstack([normal_embedding, anomalous_embedding])

    # 归一化到[0,1]范围
    min_vals = np.min(combined_embedding, axis=0)
    max_vals = np.max(combined_embedding, axis=0)
    normalized_embedding = (combined_embedding - min_vals) / (max_vals - min_vals)

    # 创建标签
    normal_labels = np.zeros(len(normal_embedding))
    anomalous_labels = np.ones(len(anomalous_embedding))
    combined_labels = np.concatenate([normal_labels, anomalous_labels])

    print(f"归一化后数据形状: {normalized_embedding.shape}")
    print(f"正常样本: {len(normal_embedding)}, 异常样本: {len(anomalous_embedding)}")

    return normalized_embedding, combined_labels

def detect_boundary_fixed_params(normalized_embedding):
    print("使用固定参数检测边界点:")
    print("注意: 由于K值较大，计算可能需要较长时间...")

    start_time = time.time()

    # 使用固定参数
    boundary_points, B = bd_lle(normalized_embedding, d=2, K=110)

    # 计算耗时
    computation_time = time.time() - start_time
    print(f"BD-LLE计算完成，耗时: {computation_time:.2f}秒 ({computation_time / 60:.1f}分钟)")

    # 应用固定阈值
    maxB = np.max(B)
    threshold = 0.7 * maxB
    boundary_indices = np.where(B >= threshold)[0]

    print(f"检测到 {len(boundary_indices)} 个边界点")
    return boundary_indices

def calculate_wasserstein_metrics(normalized_embedding, labels, boundary_indices):
    print("使用Wasserstein距离计算边界指标...")

    boundary_points = normalized_embedding[boundary_indices]

    # 分离正常和异常点
    anomalous_mask = labels == 1

    anomalous_points = normalized_embedding[anomalous_mask]

    # 将numpy数组转换为PyTorch Tensor（wasserstein_loss需要的格式）
    import torch
    anomalous_tensor = torch.tensor(anomalous_points, dtype=torch.float32)
    boundary_tensor = torch.tensor(boundary_points, dtype=torch.float32)

    # 计算异常点到边界的Wasserstein距离
    print("计算异常点到边界的Wasserstein距离...")
    anomalous_wasserstein = wasserstein_loss(anomalous_tensor, boundary_tensor)

    print(f"异常点到边界的Wasserstein距离: {anomalous_wasserstein:.4f}")

    return float(anomalous_wasserstein)

def generate_interactive_html(normalized_embedding, labels, boundary_indices, metrics):
    print("生成交互式HTML文件...")

    mapping_path = os.path.join(RESULTS_DIR, 'embedding_to_image_mapping.json')
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    N = len(mapping_data)
    normalized_embedding = normalized_embedding[:N]
    labels = labels[:N]

    points_data = []
    for i in range(N):
        point_info = {
            'index': i,  # 直接用行号作为唯一 index
            'x': float(normalized_embedding[i, 0]),
            'y': float(normalized_embedding[i, 1]),
            'cluster': 'normal' if labels[i] == 0 else 'anomalous',
            'is_boundary': int(i) in boundary_indices
        }
        points_data.append(point_info)

    IMAGE_API_URL = "http://172.16.72.114:5678/api/hand_thumb/"

    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>边界分析可视化 - 交互式</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #eee; }}
        .content {{ display: flex; gap: 20px; min-height: 600px; }}
        .plot-container {{ flex: 3; border: 1px solid #ddd; border-radius: 4px; padding: 10px; background: #fafafa; }}
        .info-panel {{ flex: 1; border: 1px solid #ddd; border-radius: 4px; padding: 15px; background: #f9f9f9; min-width: 300px; }}
        .metrics {{ background: #e8f4fd; padding: 15px; border-radius: 4px; margin-bottom: 20px; }}
        .metric-item {{ margin: 8px 0; font-size: 14px; }}
        .point-info {{ margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 4px; font-size: 14px; }}
        .image-preview {{ text-align: center; margin-top: 20px; }}
        #preview-image {{ max-width: 100%; max-height: 300px; border-radius: 5px; display: none; }}
        .selected-images {{ margin-top: 20px; max-height: 400px; overflow-y: auto; }}
        .selected-image {{ margin: 5px; border: 1px solid #ddd; border-radius: 3px; padding: 5px; }}
        .controls {{ margin-bottom: 20px; text-align: center; }}
        .btn {{ background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; margin: 0 5px; }}
        .btn:hover {{ background-color: #45a049; }}
        .legend {{ display: flex; justify-content: center; margin: 15px 0; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; margin: 0 10px; }}
        .legend-color {{ width: 15px; height: 15px; margin-right: 5px; border-radius: 50%; }}
        .normal-color {{ background-color: lightblue; }}
        .anomalous-color {{ background-color: red; }}
        .boundary-color {{ background-color: green; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>边界分析可视化 - 交互式</h1>
            <p>点击或圈选点查看对应的手部原始图像 (K=110, Threshold=0.7)</p>
        </div>
        <div class="metrics">
            <h3>分析指标</h3>
            <div class="metric-item"><strong>边界点数量:</strong> {metrics['boundary_count']}</div>
            <div class="metric-item"><strong>Wasserstein距离:</strong> {metrics['anomalous_wasserstein']:.6f}</div>
            <div class="metric-item"><strong>正常样本:</strong> {np.sum(labels == 0)}</div>
            <div class="metric-item"><strong>异常样本:</strong> {np.sum(labels == 1)}</div>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color normal-color"></div>正常手部图像</div>
            <div class="legend-item"><div class="legend-color anomalous-color"></div>异常手部图像</div>
            <div class="legend-item"><div class="legend-color boundary-color"></div>边界点</div>
        </div>
        <div class="content">
            <div class="plot-container">
                <h3>交互式散点图</h3>
                <div id="plotly-chart" style="width:100%; height:600px;"></div>
            </div>
            <div class="info-panel">
                <div class="point-info">
                    <h3>点信息</h3>
                    <div id="point-info-content">点击图表中的点查看详细信息</div>
                </div>
                <div class="image-preview">
                    <h3>图像预览</h3>
                    <div id="image-preview-content">
                        <p>选择点后显示对应图像</p>
                        <img id="preview-image" alt="预览图像">
                    </div>
                </div>
                <div class="selected-images">
                    <h3>选中点图像</h3>
                    <div id="selected-images-content">
                        <p>圈选多个点后显示所有图像</p>
                    </div>
                </div>
                <div class="controls">
                    <button class="btn" onclick="zoomToBoundary()">缩放至边界点</button>
                    <button class="btn" onclick="resetView()">重置视图</button>
                </div>
            </div>
        </div>
    </div>
    <script>
        const IMAGE_API_URL = "{IMAGE_API_URL}";
        const pointsData = {json.dumps(points_data)};
        const boundaryIndices = {json.dumps(boundary_indices.tolist())};
        let plot = null;
        let currentPointIndex = null;
        let selectedPoints = [];

        function initPlot() {{
            const normalPoints = pointsData.filter(p => p.cluster === 'normal' && !p.is_boundary);
            const anomalousPoints = pointsData.filter(p => p.cluster === 'anomalous' && !p.is_boundary);
            const boundaryPoints = pointsData.filter(p => p.is_boundary);

            const traceNormal = {{
                x: normalPoints.map(p => p.x),
                y: normalPoints.map(p => p.y),
                mode: 'markers',
                type: 'scatter',
                name: '正常手部图像',
                marker: {{ color: 'lightblue', size: 6, opacity: 0.7 }},
                text: normalPoints.map(p => `索引: ${{p.index}}<br>类型: 正常`),
                hoverinfo: 'text',
                customdata: normalPoints.map(p => p.index)
            }};
            const traceAnomalous = {{
                x: anomalousPoints.map(p => p.x),
                y: anomalousPoints.map(p => p.y),
                mode: 'markers',
                type: 'scatter',
                name: '异常手部图像',
                marker: {{ color: 'red', size: 8, opacity: 0.8, symbol: 'x' }},
                text: anomalousPoints.map(p => `索引: ${{p.index}}<br>类型: 异常`),
                hoverinfo: 'text',
                customdata: anomalousPoints.map(p => p.index)
            }};
            const traceBoundary = {{
                x: boundaryPoints.map(p => p.x),
                y: boundaryPoints.map(p => p.y),
                mode: 'markers',
                type: 'scatter',
                name: '边界点',
                marker: {{ color: 'green', size: 10, opacity: 0.9, symbol: 'diamond' }},
                text: boundaryPoints.map(p => `索引: ${{p.index}}<br>类型: 边界点`),
                hoverinfo: 'text',
                customdata: boundaryPoints.map(p => p.index)
            }};

            const layout = {{
                title: '边界分析可视化 (K=110, Threshold=0.7)',
                xaxis: {{ title: '成分 1 (归一化)' }},
                yaxis: {{ title: '成分 2 (归一化)' }},
                hovermode: 'closest',
                showlegend: false,
                height: 600,
                plot_bgcolor: '#fafafa',
                paper_bgcolor: '#ffffff',
                dragmode: 'lasso'  // 可选改成 'select'
            }};
            const config = {{ responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToAdd: ['toggleHover', 'resetViews'], scrollZoom: true }};
            plot = Plotly.newPlot('plotly-chart', [traceNormal, traceAnomalous, traceBoundary], layout, config);

            document.getElementById('plotly-chart').on('plotly_click', function(data) {{
                if (data.points && data.points.length > 0) {{
                    const pointIndex = data.points[0].customdata;
                    if (pointIndex !== null && pointIndex !== undefined) {{
                        showPointInfo(pointIndex);
                    }}
                }}
            }});
            document.getElementById('plotly-chart').on('plotly_selected', function(data) {{
                if (data.points && data.points.length > 0) {{
                    selectedPoints = [];
                    data.points.forEach(point => {{
                        if (point.customdata !== null && point.customdata !== undefined) {{
                            selectedPoints.push(point.customdata);
                        }}
                    }});
                    showSelectedImages(selectedPoints);
                }}
            }});
            document.getElementById('plotly-chart').on('plotly_deselect', function() {{
                selectedPoints = [];
                document.getElementById('selected-images-content').innerHTML = '<p>圈选多个点后显示所有图像</p>';
                Plotly.restyle('plotly-chart', {{'selectedpoints': null}});
            }});
        }}

        function showPointInfo(pointIndex) {{
            // 强制类型一致
            const point = pointsData.find(p => Number(p.index) === Number(pointIndex));
            if (!point) return;

            currentPointIndex = pointIndex;

            document.getElementById('point-info-content').innerHTML = `
                <div><strong>索引:</strong> ${{point.index}}</div>
                <div><strong>坐标:</strong> (${{point.x.toFixed(3)}}, ${{point.y.toFixed(3)}})</div>
                <div><strong>类型:</strong> ${{point.cluster === 'normal' ? '正常' : '异常'}}</div>
                <div><strong>边界点:</strong> ${{point.is_boundary ? '是' : '否'}}</div>
            `;

            const imgElement = document.getElementById('preview-image');
            const imagePreview = document.getElementById('image-preview-content');
            const imgUrl = IMAGE_API_URL + point.index;
            imgElement.src = imgUrl;
            imgElement.onload = function() {{
                imgElement.style.display = 'block';
                imagePreview.innerHTML = '<p>手部图像预览</p>';
            }};
            imgElement.onerror = function() {{
                imgElement.style.display = 'none';
                imagePreview.innerHTML = '<p>该点对应的图像未找到</p>';
            }};
        }}

        function showSelectedImages(pointIndices) {{
            const selectedImagesContent = document.getElementById('selected-images-content');
            selectedImagesContent.innerHTML = '';
            if (pointIndices.length === 0) {{
                selectedImagesContent.innerHTML = '<p>未选中任何点</p>';
                return;
            }}
            selectedImagesContent.innerHTML = `<p>选中了 ${{pointIndices.length}} 个点</p>`;
            pointIndices = [...new Set(pointIndices)];
            pointIndices.forEach(pointIndex => {{
                const point = pointsData.find(p => Number(p.index) === Number(pointIndex));
                if (point) {{
                    const imgDiv = document.createElement('div');
                    imgDiv.className = 'selected-image';
                    const imgUrl = IMAGE_API_URL + point.index;
                    imgDiv.innerHTML = `
                        <div><strong>索引 ${{point.index}}</strong></div>
                        <img src="${{imgUrl}}" alt="手部图像 ${{point.index}}" style="max-width: 100px; max-height: 100px;" onerror="this.style.display='none'">
                        <div>类型: ${{point.cluster === 'normal' ? '正常' : '异常'}}</div>
                        <div>边界点: ${{point.is_boundary ? '是' : '否'}}</div>
                    `;
                    selectedImagesContent.appendChild(imgDiv);
                }}
            }});
        }}

        function zoomToBoundary() {{
            if (boundaryIndices.length === 0) return;
            const boundaryPoints = pointsData.filter(p => p.is_boundary);
            const xValues = boundaryPoints.map(p => p.x);
            const yValues = boundaryPoints.map(p => p.y);
            const xRange = [Math.min(...xValues) - 0.1, Math.max(...xValues) + 0.1];
            const yRange = [Math.min(...yValues) - 0.1, Math.max(...yValues) + 0.1];
            Plotly.relayout('plotly-chart', {{
                'xaxis.range': xRange,
                'yaxis.range': yRange
            }});
        }}

        function resetView() {{
            Plotly.relayout('plotly-chart', {{
                'xaxis.range': [0, 1],
                'yaxis.range': [0, 1]
            }});
            selectedPoints = [];
            document.getElementById('point-info-content').innerHTML = '点击图表中的点查看详细信息';
            document.getElementById('preview-image').style.display = 'none';
            document.getElementById('image-preview-content').innerHTML = '<p>选择点后显示对应图像</p>';
            document.getElementById('selected-images-content').innerHTML = '<p>圈选多个点后显示所有图像</p>';
            Plotly.restyle('plotly-chart', {{'selectedpoints': null}});
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            initPlot();
        }});
    </script>
</body>
</html>
    """

    html_path = os.path.join(RESULTS_DIR, 'TSNE.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"交互式HTML文件已保存至: {html_path}")
    return html_path

def visualize_results(normalized_embedding, labels, boundary_indices):
    print("生成可视化图表...")

    plt.figure(figsize=(12, 10))

    # 分离数据
    normal_mask = labels == 0
    anomalous_mask = labels == 1

    # 绘制正常点
    plt.scatter(normalized_embedding[normal_mask, 0], normalized_embedding[normal_mask, 1],
                c='lightblue', alpha=0.6, s=10, label='Normal hand image')

    # 绘制异常点
    plt.scatter(normalized_embedding[anomalous_mask, 0], normalized_embedding[anomalous_mask, 1],
                c='red', alpha=0.8, s=50, marker='x', label='Anomalous hand image')

    # 绘制边界点
    boundary_points = normalized_embedding[boundary_indices]
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1],
                c='green', alpha=0.7, s=20, marker='o', label='Boundary point')

    plt.title('Normalized latent space boundary analysis(K=110, Threshold=0.7)', fontsize=14)
    plt.xlabel('Component 1 (Normalization)')
    plt.ylabel('Component 2 (Normalization)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(RESULTS_DIR, 'TSNE_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存至: {output_path}")

    plt.show()
    return output_path

def main():
    print("=" * 50)
    print("固定参数边界分析")
    print("=" * 50)

    total_start_time = time.time()

    normalized_embedding, labels = load_and_normalize_data()
    boundary_indices = detect_boundary_fixed_params(normalized_embedding)
    anomalous_wasserstein = calculate_wasserstein_metrics(normalized_embedding, labels, boundary_indices)
    output_path = visualize_results(normalized_embedding, labels, boundary_indices)

    metrics = {
        'boundary_count': len(boundary_indices),
        'anomalous_wasserstein': anomalous_wasserstein,
    }
    html_path = generate_interactive_html(normalized_embedding, labels, boundary_indices, metrics)

    results = {
        'boundary_count': len(boundary_indices),
        'anomalous_wasserstein': anomalous_wasserstein,
    }

    results_path = os.path.join(RESULTS_DIR, 'TSNE_metrics.txt')
    with open(results_path, 'w') as f:
        f.write("固定参数边界分析结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"边界点数量: {results['boundary_count']}\n")
        f.write(f"异常点到边界的Wasserstein距离: {results['anomalous_wasserstein']:.6f}\n")
        f.write("\n注: 使用Wasserstein距离计算边界指标\n")

    total_time = time.time() - total_start_time
    print("=" * 50)
    print("分析完成!")
    print(f"总耗时: {total_time:.2f}秒 ({total_time / 60:.1f}分钟)")
    print(f"边界点数量: {results['boundary_count']}")
    print(f"异常点到边界Wasserstein距离: {results['anomalous_wasserstein']:.6f}")
    print(f"静态图表保存至: {output_path}")
    print(f"交互式HTML保存至: {html_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()