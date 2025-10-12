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

# 全局K值设置
K_NEIGHBORS = 100  # 修改此处即可全局生效

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

    # 使用全局K值
    boundary_points, B = bd_lle(normalized_embedding, d=2, K=K_NEIGHBORS)

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

    # 检查本地 plotly js 是否存在
    local_plotly_path = os.path.join(RESULTS_DIR, 'plotly-2.24.1.min.js')
    if os.path.exists(local_plotly_path):
        plotly_js_tag = f'<script src="./plotly-2.24.1.min.js"></script>'
    else:
        plotly_js_tag = '<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>'

    html_content = f"""
<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>边界分析可视化 - 交互式</title>
    {plotly_js_tag}
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Montserrat', Arial, sans-serif;
            margin: 0; padding: 0;
            min-height: 100vh;
            background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%);
        }}
        .container {{
            max-width: 1400px;
            margin: 32px auto;
            background: #fff;
            padding: 32px 24px 24px 24px;
            border-radius: 18px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
            border: 1.5px solid #e3e8f0;
        }}
        .header {{
            position: relative;
            text-align: center;
            margin-bottom: 28px;
            padding: 24px 18px 28px 18px;
            border-radius: 14px;
            background: linear-gradient(90deg, rgba(54,209,196,0.15) 0%, rgba(91,134,229,0.15) 100%);
            border: 1px solid #e3e8f0;
        }}
        .header::after {{
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 14px;
            box-shadow: 0 10px 28px rgba(91, 134, 229, 0.18) inset;
            pointer-events: none;
        }}
        .header h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #2b5876 0%, #4e4376 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            letter-spacing: 1.2px;
            margin: 0 0 10px 0;
        }}
        .header .sub {{
            color: #5a6d8a;
            font-size: 1.02rem;
            margin: 0 auto 10px auto;
        }}
        .pills {{
            display: inline-flex;
            gap: 10px;
            align-items: center;
            margin-top: 8px;
        }}
        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #ffffff;
            border: 1px solid #e3e8f0;
            color: #2b5876;
            padding: 6px 12px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.92rem;
            box-shadow: 0 2px 8px rgba(31, 38, 135, 0.08);
        }}
        .content {{
            display: flex;
            gap: 28px;
            min-height: 600px;
        }}
        @media (max-width: 1100px) {{
            .content {{ flex-direction: column; }}
        }}
        .plot-container {{
            flex: 3;
            border-radius: 12px;
            background: #f7fafd;
            box-shadow: 0 2px 12px 0 rgba(31, 38, 135, 0.07);
            padding: 18px 10px 10px 10px;
            border: 1.5px solid #e3e8f0;
        }}
        .info-panel {{
            flex: 1;
            border-radius: 12px;
            background: #f9fbfd;
            box-shadow: 0 2px 12px 0 rgba(31, 38, 135, 0.07);
            padding: 20px 18px;
            border: 1.5px solid #e3e8f0;
            min-width: 320px;
        }}
        .metrics {{
            margin-bottom: 18px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }}
        @media (max-width: 900px) {{
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        @media (max-width: 560px) {{
            .metrics-grid {{ grid-template-columns: 1fr; }}
        }}
        .metric-card {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 14px 12px;
            border-radius: 12px;
            border: 1px solid #e3e8f0;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            box-shadow: 0 4px 12px rgba(31, 38, 135, 0.08);
        }}
        .metric-icon {{
            width: 34px; height: 34px; border-radius: 10px; flex: 0 0 34px;
            display: grid; place-items: center; font-size: 18px; font-weight: 800; color: #fff;
            background: linear-gradient(135deg, #36d1c4 0%, #5b86e5 100%);
            box-shadow: 0 4px 10px rgba(91, 134, 229, 0.28);
        }}
        .metric-content strong {{
            display: block; font-size: 1.06rem; color: #2b5876;
        }}
        .metric-content span {{
            display: block; font-size: 0.86rem; color: #5a6d8a;
        }}
        .point-info {{
            margin-top: 18px;
            padding: 16px 14px;
            background: linear-gradient(180deg, #fffdf3 0%, #fff8d6 100%);
            border-radius: 12px;
            font-size: 1.01rem;
            box-shadow: 0 6px 18px rgba(255, 215, 0, 0.15);
            border: 1px solid #f3e7a9;
        }}
        .point-info h3 {{
            margin: 0 0 10px 0;
            font-size: 1.15rem;
            color: #8a6d00;
        }}
        .info-header {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            margin-bottom: 10px;
        }}
        .chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            background: #fff;
            border: 1px solid #f0e6a6;
            color: #6b5b00;
            font-weight: 600;
            font-size: 0.95rem;
            box-shadow: 0 1px 4px rgba(138, 109, 0, 0.08);
        }}
        .chip .dot {{
            width: 8px; height: 8px; border-radius: 50%; background: #d4a017;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.3px;
            border: 1px solid transparent;
        }}
        .badge-normal {{
            background: linear-gradient(135deg, #e3f2ff 0%, #f0f7ff 100%);
            color: #1a6bb8;
            border-color: #cfe7ff;
        }}
        .badge-anomalous {{
            background: linear-gradient(135deg, #ffe3f0 0%, #fff0f5 100%);
            color: #b81a4c;
            border-color: #ffd1e6;
        }}
        .badge-yes {{
            background: linear-gradient(135deg, #e7ffe9 0%, #f2fff4 100%);
            color: #0e7a2f;
            border-color: #c9f1d2;
        }}
        .badge-no {{
            background: #f5f7fb;
            color: #6b778c;
            border-color: #e1e7f0;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 8px;
        }}
        .info-item {{
            background: #ffffff;
            border: 1px solid #efe6b2;
            border-radius: 10px;
            padding: 10px 12px;
            box-shadow: 0 1px 6px rgba(138, 109, 0, 0.06);
        }}
        .info-item span {{
            display: block;
            font-size: 0.78rem;
            color: #8a6d00;
            opacity: 0.8;
        }}
        .info-item strong {{
            font-size: 1.05rem;
            color: #4a3b00;
        }}
        .image-preview {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 24px;
        }}
        #preview-image {{
            max-width: 100%;
            max-height: 600px;
            width: auto;
            height: auto;
            border-radius: 12px;
            display: none;
            box-shadow: 0 4px 24px 0 rgba(31, 38, 135, 0.18);
            transition: box-shadow 0.2s;
            margin: 0 auto;
        }}
        #preview-image:hover {{
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.22);
        }}
        .selected-images {{
            margin-top: 24px;
            max-height: 400px;
            overflow-y: auto;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: flex-start;
        }}
        .selected-image {{
            margin: 0;
            border: 1.5px solid #e3e8f0;
            border-radius: 7px;
            padding: 7px 7px 4px 7px;
            background: #fff;
            box-shadow: 0 1px 4px 0 rgba(31, 38, 135, 0.07);
            transition: box-shadow 0.2s, border 0.2s;
        }}
        .selected-image:hover {{
            box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.13);
            border: 1.5px solid #a0b8d8;
        }}
        .selected-image img {{
            max-width: 110px;
            max-height: 110px;
            border-radius: 5px;
            margin-bottom: 4px;
        }}
        .controls {{
            margin: 24px 0 0 0;
            text-align: center;
        }}
        .btn {{
            background: linear-gradient(90deg, #36d1c4 0%, #5b86e5 100%);
            color: white;
            padding: 11px 28px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1.08rem;
            font-weight: 600;
            margin: 0 8px;
            box-shadow: 0 2px 8px 0 rgba(91, 134, 229, 0.10);
            transition: background 0.2s, box-shadow 0.2s;
        }}
        .btn:hover {{
            background: linear-gradient(90deg, #5b86e5 0%, #36d1c4 100%);
            box-shadow: 0 4px 16px 0 rgba(91, 134, 229, 0.18);
        }}
        .legend {{
            display: flex;
            justify-content: center;
            margin: 18px 0 10px 0;
            flex-wrap: wrap;
            gap: 18px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 1.05rem;
            color: #4a6073;
        }}
        .legend-color {{
            width: 18px;
            height: 18px;
            margin-right: 7px;
            border-radius: 50%;
            border: 2.5px solid #e3e8f0;
            box-shadow: 0 1px 4px 0 rgba(31, 38, 135, 0.07);
        }}
        .normal-color {{ background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); }}
        .anomalous-color {{ background: linear-gradient(135deg, #f857a6 0%, #ff5858 100%); }}
        .boundary-color {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>边界分析可视化 · 交互式</h1>
            <div class="sub">点击或圈选点查看对应的手部原始图像</div>
            <div class="pills">
                <span class="pill" title="邻居数 K">K = {K_NEIGHBORS}</span>
                <span class="pill" title="阈值">Threshold = 0.7</span>
            </div>
        </div>
        <div class="metrics">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-icon">B</div>
                    <div class="metric-content">
                        <strong>边界点数量</strong>
                        <span>{metrics['boundary_count']}</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">W</div>
                    <div class="metric-content">
                        <strong>Wasserstein 距离</strong>
                        <span>{metrics['anomalous_wasserstein']:.6f}</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">N</div>
                    <div class="metric-content">
                        <strong>正常样本</strong>
                        <span>{np.sum(labels == 0)}</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">A</div>
                    <div class="metric-content">
                        <strong>异常样本</strong>
                        <span>{np.sum(labels == 1)}</span>
                    </div>
                </div>
            </div>
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
                        <p id="image-caption">选择点后显示对应图像</p>
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
    <script type="application/json" id="points-data">
{json.dumps(points_data, ensure_ascii=False)}
    </script>
    <script type="application/json" id="boundary-indices">
{json.dumps(boundary_indices.tolist())}
    </script>
    <script>
        const IMAGE_API_URL = "{IMAGE_API_URL}";
        const pointsData = JSON.parse(document.getElementById('points-data').textContent);
        const boundaryIndices = JSON.parse(document.getElementById('boundary-indices').textContent);
        let plot = null;
        let currentPointIndex = null;
        let selectedPoints = [];
        let currentDragMode = 'zoom'; // 默认缩放模式

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
                title: '边界分析可视化 (K={K_NEIGHBORS}, Threshold=0.7)',
                xaxis: {{ title: '成分 1 (归一化)' }},
                yaxis: {{ title: '成分 2 (归一化)' }},
                hovermode: 'closest',
                showlegend: false,
                height: 600,
                plot_bgcolor: '#fafafa',
                paper_bgcolor: '#ffffff',
                dragmode: 'zoom'  // 默认缩放模式
            }};
            const config = {{ responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToAdd: ['toggleHover', 'resetViews'], scrollZoom: true }};
            plot = Plotly.newPlot('plotly-chart', [traceNormal, traceAnomalous, traceBoundary], layout, config);

            document.getElementById('plotly-chart').on('plotly_click', function(data) {{
                // 在框选/套索模式下禁用单点预览
                if (currentDragMode === 'lasso' || currentDragMode === 'select') {{
                    return;
                }}
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
                    // 选择时隐藏单图预览
                    const imgEl = document.getElementById('preview-image');
                    const caption = document.getElementById('image-caption');
                    if (imgEl) {{ imgEl.style.display = 'none'; imgEl.src = ''; }}
                    if (caption) {{ caption.textContent = `已圈选 ${{selectedPoints.length}} 个点，单图预览已隐藏`; }}
                }}
            }});
            document.getElementById('plotly-chart').on('plotly_deselect', function() {{
                selectedPoints = [];
                document.getElementById('selected-images-content').innerHTML = '<p>圈选多个点后显示所有图像</p>';
                // 取消选择时也重置预览
                const imgEl = document.getElementById('preview-image');
                const caption = document.getElementById('image-caption');
                if (imgEl) {{ imgEl.style.display = 'none'; imgEl.src = ''; }}
                if (caption) {{ caption.textContent = '选择点后显示对应图像'; }}
                // 恢复所有点的高亮（selectedpoints=null），并刷新图表
                Plotly.restyle('plotly-chart', {{'selectedpoints': null}});
                Plotly.redraw('plotly-chart');
                // 清除图上的框选叠加层与可能残留的形状
                Plotly.relayout('plotly-chart', {{'selections': [], 'shapes': []}});
            }});

            // 监听工具切换（拖拽模式变化），当退出框选/套索时清空框选
            document.getElementById('plotly-chart').on('plotly_relayout', function(eventData) {{
                if (eventData && eventData['dragmode']) {{
                    currentDragMode = eventData['dragmode'];
                    if (currentDragMode !== 'lasso' && currentDragMode !== 'select') {{
                        clearSelectionUI();
                        // 立即恢复全图高光
                        Plotly.restyle('plotly-chart', {{'selectedpoints': null}});
                        Plotly.redraw('plotly-chart');
                    }}
                }}
            }});
        }}

        function showPointInfo(pointIndex) {{
            // 强制类型一致
            const point = pointsData.find(p => Number(p.index) === Number(pointIndex));
            if (!point) return;

            currentPointIndex = pointIndex;

            const typeBadgeClass = point.cluster === 'normal' ? 'badge-normal' : 'badge-anomalous';
            const boundaryBadgeClass = point.is_boundary ? 'badge-yes' : 'badge-no';
            const typeText = point.cluster === 'normal' ? '正常' : '异常';
            const boundaryText = point.is_boundary ? '边界点: 是' : '边界点: 否';

            document.getElementById('point-info-content').innerHTML = `
                <div class="info-header">
                    <span class="chip" title="唯一索引"><span class="dot"></span> ID #${{point.index}}</span>
                    <span class="badge ${{typeBadgeClass}}" title="样本类型">类型: ${{typeText}}</span>
                    <span class="badge ${{boundaryBadgeClass}}" title="是否属于边界">${{boundaryText}}</span>
                </div>
                <div class="info-grid">
                    <div class="info-item"><span>坐标 X</span><strong>${{point.x.toFixed(5)}}</strong></div>
                    <div class="info-item"><span>坐标 Y</span><strong>${{point.y.toFixed(5)}}</strong></div>
                </div>
            `;

            const imgElement = document.getElementById('preview-image');
            const caption = document.getElementById('image-caption');
            const imgUrl = IMAGE_API_URL + point.index + `?t=${{Date.now()}}`; // 避免缓存

            // 先设置占位文本并显示图片占位
            caption.textContent = '正在加载图像…';
            imgElement.style.display = 'block';

            // 绑定事件（每次重设，避免堆叠）
            imgElement.onload = function() {{
                caption.textContent = '手部图像预览';
                imgElement.style.display = 'block';
            }};
            imgElement.onerror = function() {{
                caption.textContent = '等待选择';
                imgElement.style.display = 'none';
            }};

            imgElement.src = imgUrl;
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
                    const imgUrl = IMAGE_API_URL + point.index + `?t=${{Date.now()}}`;
                    imgDiv.innerHTML = `
                        <div><strong>索引 ${{point.index}}</strong></div>
                        <img src="${{imgUrl}}" alt="手部图像 ${{point.index}}" style="max-width: 100px; max-height: 100px; cursor:pointer;" onerror="this.style.display='none'">
                        <div>类型: ${{point.cluster === 'normal' ? '正常' : '异常'}}</div>
                        <div>边界点: ${{point.is_boundary ? '是' : '否'}}</div>
                    `;
                    // 点击缩略图放大到右侧大图预览
                    imgDiv.querySelector('img').onclick = function() {{
                        showPointInfo(point.index);
                    }};
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
                'yaxis.range': [0, 1],
                'selections': [],
                'shapes': []
            }});
            selectedPoints = [];
            document.getElementById('point-info-content').innerHTML = '点击图表中的点查看详细信息';
            const imgEl = document.getElementById('preview-image');
            const caption = document.getElementById('image-caption');
            if (imgEl) {{ imgEl.style.display = 'none'; imgEl.src = ''; }}
            if (caption) {{ caption.textContent = '选择点后显示对应图像'; }}
            document.getElementById('selected-images-content').innerHTML = '<p>圈选多个点后显示所有图像</p>';
            Plotly.restyle('plotly-chart', {{'selectedpoints': null}});
        }}

        function clearSelectionUI() {{
            selectedPoints = [];
            document.getElementById('selected-images-content').innerHTML = '<p>圈选多个点后显示所有图像</p>';
            const imgEl = document.getElementById('preview-image');
            const caption = document.getElementById('image-caption');
            if (imgEl) {{ imgEl.style.display = 'none'; imgEl.src = ''; }}
            if (caption) {{ caption.textContent = '选择点后显示对应图像'; }}
            Plotly.restyle('plotly-chart', {{'selectedpoints': null}});
            // 同步清除图上的框选/套索轮廓和可能的形状
            Plotly.relayout('plotly-chart', {{'selections': [], 'shapes': []}});
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
    # 不弹出窗口
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