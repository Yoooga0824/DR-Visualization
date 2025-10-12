# Hand-DR-Project 使用说明（中文）

一个用于手部图像降维可视化与边界分析的小型工作流，包括：
- 生成 embedding 与原始图像的一一映射（mapping）
- 启动图片服务，供前端 HTML 以 index 取缩略图
- 生成可交互的 HTML（支持本地 file:// 直接打开）

当前已支持多种降维方法前缀（如 TSNE、NeuralTSNE、UMAP 等），每种方法可独立生成 mapping 和 HTML。

---

## 目录结构

```
Hand-DR-Project/
├─ code/
│  ├─ main.py                     # 批量入口（交互选择方法或指定 --prefix）
│  ├─ make_embedding_mapping.py   # 生成映射 JSON（embedding ↔ 图片）
│  ├─ boundary_analysis.py        # 生成交互式 HTML（含边界检测与指标）
│  ├─ image_server.py             # Flask 图片缩略图服务（/api/hand_thumb/<prefix>/<index>）
│  ├─ tool_functions.py           # 工具函数（含 wasserstein_loss）
│  └─ detection/BDLLE.py          # 边界检测算法（BD-LLE）
│
├─ data/
│  ├─ raw/
│  │  ├─ Hands/Hands/             # 正常样本图片目录（Hand_*.jpg）
│  │  └─ Anomalous_Hands/         # 异常样本图片目录（*.jpg）
│  └─ features/
│     ├─ <prefix>_embedding.npy               # 正常样本降维向量
│     ├─ anomalous_<prefix>_embedding.npy     # 异常样本降维向量
│     ├─ failed_images.txt                    # 正常样本默认过滤清单（可选）
│     ├─ failed_images_<prefix>.txt           # 正常样本按前缀过滤清单（可选）
│     ├─ anomalous_failed_images.txt          # 异常样本默认过滤清单（可选）
│     └─ anomalous_failed_images_<prefix>.txt # 异常样本按前缀过滤清单（可选）
│
└─ results/
   ├─ embedding_to_image_mapping_<prefix>.json # 每种方法的映射文件
   └─ <prefix>.html                           # 每种方法的交互式可视化页面
```

---

## 环境依赖

建议 Python 3.9+。需要安装的主要第三方库：
- numpy, matplotlib, scipy, tqdm
- torch（用于 wasserstein 距离计算）
- Pillow（PIL）
- Flask, flask-cors（图片服务）

Plotly 通过 CDN 加载，HTML 本地打开即可。

---

## 快速开始

1) 准备数据
- 将正常样本在 `data/raw/Hands/Hands/`，命名形如 `Hand_*.jpg`
- 将异常样本在 `data/raw/Anomalous_Hands/`
- 将降维结果放入 `data/features/`：
  - 正常：`<prefix>_embedding.npy`
  - 异常：`anomalous_<prefix>_embedding.npy`

2) 生成映射（mapping）
- 单方法：运行 `code/make_embedding_mapping.py`，示例：
  - `python code/make_embedding_mapping.py --prefix TSNE`
  - 可选：`--no-filter` 关闭正常与异常的 failed 过滤
  - 可选：`--anom-list D:\path\to\anom_list.txt` 精确指定异常图片集合与顺序（跳过异常 failed 过滤）

3) 启动图片服务（建议先启动）
- `python code/image_server.py`
- 服务默认监听 `0.0.0.0:5678`，接口：`/api/hand_thumb/<prefix>/<index>`

4) 生成交互式 HTML
- `python code/boundary_analysis.py --prefix TSNE`
- 生成 `results/TSNE.html`，可直接双击用浏览器打开（file:// 方式）

5) 打开页面
- 打开 `results/<prefix>.html` 即可交互查看、点击/圈选显示图像。

---

## 批量处理（推荐）

- 入口：`code/main.py`
- 支持交互式选择方法或通过 `--prefix` 只处理某一方法：
  - `python code/main.py`（交互式多选/全选）
  - `python code/main.py --prefix NeuralTSNE`（仅处理指定方法）
- 对每个选择的方法，依次执行：生成 mapping → 生成 HTML。

---

## 过滤与严格校验规则（重要）

1) 正常样本 failed 过滤
- 优先使用 `data/features/failed_images_<prefix>.txt`，否则回退 `failed_images.txt`
- 可用 `--no-filter` 全局关闭（正常与异常都不再过滤）

2) 异常样本 failed 过滤
- 优先使用 `data/features/anomalous_failed_images_<prefix>.txt`，否则回退 `anomalous_failed_images.txt`
- 若提供 `--anom-list`，则跳过异常侧的 failed 过滤（按清单精确选图与排序）

3) 严格一致性校验
- 正常侧：过滤后图片数量必须等于 `len(<prefix>_embedding.npy)`
- 异常侧：图片数量必须等于 `len(anomalous_<prefix>_embedding.npy)`
- 否则脚本直接抛错，避免 embedding/图片错位

---

## 映射与图片 API 说明

- 映射文件：`results/embedding_to_image_mapping_<prefix>.json`
  - 正常样本 index 从 0 开始，异常样本紧接其后，保证全局唯一连续
  - 每条记录包含：
    - `index`：全局索引
    - `embedding_coords`：对应降维向量
    - `image_path`：该样本图片绝对路径
    - `features`：预留字段

- 图片接口：`/api/hand_thumb/<prefix>/<index>`
  - 由 `code/image_server.py` 提供，会读取对应 `mapping_<prefix>.json`，根据 index 找到图片并返回一个 200x200 的 JPEG 缩略图

---

## HTML 图片显示与 IP 设置

- 生成的 HTML 中，图片接口 URL 为写死的绝对地址，示例：
  - `http://<你的机器IP>:5678/api/hand_thumb/<prefix>/`
- 请确保：
  1. `image_server.py` 已在该 IP:5678 上运行
  2. 防火墙允许 5678 端口访问（内网环境）
  3. 如果你的机器 IP 不是代码里写死的值，请修改 `code/boundary_analysis.py` 中的 `IMAGE_API_PATH`，或重新生成 HTML。

---

## 常见问题排查（FAQ）

1) HTML 打开后不显示图片
- 检查 `image_server.py` 是否在 5678 端口运行
- 检查 HTML 中写死的 IP 是否与你运行服务的机器一致
- 检查防火墙/端口是否通

2) 生成映射时报错“图片数与 embedding 数不一致”
- 正常侧：检查 `failed_images_<prefix>.txt` 或 `failed_images.txt` 是否多删/少删；必要时加 `--no-filter`
- 异常侧：建议提供 `--anom-list` 精确指定集合与顺序（或检查 `anomalous_failed_images_*` 清单）

3) 圈选/点击无响应或报错
- 直接重新生成该方法的 HTML（`boundary_analysis.py --prefix <prefix>`）
- 确保 `results/embedding_to_image_mapping_<prefix>.json` 与 HTML 的前缀一致且未被覆盖

---

## 进阶：参数与可调项

- `boundary_analysis.py`
  - `--prefix` 指定方法
  - `K_NEIGHBORS`（默认 100）：BD-LLE 的邻居数
  - 阈值固定 0.7×max(B)
  - 页面内显示边界点数量与 Wasserstein 距离指标

- `make_embedding_mapping.py`
  - `--prefix` 指定方法
  - `--no-filter` 关闭正常与异常过滤
  - `--anom-list` 指定异常图片清单（绝对路径或文件名，逐行一条）

- 新增一种降维方法的接入
  - 将 `<prefix>_embedding.npy` 与 `anomalous_<prefix>_embedding.npy` 放入 `data/features/`
  - 运行 `python code/main.py` 选择该方法即可

---

## 备注

- 本项目主要在 Windows 环境（路径使用绝对盘符）下开发与测试。
- 如需迁移到其他平台，请统一路径风格与服务 IP/端口配置。
