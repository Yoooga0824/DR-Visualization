# tool_functions.py
import numpy as np  # 添加这行导入
from detection.BDLLE import bd_lle
from geomloss import SamplesLoss

def latent_detection(target_data_edge, target_data_labels, d=2, K = 150, method='global'):
    # 参数设置
    d = 2  # 流形维度
    K = 50  # 近邻数
    
    # 执行BD-LLE算法
    if method=='global':
        boundary_points, B = bd_lle(target_data_edge, d, K)
        boundary_indices = np.where(boundary_points)[0]  # 这里使用了np
        boundary_points_labels = target_data_labels[boundary_indices]
    elif method=='local':
        unique_elements = np.unique(target_data_labels)  # 这里使用了np
        num_cls = len(unique_elements)
        
        ind_boundarys = []
        for label in range(num_cls):
            #Subset labels
            ind_label = np.where(target_data_labels == label)[0]  # 这里使用了np
            X_sub = target_data_edge[ind_label,:]
            boundary_points, B = bd_lle(X_sub, d, K)
            boundary_indices = np.where(boundary_points)[0]  # 这里使用了np
            boundary_points_labels = ind_label[boundary_indices]
            ind_boundarys.append(boundary_points_labels)
                
        boundary_indices = np.concatenate(ind_boundarys, axis=0)  # 这里使用了np
        boundary_points_labels = target_data_labels[boundary_indices]

    return boundary_indices, boundary_points_labels

def wasserstein_loss(z_source, z_target):
    """
    可微分的瓦瑟斯坦损失（支持梯度反向传播）
    z_source: 原空间边界样本的编码结果 f(A)，shape=(n, 2)
    z_target: 隐空间边界样本 B，shape=(m, 2)
    """
    # 定义瓦瑟斯坦损失计算器（Wasserstein-2距离，自动处理不同数量的样本）
    # 'sinkhorn'是高效的近似算法，适合作为损失函数
    loss_func = SamplesLoss(
        loss="sinkhorn",  # 基于Sinkhorn算法的近似计算
        p=2,              # 计算Wasserstein-2距离
        blur=0.001,        # 模糊系数（控制近似精度，越小越精确但计算量越大）
        scaling=0.9       # 缩放参数（加速收敛）
    )
    
    # 直接计算两个点集的瓦瑟斯坦距离（完全基于PyTorch Tensor，支持梯度传播）
    # 无需转换为NumPy，也无需detach()
    w_dist = loss_func(z_source, z_target)
    
    return w_dist