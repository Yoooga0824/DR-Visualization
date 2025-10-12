import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.linalg import eigh, solve

def bd_lle(X, d, K=20):
    """
    基于局部线性嵌入（LLE）的边界检测算法
    输入：
        X - 输入数据矩阵，n×p，n为样本数，p为维度
        d - 流形维度（已知）
        K - K近邻参数（默认20）
    输出：
        boundary_points - 边界点索引（布尔数组）
        B - 所有样本的边界指示器值
    """
    n, p = X.shape
    B = np.zeros(n)  # 边界指示器
    
    # 步骤1：计算每个样本的K近邻
    print('计算K近邻...')
    # 确保K不超过样本数量-1（因为要排除自身）
    K = min(K, n-1)  # 新增：防止K过大
    
    tree = cKDTree(X)
    # 查找K+1个近邻（包括自身），然后排除自身
    distances, idx = tree.query(X, k=K+1)
    
    # 新增：过滤掉超出范围的索引
    idx = np.clip(idx, 0, n-1)  # 确保索引在有效范围内
    
    idx = idx[:, 1:]  # 排除自身，保留K个近邻
    
    # 步骤2：计算重心坐标与局部协方差矩阵
    print('计算重心坐标与局部协方差矩阵...')
    G_list = []  # 存储局部数据矩阵G
    C_list = []  # 存储局部协方差矩阵C
    y_list = []  # 存储LLE中的y_k
    
    for i in range(n):
        # 局部数据矩阵G：p×K，每列是近邻点减中心点
        neighbors = X[idx[i, :], :]
        G = neighbors.T - np.outer(X[i, :], np.ones(K))
        G_list.append(G)
        
        # 计算局部协方差矩阵C = G*G'
        C = G @ G.T
        C_list.append(C)
        
        # 求解LLE中的y_k（带正则化）
        GtG = G.T @ G
        c = get_regularization_param(C, d)  # 自动计算正则化参数c
        y, _ = solve_lle(GtG, c, K)
        y_list.append(y)
    
    # 步骤3：计算边界指示器B
    print('计算边界指示器...')
    for i in range(n):
        y = y_list[i]
        # 边界指示器公式（论文核心公式）
        B[i] = (K - c * np.sum(y)) / K
    
    # 步骤4：确定边界点（阈值为最大值的1/2）
    maxB = np.max(B)
    threshold = maxB / 2
    boundary_points = B >= threshold
    
    print('BD-LLE算法执行完成')
    return boundary_points, B

def get_regularization_param(C, d):
    """计算正则化参数c"""
    # 基于局部协方差矩阵的特征值计算c（论文推荐策略）
    # 计算特征值（使用eigh，因为C是对称矩阵）
    lambda_vals = eigh(C, eigvals_only=True)
    lambda_vals = np.sort(lambda_vals)[::-1]  # 降序排列特征值
    
    if len(lambda_vals) < d + 1:
        return 1e-4  # 特征值不足时的默认值
    
    # c取第d个和第d+1个特征值乘积的平方根
    sum_lambda_d = lambda_vals[d-1]  # Python是0索引
    sum_lambda_d1 = lambda_vals[d]
    c = np.sqrt(sum_lambda_d * sum_lambda_d1)
    
    # 防止过小值导致数值不稳定
    return max(c, 1e-6)

def solve_lle(GtG, c, K):
    """求解LLE中的优化问题"""
    # 求解 (GtG + c*I)y = 1，然后归一化得到重心坐标w
    A = GtG + c * np.eye(K)
    b = np.ones(K)
    y = solve(A, b)  # 带正则化的解
    
    # 归一化得到重心坐标w
    w = y / np.sum(y)
    return y, w

# 主程序
if __name__ == "__main__":
    # 加载数据
    # 注意：这里假设您有latent_space_points.csv文件
    # 如果需要加载.mat文件，可以使用scipy.io.loadmat
    data = np.genfromtxt('/root/autodl-tmp/Hand-DR-Project/code/detection/normal_hand_embedding.csv', delimiter=',')
    target_data_edge = data/40
    print(data)
    # aaa
    
    # 参数设置
    d = 2  # 流形维度
    K = 1300  # 近邻数
    
    # 执行BD-LLE算法
    boundary_points, B = bd_lle(target_data_edge, d, K)
    
    # 可视化结果
    plt.figure(figsize=(10, 8))
    # 绘制所有点
    plt.scatter(target_data_edge[:, 0], target_data_edge[:, 1], 
                c='blue', s=10, alpha=0.6, label='所有点')
    # 绘制边界点
    boundary_indices = np.where(boundary_points)[0]
    plt.scatter(target_data_edge[boundary_indices, 0], 
                target_data_edge[boundary_indices, 1], 
                c='red', s=15, alpha=0.8, label='边界点')
    
    plt.title('BD-LLE边界检测结果')
    plt.xlabel('维度1')
    plt.ylabel('维度2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.savefig(r"detection_visualization.png", dpi=300, bbox_inches='tight')
