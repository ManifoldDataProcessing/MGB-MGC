import os
os.environ["OMP_NUM_THREADS"] = "1"
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import scipy.io as sio
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

try:
    from GBCreate2 import  get_gb_division_to_indices
except ImportError:
    print("wrong: 'GbCreate'")

def calculate_center_and_radius(gb_coords):
    num_points = gb_coords.shape[0]
    if num_points == 0: return None, None
    if num_points == 1: return gb_coords[0], 0.0
    center = gb_coords.mean(axis=0)
    dim = gb_coords.shape[1]
    radius = 0.0
    if num_points > dim:
        try:
            cov_matrix = np.cov(gb_coords.T);
            inv_cov = np.linalg.pinv(cov_matrix, rcond=1e-10)
            delta = gb_coords - center;
            mahal_sq = np.einsum('ij,jk,ik->i', delta, inv_cov, delta)
            mahal_sq[mahal_sq < 0] = 0;
            distances = np.sqrt(mahal_sq)
            radius = np.percentile(distances, 95) if len(distances) > 0 else 0.0
        except (np.linalg.LinAlgError, ValueError):
            distances_sq = np.sum((gb_coords - center) ** 2, axis=1)
            radius = np.sqrt(np.max(distances_sq)) if len(distances_sq) > 0 else 0.0
    else:
        distances_sq = np.sum((gb_coords - center) ** 2, axis=1)
        radius = np.sqrt(np.max(distances_sq)) if len(distances_sq) > 0 else 0.0
    return center, max(0.0, radius)


def get_ball_quality(gb_coords, center):
    N = gb_coords.shape[0];
    ball_quality = N;
    mean_r = 0.0
    if N > 1 and center is not None:
        try:
            mean_r = np.mean(np.linalg.norm(gb_coords - center, axis=1))
        except TypeError:
            mean_r = 0.0
    return ball_quality, mean_r


def ball_density(radiusAD, ball_qualitysA, dim, min_radius_threshold=1e-4):
    N = radiusAD.shape[0];
    ball_dens = np.zeros(shape=N);
    epsilon = 1e-12
    valid_indices = (radiusAD >= min_radius_threshold) & (ball_qualitysA > 0)
    if np.any(valid_indices):
        radii_valid = radiusAD[valid_indices];
        quality_valid = ball_qualitysA[valid_indices]
        volume_valid = np.power(radii_valid, dim)
        ball_dens[valid_indices] = quality_valid / np.maximum(volume_valid, epsilon)
    return ball_dens


def ball_min_dist(ball_distS, ball_densS):
    N3 = ball_distS.shape[0]
    if N3 == 0: return np.array([]), np.array([])
    np.fill_diagonal(ball_distS, 0)
    ball_min_distAD = np.full(N3, np.inf);
    ball_nearestAD = np.full(N3, -1, dtype=int)
    index_ball_dens = np.argsort(-ball_densS)
    if N3 > 0:
        highest_idx = index_ball_dens[0]
        for i3 in range(1, N3):
            current_idx = index_ball_dens[i3];
            higher_dens_indices = index_ball_dens[:i3]
            if len(higher_dens_indices) > 0:
                distances_to_higher = ball_distS[current_idx, higher_dens_indices]
                min_dist_local_idx = np.argmin(distances_to_higher)
                ball_min_distAD[current_idx] = distances_to_higher[min_dist_local_idx]
                ball_nearestAD[current_idx] = higher_dens_indices[min_dist_local_idx]
        finite_deltas = ball_min_distAD[np.isfinite(ball_min_distAD)]
        if len(finite_deltas) > 0:
            ball_min_distAD[highest_idx] = np.max(finite_deltas)
        elif N3 > 1:
            other_indices = np.delete(np.arange(N3), highest_idx)
            if len(other_indices) > 0:
                ball_min_distAD[highest_idx] = np.max(ball_distS[highest_idx, other_indices])
            else:
                ball_min_distAD[highest_idx] = 0.0
        else:
            ball_min_distAD[highest_idx] = 0.0
        ball_nearestAD[highest_idx] = highest_idx
    return ball_min_distAD, ball_nearestAD.astype(int)


def ball_find_centers_auto_ultra_simplified(ball_densS, ball_min_distS, gamma_top_percentage=10):
    N_balls = len(ball_densS)
    if N_balls == 0: return np.array([])
    top_p = np.clip(gamma_top_percentage, 0.01, 100.0);
    gamma = ball_densS * ball_min_distS
    valid_indices_mask = np.isfinite(gamma);
    gamma_valid = gamma[valid_indices_mask]
    original_indices_of_valid = np.arange(N_balls)[valid_indices_mask]
    if len(gamma_valid) == 0: return np.array([0]) if N_balls > 0 else np.array([])
    sorted_indices_local = np.argsort(-gamma_valid)
    num_centers_to_select = max(1, math.ceil(N_balls * (top_p / 100.0)))
    num_centers_to_select = min(num_centers_to_select, len(gamma_valid))
    selected_local_indices = sorted_indices_local[:num_centers_to_select]
    centers_indices = original_indices_of_valid[selected_local_indices]
    print(f"自动选择 gamma 值最高的 {top_p}% (约 {num_centers_to_select} 个) 作为中心。")
    print(f"自动找到 {len(centers_indices)} 个聚类中心: {centers_indices}")
    return centers_indices.astype(int)


def ball_cluster(ball_densS, ball_centers, ball_nearest):
    K1 = len(ball_centers);
    N5 = ball_densS.shape[0]
    if K1 == 0: return -1 * np.ones(N5, dtype=int)
    ball_labs = -1 * np.ones(N5, dtype=int)
    for i5, cen1 in enumerate(ball_centers):
        if cen1 < N5: ball_labs[cen1] = int(i5)
    ball_index_density = np.argsort(-ball_densS)
    for i5, index2 in enumerate(ball_index_density):
        if ball_labs[index2] == -1:
            nearest_higher_idx = int(ball_nearest[index2])
            if nearest_higher_idx != -1 and nearest_higher_idx < N5 and ball_labs[nearest_higher_idx] != -1:
                ball_labs[index2] = ball_labs[nearest_higher_idx]
    return ball_labs


def drawcluster2(A, cluster, ncluster):
    n, d = A.shape
   
    plt.figure(figsize=(10, 8))  
    plt.xlabel('特征 1')  
    plt.ylabel('特征 2')  

    cmap = plt.cm.get_cmap('viridis', 64)

    for i in range(n):
        if cluster[i] > 0:
            ic = int(((cluster[i]) * 64) / (ncluster * 1.0)) - 1
            if ic < 0: ic = 0
            if ic > 63: ic = 63

            color = cmap(ic / 63.0)

            x = A[i, 0]
            y = A[i, 1]
            plt.plot(x, y, 'o', markersize=5, markerfacecolor=color, markeredgecolor=color)
            
        else:
            x = A[i, 0]
            y = A[i, 1]
            plt.plot(x, y, 'k.')
            

    plt.grid(True)  
    plt.show()  
def evaluate_clustering_metrics(original_data_with_labels, gb_indices_list, final_ball_labs):
    """
    计算聚类结果的 NMI, ARI 和 ACC 指标。

    参数:
    original_data_with_labels (np.ndarray): 包含特征和真实标签的原始数据。
                                            假设最后一列是真实标签。
    gb_indices_list (list): 存储每个粒球包含的原始数据点索引的列表。
    final_ball_labs (np.ndarray): 每个粒球的最终聚类标签数组。

    返回:
    dict: 包含 'nmi', 'ari', 'acc' 的字典。
    """
    print("\n--- 阶段 E: 聚类评估指标计算 ---")

    # 1. 提取真实标签
    true_labels_original = original_data_with_labels[:, -1].astype(int)

    # 2. 将粒球标签映射回原始数据点的标签
    mapped_predicted_labels = -1 * np.ones(original_data_with_labels.shape[0], dtype=int)
    for ball_idx, cluster_id in enumerate(final_ball_labs):
        if ball_idx < len(gb_indices_list):  # 确保索引有效
            gb_indices = gb_indices_list[ball_idx]
            if len(gb_indices) > 0:  # 确保粒球非空
                # 确保gb_indices中的索引在mapped_predicted_labels的有效范围内
                valid_gb_indices = gb_indices[gb_indices < mapped_predicted_labels.shape[0]]
                mapped_predicted_labels[valid_gb_indices] = cluster_id

    # 3. 过滤噪声点（-1），只对非噪声点进行评估
    non_noise_mask = (mapped_predicted_labels != -1)

    if not np.any(non_noise_mask):
        print("警告: 没有检测到任何非噪声点进行评估。所有指标将返回 NaN。")
        return {'nmi': np.nan, 'ari': np.nan, 'acc': np.nan}

    true_labels_filtered = true_labels_original[non_noise_mask]
    predicted_labels_filtered = mapped_predicted_labels[non_noise_mask]

    # 4. 确保真实标签和预测标签的类别是连续的 (0 到 K-1)
    # 真实标签重映射
    unique_true_labels = np.unique(true_labels_filtered)
    true_label_map = {old_label: new_label for new_label, old_label in enumerate(unique_true_labels)}
    true_labels_remapped = np.array([true_label_map[label] for label in true_labels_filtered])

    # 预测标签重映射
    unique_pred_labels = np.unique(predicted_labels_filtered)
    pred_label_map = {old_label: new_label for new_label, old_label in enumerate(unique_pred_labels)}
    predicted_labels_remapped = np.array([pred_label_map[label] for label in predicted_labels_filtered])

    # --- 计算 NMI (Normalized Mutual Information) ---
    nmi_score = normalized_mutual_info_score(true_labels_remapped, predicted_labels_remapped)

    # --- 计算 ARI (Adjusted Rand Index) ---
    ari_score = adjusted_rand_score(true_labels_remapped, predicted_labels_remapped)

    # --- 计算 ACC (Accuracy) ---
    def _calculate_acc_internal(y_true, y_pred):
        # 这里的 y_true 和 y_pred 已经是重映射过的连续整数标签
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)

        n_clusters_true = len(np.unique(y_true))
        n_clusters_pred = len(np.unique(y_pred))

        # 创建一个代价矩阵，cost[i,j] 是真实簇 i 和预测簇 j 中样本的负计数
        cost_matrix = np.zeros((n_clusters_true, n_clusters_pred), dtype=int)
        for i in range(len(y_true)):
            cost_matrix[y_true[i], y_pred[i]] -= 1

        # 使用匈牙利算法找到最佳匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 计算准确率
        acc = -cost_matrix[row_ind, col_ind].sum() / len(y_true)
        return acc

    acc_score = _calculate_acc_internal(true_labels_remapped, predicted_labels_remapped)

    print(f"  - 非噪声点数量: {len(true_labels_filtered)} / {original_data_with_labels.shape[0]}")
    print(f"  - NMI (归一化互信息): {nmi_score:.4f}")
    print(f"  - ARI (调整兰德指数): {ari_score:.4f}")
    print(f"  - ACC (准确率): {acc_score:.4f}")

    return {'nmi': nmi_score, 'ari': ari_score, 'acc': acc_score}


def ball_draw_cluster(original_data, gb_indices_list, ball_labs, dic_colors, ball_centers, centersA,
                      show_initial_centers=False):
    """
    【最终健壮版】可视化聚类结果。
    - 解决了因簇标签不连续导致的颜色显示错误问题。
    - 确保每个独立的簇都有一个独一无二的颜色。
    """
    plt.figure(figsize=(10, 8))
    plt.axis('equal')

    # --- 【核心修改 1】: 创建一个健壮的颜色映射 ---
    # 1. 找出所有唯一的、非噪声的簇标签
    unique_labels = np.unique(ball_labs[ball_labs != -1])
    num_clusters = len(unique_labels)
    print(f"准备绘制 {num_clusters} 个簇的结果...")

    # 2. 创建一个从“原始标签”到“连续颜色索引”的映射
    #    例如: 如果 unique_labels 是 [0, 70]，这个映射就是 {0: 0, 70: 1}
    label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}

    # --- 循环绘图 ---
    plotted_labels = set()
    num_balls_to_plot = min(len(gb_indices_list), len(ball_labs))

    for i in range(num_balls_to_plot):
        gb_indices = gb_indices_list[i]
        if len(gb_indices) == 0: continue

        try:
            gb_data_coords = original_data[gb_indices]
        except IndexError:
            continue

        current_label = ball_labs[i]

        # --- 【核心修改 2】: 使用新的映射来获取颜色 ---
        if current_label == -1:
            # 噪声点使用固定的灰色
            color = (0.5, 0.5, 0.5)
        else:
            # 根据映射获取连续的颜色索引
            color_idx = label_to_color_idx[current_label]
            # 使用取余数的方式从颜色字典中安全地获取颜色
            color = dic_colors.get(color_idx % len(dic_colors), '#1f77b4')

        label_text = f'簇 {int(current_label)}' if current_label != -1 else '噪声点'

        # 为了生成正确的图例，我们只在第一次遇到某个标签时添加 label 参数
        label_to_use_for_legend = label_text if current_label not in plotted_labels else None

        plt.scatter(gb_data_coords[:, 0], gb_data_coords[:, 1], s=15., color=color, alpha=0.6,
                    label=label_to_use_for_legend)

        # if label_to_use_for_legend:
        #     plotted_labels.add(current_label)

    # --- 绘制DPC中心的部分 (无修改) ---
    if show_initial_centers:
        print("正在尝试绘制DPC初始中心（黄星星）...")
        if len(ball_centers) > 0 and centersA is not None and len(centersA) > 0 and np.max(ball_centers) < len(
                centersA):
            try:
                center_coords_to_plot = centersA[ball_centers]
                plt.scatter(center_coords_to_plot[:, 0], center_coords_to_plot[:, 1],
                            s=200, c='yellow', marker='*', edgecolors='black', label='DPC初始中心',
                            zorder=10)
                if 'DPC初始中心' not in plotted_labels:
                    plotted_labels.add('DPC初始中心')
            except IndexError:
                print("警告：因索引错误，未能绘制DPC初始中心。")
        else:
            print("信息：因未找到有效中心或安全检查未通过，不绘制DPC初始中心。")

    # if len(plotted_labels) > 0:
    #     plt.legend(loc='best', markerscale=2)

    # plt.title('聚类结果 (HCBNR + Ward Linkage 合并)')
    plt.grid(True)

def build_knn_graph_manual_kdtree(data, k, mode='connectivity'):
    """
    【可复用优化版】使用 scipy.KDTree 手动构建 k-NN 图。
    """
    print(f"  > [高效版] 正在使用手动KDTree构建 {k}-NN 图 (模式: {mode})...")
    n_samples = data.shape[0]
    kdtree = KDTree(data)
    distances, indices = kdtree.query(data, k=k + 1)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_samples))
    if mode == 'connectivity':
        for i in range(n_samples):
            for neighbor_j in indices[i, 1:]:
                graph.add_edge(i, neighbor_j)
    elif mode == 'distance':
        for i in range(n_samples):
            for j in range(1, k + 1):
                neighbor_idx = indices[i, j]
                dist = distances[i, j]
                graph.add_edge(i, neighbor_idx, distance=dist)
    return graph

# ========================================================================================
# --- 【速度优化版】合并函数 ---
# ========================================================================================
def calculate_ward_linkage_cost(points_A, points_B):
    """
    计算合并两个点集 A 和 B 后的 Ward's Linkage 代价（方差增量）。
    代价越小，说明合并越合理。
    """
    n_A = len(points_A)
    n_B = len(points_B)

    if n_A == 0 or n_B == 0:
        return np.inf

    # 计算各自的中心
    center_A = np.mean(points_A, axis=0)
    center_B = np.mean(points_B, axis=0)

    # 沃德连接距离的平方 等价于 (n_A * n_B / (n_A + n_B)) * ||center_A - center_B||^2
    # 我们直接计算这个代价，代价越小越好。
    distance_sq = np.sum((center_A - center_B) ** 2)
    cost = (n_A * n_B) / (n_A + n_B) * distance_sq

    return cost


def merge_clusters_by_hcbnr_similarity_optimized(
        initial_ball_labs,
        gb_indices_list,
        original_data,
        precomputed_graph,
        target_num_clusters
):
    """
    【最终修正版】
    - 接收一个带 'distance' 属性的预计算图。
    - 在函数内部，将 'distance' 转换为 'weight'，确保后续逻辑正确。
    - 后续所有合并逻辑不变。
    """
    print("\n--- 阶段四：在预计算好的图上进行合并 ---")
    current_labs = np.copy(initial_ball_labs)

    # --- 直接使用传入的图 ---
    G = precomputed_graph

    # --- 【关键修正】: 在这里将 'distance' 转换为 'weight' ---
    print("  > 正在将图的 'distance' 属性转换为 'weight' 属性...")
    epsilon = 1e-6
    # 我们需要创建一个新的图或者修改当前图的边属性
    # 为了不修改原始的 master_graph，我们创建一个带权重的副本
    G_with_weight = nx.Graph()
    for u, v, data in G.edges(data=True):
        distance = data.get('distance', 1.0)  # 如果没有distance属性，默认距离为1
        weight = 1.0 / (distance + epsilon)
        G_with_weight.add_edge(u, v, weight=weight)

    print("  > 带权图准备完成。")

    initial_cluster_ids = np.unique(initial_ball_labs[initial_ball_labs != -1])
    num_initial_clusters = len(initial_cluster_ids)
    if num_initial_clusters <= target_num_clusters:
        return current_labs

    print(f"  > 正在为 {original_data.shape[0]} 个点构建 '点 -> 簇' 的快速查找映射...")
    point_to_cluster_id_map = -1 * np.ones(original_data.shape[0], dtype=np.int32)
    for ball_idx, cid in enumerate(initial_ball_labs):
        if cid != -1:
            points_in_ball = gb_indices_list[ball_idx]
            point_to_cluster_id_map[points_in_ball] = cid
    print("  > 查找映射构建完成。")

    print(f"  > 正在预计算 {num_initial_clusters} 个初始簇之间的关系 (高效版)...")
    cid_to_matrix_idx = {cid: i for i, cid in enumerate(initial_cluster_ids)}

    connectivity_matrix = np.zeros((num_initial_clusters, num_initial_clusters), dtype=np.int32)
    boundary_weight_matrix = np.zeros((num_initial_clusters, num_initial_clusters), dtype=np.float64)

    # 现在 G_with_weight 有了 'weight' 属性，这个循环可以正常工作了
    for u, v, data in G_with_weight.edges(data=True):
        weight = data['weight']  # 直接获取 weight
        cid_u = point_to_cluster_id_map[u]
        cid_v = point_to_cluster_id_map[v]

        if cid_u != -1 and cid_v != -1 and cid_u != cid_v:
            idx_u = cid_to_matrix_idx[cid_u]
            idx_v = cid_to_matrix_idx[cid_v]

            connectivity_matrix[idx_u, idx_v] += 1
            boundary_weight_matrix[idx_u, idx_v] += weight

    # ... (后续所有合并逻辑完全不变, 因为它们依赖的矩阵已经正确计算) ...
    connectivity_matrix += connectivity_matrix.T
    boundary_weight_matrix += boundary_weight_matrix.T

    with np.errstate(divide='ignore', invalid='ignore'):
        closeness_matrix = np.nan_to_num(boundary_weight_matrix / connectivity_matrix)
    similarity_matrix = connectivity_matrix * (closeness_matrix ** 2)
    np.fill_diagonal(similarity_matrix, -1)
    print("  > 预计算完成。")

    # --- 【核心优化 3】: 为Ward's Linkage后备策略准备数据 ---
    # 这部分逻辑与之前类似，但现在可以更高效地构建
    print("  > 正在为 Ward's Linkage 后备策略准备数据...")
    cluster_to_points_coords_map = {cid: [] for cid in initial_cluster_ids}
    all_points_indices = np.where(point_to_cluster_id_map != -1)[0]
    for point_idx in all_points_indices:
        cid = point_to_cluster_id_map[point_idx]
        cluster_to_points_coords_map[cid].append(original_data[point_idx])

    for cid in cluster_to_points_coords_map:
        if cluster_to_points_coords_map[cid]:
            cluster_to_points_coords_map[cid] = np.array(cluster_to_points_coords_map[cid])
        else:
            cluster_to_points_coords_map[cid] = np.array([]).reshape(0, original_data.shape[1])
    print("  > 数据准备完成。")

    # --- 步骤 4: 迭代合并 (这部分逻辑完全不变，因为它已经很快了) ---
    active_mask = np.ones(num_initial_clusters, dtype=bool)
    num_current_clusters = num_initial_clusters
    matrix_idx_to_cid = {i: cid for cid, i in cid_to_matrix_idx.items()}

    while num_current_clusters > target_num_clusters:
        active_indices = np.where(active_mask)[0]
        if len(active_indices) < 2: break

        sub_sim_matrix = similarity_matrix[np.ix_(active_indices, active_indices)]
        max_sim_in_active = np.max(sub_sim_matrix)

        if max_sim_in_active > 0:
            flat_sub_idx = np.argmax(sub_sim_matrix)
            sub_idx_A, sub_idx_B = np.unravel_index(flat_sub_idx, sub_sim_matrix.shape)
            idx_A = active_indices[sub_idx_A];
            idx_B = active_indices[sub_idx_B]
            print(
                f"  > 合并中 ({num_current_clusters - 1}/{target_num_clusters}): [主策略] 图相似度最高，Sim={max_sim_in_active:.4f}")
        else:
            print("  > 所有簇对图相似度为0，启动【终极后备策略】：Ward's Linkage (最小方差增量)...")
            ward_cost_matrix = np.full((len(active_indices), len(active_indices)), np.inf)
            for i in range(len(active_indices)):
                for j in range(i + 1, len(active_indices)):
                    cid_A = matrix_idx_to_cid[active_indices[i]]
                    cid_B = matrix_idx_to_cid[active_indices[j]]
                    cost = calculate_ward_linkage_cost(
                        cluster_to_points_coords_map[cid_A],
                        cluster_to_points_coords_map[cid_B]
                    )
                    ward_cost_matrix[i, j] = cost
            if np.all(np.isinf(ward_cost_matrix)): break
            flat_sub_idx = np.nanargmin(ward_cost_matrix)
            sub_idx_A, sub_idx_B = np.unravel_index(flat_sub_idx, ward_cost_matrix.shape)
            idx_A = active_indices[sub_idx_A];
            idx_B = active_indices[sub_idx_B]
            min_cost = ward_cost_matrix[sub_idx_A, sub_idx_B]
            print(
                f"  > 合并中 ({num_current_clusters - 1}/{target_num_clusters}): [后备策略] Ward's Linkage 代价最小，Cost={min_cost:.2f}")

        # --- 执行合并 (逻辑不变) ---
        cid_A = matrix_idx_to_cid[idx_A]
        cid_B = matrix_idx_to_cid[idx_B]
        id_to_keep, id_to_merge = (cid_A, cid_B)
        idx_to_update, idx_to_disable = (idx_A, idx_B)

        for idx_X in active_indices:
            if idx_X != idx_to_update and idx_X != idx_to_disable:
                new_conn = connectivity_matrix[idx_to_update, idx_X] + connectivity_matrix[idx_to_disable, idx_X]
                new_weight = boundary_weight_matrix[idx_to_update, idx_X] + boundary_weight_matrix[
                    idx_to_disable, idx_X]
                connectivity_matrix[idx_to_update, idx_X] = connectivity_matrix[idx_X, idx_to_update] = new_conn
                boundary_weight_matrix[idx_to_update, idx_X] = boundary_weight_matrix[idx_X, idx_to_update] = new_weight
                new_close = new_weight / new_conn if new_conn > 0 else 0
                new_sim = new_conn * (new_close ** 2)
                similarity_matrix[idx_to_update, idx_X] = similarity_matrix[idx_X, idx_to_update] = new_sim

        if len(cluster_to_points_coords_map[id_to_merge]) > 0:
            cluster_to_points_coords_map[id_to_keep] = np.vstack([
                cluster_to_points_coords_map[id_to_keep],
                cluster_to_points_coords_map[id_to_merge]
            ])
        cluster_to_points_coords_map.pop(id_to_merge, None)

        active_mask[idx_to_disable] = False
        current_labs[current_labs == id_to_merge] = id_to_keep
        num_current_clusters -= 1

    print("\n--- 合并流程结束 ---")
    return current_labs

def build_knn_graph_manual_kdtree(data, k, mode='connectivity'):
    """
    【可复用优化版】使用 scipy.KDTree 手动构建 k-NN 图。
    """
    print(f"  > [高效版] 正在使用手动KDTree构建 {k}-NN 图 (模式: {mode})...")
    n_samples = data.shape[0]
    kdtree = KDTree(data)
    distances, indices = kdtree.query(data, k=k + 1)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_samples))
    if mode == 'connectivity':
        for i in range(n_samples):
            for neighbor_j in indices[i, 1:]:
                graph.add_edge(i, neighbor_j)
    elif mode == 'distance':
        for i in range(n_samples):
            for j in range(1, k + 1):
                neighbor_idx = indices[i, j]
                dist = distances[i, j]
                graph.add_edge(i, neighbor_idx, distance=dist)
    return graph


def load_dataset(file_path):
    filename = os.path.basename(file_path)
    _, file_extension = os.path.splitext(filename)

    if file_extension == '.txt':
        return np.loadtxt(file_path)

    elif file_extension == '.mat':
        print(f"  > 正在加载 MAT 文件: {filename}")
        try:
            mat_contents = sio.loadmat(file_path)
        except Exception:
            with h5py.File(file_path, 'r') as f:
                mat_contents = {key: np.array(f[key]) for key in f.keys()}

        # 明确从 MAT 中读取 'fea' 作为特征，'gt' 作为标签
        if 'fea' in mat_contents and 'gt' in mat_contents:
            features = mat_contents['fea']
            labels = mat_contents['gt']
            # 确保标签是一维数组（去掉多余维度，如 (32768,1) → (32768,)）
            labels = labels.flatten()
            # 拼接特征和标签，形成「特征+标签」的矩阵
            original_data_with_labels = np.column_stack((features, labels))
            print(f"    - 成功读取 'fea' (形状: {features.shape}) 和 'gt' (形状: {labels.shape})")
            return original_data_with_labels
        else:
            raise ValueError(f"MAT 文件 '{filename}' 中缺少 'fea' 或 'gt' 变量！")

    else:
        raise ValueError(f"不支持的文件类型: '{file_extension}'请提供 .txt 或 .mat 文件。")
# ==========================================================
# 主执行块 (__main__) - 【彻底修正版】
# ==========================================================
if __name__ == "__main__":
    # --- 1. 参数配置 ---
    unified_k = 10
    gamma_top_percentage_for_centers = 15
    target_num_clusters = 4 # D3.txt 的真实簇数是 6
    dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0), 2: (0, 0, .8), 3: (.8, .8, 0),
                  4: (.8, 0, .8), 5: (0, .8, .8), 6: ('#1f77b4'), 7: ('#ff7f0e'),
                  8: ('#2ca02c'), 9: ('#d62728'), 10: ('#9467bd'), 11: ('#8c564b'),
                  12: ('#e377c2'), 13: ('#7f7f7f')}

    # --- 2. 指定数据文件 ---
    data_file_path = r'./Data/synthetic datasets/SF_3_4.mat'

    # --- 3. 核心算法执行流程 ---
    try:
        print("=" * 80)
        print(f"--- 开始处理: {os.path.basename(data_file_path)} ---")

        dataset_name = os.path.splitext(os.path.basename(data_file_path))[0]
        original_data_with_labels = load_dataset(data_file_path)
        original_data_features = original_data_with_labels[:, :-1]
        
        # 这个总计时器仅用于参考，我们不再用它来报告最终时间
        # total_start_time = time.time()
        
        timings = {} # 用于存储每个阶段的纯计算时间

        # --- 阶段 A, B, C (计时方式不变，非常精确) ---
        stage_start_time = time.time()
        master_graph_with_distance = build_knn_graph_manual_kdtree(original_data_features, k=unified_k, mode='distance')
        timings['A_Build_kNN_Graph'] = time.time() - stage_start_time

        stage_start_time = time.time()
        louvain_graph = nx.Graph(master_graph_with_distance.edges())
        gb_indices_list = get_gb_division_to_indices(louvain_graph)
        actual_gb_indices_list = [indices for indices in gb_indices_list if len(indices) > 0]
        timings['B_Granular_Ball_Generation'] = time.time() - stage_start_time

        stage_start_time = time.time()
        centers_list, radiuss_list, ball_qualitys_list = [], [], []
        for gb_indices in actual_gb_indices_list:
            gb_data_coords = original_data_features[gb_indices]
            center, radius = calculate_center_and_radius(gb_data_coords)
            if center is not None:
                ball_quality, _ = get_ball_quality(gb_data_coords, center)
                centers_list.append(center); radiuss_list.append(radius); ball_qualitys_list.append(ball_quality)
        centersA = np.array(centers_list); radiusA = np.array(radiuss_list); ball_qualitys_A = np.array(ball_qualitys_list)
        ball_densS = np.log1p(ball_density(radiusA, ball_qualitys_A, dim=original_data_features.shape[1]))
        ball_distS_euclidean = squareform(pdist(centersA, metric='euclidean'))
        ball_min_distS, ball_nearest = ball_min_dist(ball_distS_euclidean, ball_densS)
        ball_centers = ball_find_centers_auto_ultra_simplified(ball_densS, ball_min_distS, gamma_top_percentage=gamma_top_percentage_for_centers)
        initial_ball_labs = ball_cluster(ball_densS, ball_centers, ball_nearest)
        timings['C_DPC_Clustering'] = time.time() - stage_start_time
        
        # --- 中间调试绘图 (这部分不计时) ---
        #print("\n--- 正在生成【中间过程】的可视化图：DPC 初始聚类结果 (合并前) ---")
        #ball_draw_cluster(original_data_features, actual_gb_indices_list, initial_ball_labs,
        #                  dic_colors, ball_centers, centersA, show_initial_centers=True)
        #plt.title('DPC 初始聚类结果 (合并前)', fontsize=16)
        #plt.show() # 程序会在此暂停，等待您关闭窗口
        #print("--- 中间可视化图已关闭，程序继续执行。 ---")

        # --- 阶段 D: 层次化合并 ---
        stage_start_time = time.time()
        final_ball_labs = merge_clusters_by_hcbnr_similarity_optimized(
            initial_ball_labs, actual_gb_indices_list, original_data_features,
            master_graph_with_distance, target_num_clusters)
        timings['D_Hierarchical_Merging'] = time.time() - stage_start_time

        # --- 阶段 E: 评估 ---
        stage_start_time = time.time()
        evaluation_results = evaluate_clustering_metrics(original_data_with_labels, actual_gb_indices_list, final_ball_labs)
        timings['E_Clustering_Metrics'] = time.time() - stage_start_time

        # --- 【关键修正】: 总计算时间 = 各个阶段纯计算时间之和 ---
        total_computation_time = sum(timings.values())
        final_cluster_count = len(np.unique(final_ball_labs[final_ball_labs != -1]))

        # --- 【最终报告】 ---
        print("\n" + "=" * 80)
        print(f"--- 【处理完成】 ---")
        print(f"数据集: {os.path.basename(data_file_path)}")
        print(f"最终聚类簇数量: {final_cluster_count}")
        print("-" * 30)
        print("【聚类评估指标】:")
        print(f"  - NMI: {evaluation_results['nmi']:.4f}")
        print(f"  - ARI: {evaluation_results['ari']:.4f}")
        print(f"  - ACC: {evaluation_results['acc']:.4f}")
        print("-" * 30)
        print("【性能剖析报告】:")
        for stage, duration in timings.items():
            print(f"  - {stage:<30}: {duration:8.3f} 秒")
        print("-" * 30)
        # 使用修正后的总时间
        print(f"【总计算时间】: {total_computation_time:.3f} 秒")
        print("=" * 80 + "\n")

        # --- 最终绘图 (这部分不计时) ---
        print("正在生成【最终结果】可视化图像...")
        ball_draw_cluster(original_data_features, actual_gb_indices_list, final_ball_labs,
                          dic_colors, ball_centers, centersA, show_initial_centers=False)
        plt.title('最终聚类结果 (合并后)', fontsize=16)
        plt.show()

    except Exception as e:
        print(f"\n!!!!!! 发生严重错误: {e} !!!!!!\n")
        import traceback
        traceback.print_exc()