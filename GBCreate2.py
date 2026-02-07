import numpy as np
import igraph as ig
import leidenalg as la

def get_gb_division_to_indices(precomputed_graph, resolution=50.0):
    """
    使用 Leiden 算法进行粒球划分（无权重）
    兼容 leidenalg 0.11.0：使用 RBConfigurationVertexPartition（支持 resolution）
    """
    print(f"\n--- 使用 Leiden 粒球生成 (无权重, resolution={resolution}) ---")

    # Step 1：保存原始节点编号
    original_nodes = list(precomputed_graph.nodes())

    # Step 2：NetworkX → igraph
    g_igraph = ig.Graph.from_networkx(precomputed_graph)

    # Step 3：调用 Leiden（RBConfigurationVertexPartition）
    print("正在调用 Leiden ...")

    partition = la.find_partition(
        g_igraph,
        la.RBConfigurationVertexPartition,   # ← 唯一正确选择
        resolution_parameter=resolution,
        weights=None
    )

    print("Leiden 完成。")

    # Step 4：解析结果
    labels = np.array(partition.membership)
    num_communities = labels.max() + 1

    granules = [[] for _ in range(num_communities)]

    for ig_id, cid in enumerate(labels):
        real_id = original_nodes[ig_id]
        granules[cid].append(real_id)

    final_granules = [np.array(g, dtype=np.int32) for g in granules if len(g) > 0]

    print(f"粒球划分完成，共 {len(final_granules)} 个粒球。")
    return final_granules
