import numpy as np
import heapq
from typing import List, Tuple, Dict, Optional, Set
import random
import math
import time


# ─────────────────────────────────────────────
#  距离函数
# ─────────────────────────────────────────────

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 计算向量 a 和 b 的欧几里得距离
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 计算向量 a 和 b 的余弦向量相似度
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return float(1.0 - dot / norm)


DISTANCE_FUNCS = {
    'l2': euclidean_distance,
    'cosine': cosine_distance,
}


# ─────────────────────────────────────────────
#  核心数据结构
# ─────────────────────────────────────────────

class Node:
    """图中的一个节点，对应一条向量记录"""

    def __init__(self, node_id: int, vector: np.ndarray, max_layer: int):
        self.id = node_id
        # 这里向量直接存储到结点中。
        self.vector = vector
        # 每个节点在插入时随机分配的最高层
        self.max_layer = max_layer
        # neighbors[layer] = list of neighbor node_ids
        # 只分配该节点实际存在的层，不浪费空间
        self.neighbors: List[List[int]] = [[] for _ in range(max_layer + 1)]

    def __repr__(self):
        return f"Node(id={self.id}, max_layer={self.max_layer})"


# ─────────────────────────────────────────────
#  HNSW 主类
# ─────────────────────────────────────────────

class HNSW:
    """
    HNSW 近似最近邻索引

    参数:
        dim              向量维度
        space            距离空间: 'l2' | 'cosine'
        M                每层每个节点的最大邻居数（低层为 2*M）
        ef_construction  构建时的动态候选集大小（影响图质量）
        ef_search        查询时的动态候选集大小（影响召回率）
        seed             随机种子
    """

    def __init__(
            self,
            dim: int,
            space: str = 'l2',
            M: int = 16,
            ef_construction: int = 200,
            ef_search: int = 50,
            seed: int = 42,
    ):
        self.dim = dim
        self.M = M
        #  0 层存所有节点，是最密集的基础层，需要更多连接来保证在精细搜索阶段不遗漏近邻。
        self.M0 = 2 * M  # 第 0 层最大邻居数
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.dist_func = DISTANCE_FUNCS[space]

        # m_L 控制层级分配概率（来自论文）
        # 让节点分布在各层的期望数量呈指数衰减，M=16 时 m_L ≈ 0.36。M 越大，层数越少越集中。
        self.m_L = 1.0 / math.log(M)

        self.nodes: Dict[int, Node] = {}
        self.entry_point: Optional[int] = None  # 当前入口节点 id
        self.max_layer: int = -1  # 当前最高层

        random.seed(seed)
        np.random.seed(seed)

    # ──────────────────────────────────────────
    #  公开 API
    # ──────────────────────────────────────────

    def add(self, node_id: int, vector: np.ndarray) -> None:
        """插入一条向量"""
        vector = np.array(vector, dtype=np.float32)
        assert vector.shape == (self.dim,), f"期望维度 {self.dim}，实际 {vector.shape}"
        assert node_id not in self.nodes, f"node_id {node_id} 已存在"

        # 1. 随机决定该节点的最高层
        node_layer = self._random_layer()
        node = Node(node_id, vector, node_layer)
        self.nodes[node_id] = node

        # 2. 空图特殊处理
        # 第一个节点，直接设为入口点，无需建边
        if self.entry_point is None:
            # 把这个节点设为全局入口点
            self.entry_point = node_id
            # 更新图的最高层记录
            self.max_layer = node_layer
            # 直接返回（因为没有其他节点可以建边）
            return

        # 3. 初始化入口结点，从顶层入口点开始向下搜索
        curr_ep = [self.entry_point]

        # 高于 node_layer 的层：贪心下降
        for layer in range(self.max_layer, node_layer, -1):
            # 获取当前层离 query 最近 ef 个 node_id
            results = self._search_layer(vector, curr_ep, ef=1, layer=layer)
            # 把这个最近节点作为下一层的入口点，丢掉距离信息只保留 id
            curr_ep = [nid for _, nid in results]

        # 3b. node_layer 到 0 层：搜索 + 建边
        for layer in range(min(node_layer, self.max_layer), -1, -1):
            # 在当前层搜索 ef_construction 个候选邻居
            candidates = self._search_layer(vector, curr_ep, self.ef_construction, layer)

            # 启发式选出 M 个邻居（第 0 层用 M0）
            max_conn = self.M0 if layer == 0 else self.M
            neighbors = self._select_neighbors_heuristic(vector, candidates, max_conn, layer, extend_candidates=False)

            # 建立双向连接
            node.neighbors[layer] = [nid for _, nid in neighbors]
            for _, nid in neighbors:
                self.nodes[nid].neighbors[layer].append(node_id)
                # 如果邻居超出连接上限，裁剪
                if len(self.nodes[nid].neighbors[layer]) > max_conn:
                    self._prune_neighbors(nid, layer, max_conn)

            # 本层结果作为下层入口
            curr_ep = [nid for _, nid in candidates]

        # 4. 更新全局入口点
        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entry_point = node_id

    def search(
            self, query: np.ndarray, k: int = 10, ef: Optional[int] = None
    ) -> List[Tuple[float, int]]:
        """
        查询最近的 k 个向量

        返回: [(distance, node_id), ...] 按距离升序
        """
        query = np.array(query, dtype=np.float32)
        if self.entry_point is None:
            return []

        ef = ef or max(self.ef_search, k)
        curr_ep = [self.entry_point]

        # 从顶层贪心下降到第 1 层
        for layer in range(self.max_layer, 0, -1):
            results_layer = self._search_layer(query, curr_ep, ef=1, layer=layer)
            curr_ep = [nid for _, nid in results_layer]

        # 第 0 层精细搜索
        candidates = self._search_layer(query, curr_ep, ef=ef, layer=0)

        # 取 top-k
        results = sorted(candidates)[:k]
        return results

    def __len__(self):
        return len(self.nodes)

    # ──────────────────────────────────────────
    #  核心内部方法
    # ──────────────────────────────────────────

    def _random_layer(self) -> int:
        """
        按指数衰减概率随机分配层级
        P(layer >= l) = exp(-l / m_L)
        """
        return int(-math.log(random.random()) * self.m_L)

    def _dist(self, a_id_or_vec, b_id: int) -> float:
        """计算向量到节点的距离"""
        if isinstance(a_id_or_vec, np.ndarray):
            vec_a = a_id_or_vec
        else:
            vec_a = self.nodes[a_id_or_vec].vector
        vec_b = self.nodes[b_id].vector
        return self.dist_func(vec_a, vec_b)

    def _search_layer(
            self,
            query: np.ndarray,
            entry_points: List[int],
            ef: int,
            layer: int,
    ) -> List[Tuple[float, int]]:
        """
        论文 Algorithm 2
        在图的某一层里，从给定入口点出发，用广度优先式的贪心扩展
        找到距离 query 最近的 ef 个节点
        返回: [(dist, node_id), ...] ef 个最近候选，升序
        """

        # 候选堆，dist 小的在堆顶，决定"下一个扩展谁"
        candidates: List[Tuple[float, int]] = []
        # 结果堆，dist 大的在堆顶（存负距离模拟），维护"当前最优的 ef 个"
        results: List[Tuple[float, int]] = []

        for ep in entry_points:
            d = self._dist(query, ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        visited: Set[int] = set(entry_points)
        while candidates:
            # 从 candidates 取最近节点 c
            c_dist, c_id = heapq.heappop(candidates)  # 最近的候选

            # 如果候选比结果集中最远的还远，证明不可能再找到更近的了，停止
            worst_result_dist = -results[0][0]
            if c_dist > worst_result_dist:
                break

            # 遍历 c 在该层的所有邻居 nb，判断是否可以加入扩展集
            for nb_id in self.nodes[c_id].neighbors[layer]:
                if nb_id in visited:
                    continue
                visited.add(nb_id)

                nb_dist = self._dist(query, nb_id)
                worst_result_dist = -results[0][0]
                # 如果当前邻居距离更近或者结果集尚未满，可以直接加入
                if nb_dist < worst_result_dist or len(results) < ef:
                    heapq.heappush(candidates, (nb_dist, nb_id))
                    heapq.heappush(results, (-nb_dist, nb_id))
                    # 超出 ef 则弹出最远的
                    if len(results) > ef:
                        heapq.heappop(results)
        # 返回当前层距离 query 最近的 ef 个节点（升序）
        return sorted((-d, nid) for d, nid in results)

    def _select_neighbors_heuristic(
            self,
            query: np.ndarray,
            candidates: List[Tuple[float, int]],
            M: int,
            layer: int = 0,
            extend_candidates: bool = False,
    ) -> List[Tuple[float, int]]:
        """
        启发式邻居选择（论文 Algorithm 4）
        相比简单取最近 M 个，能更好地保证图的连通性和方向覆盖。
        搜索路径完全依赖图的边。如果某个区域和入口点之间没有边相连，那个区域就完全不会被访问到。
        入口点                        孤立区域
        [B]─[A]─[q]                 [D]─[E]─[F]

        搜索从入口出发，沿边扩展：
        入口 → A → B → q → （q的邻居只有A,B，走投无路）

        D、E、F 没有任何边连过来，整个区域对这次搜索不可见
        """
        # 扩展候选集（可选）
        if extend_candidates:
            extra: Set[int] = set()
            for _, c_id in candidates:
                for nb in self.nodes[c_id].neighbors[layer]:  # 修复：用传入的 layer 替代硬编码的 0
                    extra.add(nb)
            extra_candidates = [(self._dist(query, nb), nb) for nb in extra]
            candidates = sorted(set(candidates) | set(extra_candidates))

        result: List[Tuple[float, int]] = []
        # 剩余候选（最小堆）
        W = list(candidates)
        heapq.heapify(W)

        discarded: List[Tuple[float, int]] = []

        while W and len(result) < M:
            e_dist, e_id = heapq.heappop(W)  # 距离 query 最近的候选

            # 启发式：只有当 e 比已选邻居中任意一个更近时才加入
            # 这样避免所有邻居都聚集在同一方向
            closer_to_query_than_to_result = True
            for r_dist, r_id in result:
                d_e_r = self._dist(self.nodes[e_id].vector, r_id)
                if d_e_r < e_dist:
                    closer_to_query_than_to_result = False
                    break

            if closer_to_query_than_to_result:
                result.append((e_dist, e_id))
            else:
                discarded.append((e_dist, e_id))

        # 如果结果不足 M，用丢弃的补充
        for item in discarded:
            if len(result) >= M:
                break
            result.append(item)

        return result

    def _prune_neighbors(self, node_id: int, layer: int, max_conn: int) -> None:
        """裁剪节点的邻居列表至 max_conn"""
        node = self.nodes[node_id]
        neighbors = node.neighbors[layer]
        if len(neighbors) <= max_conn:
            return
        candidates = [(self._dist(node.vector, nb), nb) for nb in neighbors]
        kept = self._select_neighbors_heuristic(node.vector, candidates, max_conn, layer=layer, extend_candidates=False)
        node.neighbors[layer] = [nid for _, nid in kept]

    # ──────────────────────────────────────────
    #  辅助 / 调试
    # ──────────────────────────────────────────

    def stats(self) -> dict:
        """打印索引统计信息"""
        layer_counts = {}
        for node in self.nodes.values():
            l = node.max_layer
            layer_counts[l] = layer_counts.get(l, 0) + 1

        avg_degree = {}
        for layer in range(self.max_layer + 1):
            degrees = [len(n.neighbors[layer]) for n in self.nodes.values()
                       if layer <= n.max_layer]
            avg_degree[layer] = round(sum(degrees) / len(degrees), 2) if degrees else 0

        return {
            'total_nodes': len(self.nodes),
            'max_layer': self.max_layer,
            'entry_point': self.entry_point,
            'nodes_per_layer': layer_counts,
            'avg_degree_per_layer': avg_degree,
        }


# ─────────────────────────────────────────────
#  演示 & 测试
# ─────────────────────────────────────────────

def recall_at_k(true_ids: List[int], pred_ids: List[int], k: int) -> float:
    true_set = set(true_ids[:k])
    pred_set = set(pred_ids[:k])
    return len(true_set & pred_set) / k


def demo():
    print("=" * 60)
    print("  HNSW 从零实现演示")
    print("=" * 60)

    # ── 参数 ──
    DIM = 64
    N_TRAIN = 2000
    N_QUERY = 100
    K = 10

    rng = np.random.default_rng(0)
    data = rng.random((N_TRAIN, DIM)).astype(np.float32)
    queries = rng.random((N_QUERY, DIM)).astype(np.float32)

    # ── 构建索引 ──
    print(f"\n[1] 构建索引  n={N_TRAIN}, dim={DIM}, M=16, ef_construction=200")
    index = HNSW(dim=DIM, space='l2', M=16, ef_construction=200, ef_search=50)

    t0 = time.time()
    for i, vec in enumerate(data):
        index.add(i, vec)
    build_time = time.time() - t0
    print(f"    构建耗时: {build_time:.3f}s  ({N_TRAIN / build_time:.0f} inserts/s)")

    s = index.stats()
    print(f"    最高层: {s['max_layer']}")
    print(f"    各层节点数: {s['nodes_per_layer']}")
    print(f"    各层平均度: {s['avg_degree_per_layer']}")

    # ── 查询 ──
    print(f"\n[2] 搜索  n_queries={N_QUERY}, K={K}, ef_search=50")
    t0 = time.time()
    hnsw_results = [index.search(q, k=K) for q in queries]
    query_time = time.time() - t0
    print(f"    查询耗时: {query_time * 1000:.1f}ms  ({N_QUERY / query_time:.0f} QPS)")

    # ── 暴力搜索作为 ground truth ──
    print(f"\n[3] 计算 Recall@{K} (对比暴力搜索)")
    recalls = []
    for qi, q in enumerate(queries):
        dists = np.linalg.norm(data - q, axis=1)
        true_top_k = np.argsort(dists)[:K].tolist()
        hnsw_top_k = [nid for _, nid in hnsw_results[qi]]
        recalls.append(recall_at_k(true_top_k, hnsw_top_k, K))

    print(f"    平均 Recall@{K}: {np.mean(recalls):.4f}")
    print(f"    最低 Recall@{K}: {np.min(recalls):.4f}")

    # ── ef_search 对比 ──
    print(f"\n[4] ef_search 对 Recall 和延迟的影响")
    print(f"    {'ef_search':>10} | {'Recall@10':>10} | {'QPS':>8}")
    print(f"    {'-' * 34}")
    for ef in [10, 20, 50, 100, 200]:
        t0 = time.time()
        results = [index.search(q, k=K, ef=ef) for q in queries]
        qt = time.time() - t0
        rc = []
        for qi, q in enumerate(queries):
            dists = np.linalg.norm(data - q, axis=1)
            true_top = np.argsort(dists)[:K].tolist()
            hnsw_k = [nid for _, nid in results[qi]]
            rc.append(recall_at_k(true_top, hnsw_k, K))
        print(f"    {ef:>10} | {np.mean(rc):>10.4f} | {N_QUERY / qt:>8.0f}")

    # ── 单条查询示例 ──
    print(f"\n[5] 单条查询示例")
    q = queries[0]
    results = index.search(q, k=5)
    print(f"    Query 向量 (前4维): {q[:4]}")
    print(f"    Top-5 结果:")
    for rank, (dist, nid) in enumerate(results, 1):
        print(f"      #{rank}  node_id={nid:4d}  distance={dist:.5f}")


if __name__ == '__main__':
    demo()
