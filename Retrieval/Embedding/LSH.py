"""
局部敏感哈希 (LSH) — 详细实现
包含两种版本：
  1. 随机投影 LSH（适用于余弦相似度）
  2. 随机超平面 LSH（适用于欧氏距离）
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


# ============================================================
# Part 1: 随机投影 LSH（余弦相似度 / 符号随机投影）
# ============================================================

class CosineLSH:
    """
    基于符号随机投影的 LSH，用于近似余弦相似度搜索。

    原理：
      - 随机生成超平面（法向量 r）
      - 对向量 v，哈希位 h = sign(r · v)
      - 两个向量哈希冲突的概率 = 1 - θ/π，θ 为夹角
      - 夹角越小（越相似），冲突概率越高
    相当于在向量空间内插入超平面，如果在同侧就是1，在异侧就是-1。
    最终向量如果被K个超平面切割，哈希表示就是K维[-1,1,1,-1...]
    因为超平面分割位置的随机性，可能存在两个相似的向量表示不同
    因此引入多张表：
        Table 1 (超平面组1): v → [+1, -1, +1, +1, -1] → 桶 "10110"
        Table 2 (超平面组2): v → [-1, -1, +1, -1, +1] → 桶 "00101"
        Table 3 (超平面组3): v → [+1, +1, -1, +1, +1] → 桶 "11011"
    只要向量在一个表或者说在一个超平面组中发生碰撞，就代表在同一个桶中

    准备阶段：随机生成 T 组超平面（每组 K 个）
          shape: (T, K, dim)
    对每个向量 v：
        for 每张表 t:
            用第 t 组超平面切割 v
            得到 K 维哈希码 [-1,+1,-1,+1...]
            把 v 的索引 i 放入 table[t][哈希码]

    检索：
    输入查询向量 q
    for 每张表 t:
        用第 t 组超平面切割 q  →  得到哈希码
        取出 tables[t][哈希码]  →  候选向量 id

    所有表候选取并集  →  候选集

    对候选集中每个向量精确计算相似度
    """

    def __init__(self, dim: int, n_hyperplanes: int, n_tables: int):
        """
        参数：
            dim           : 向量维度
            n_hyperplanes : 每张哈希表的超平面数（即哈希码长度）
            n_tables      : 哈希表数量（多表提高召回率）
        """
        self.dim = dim
        self.n_hyperplanes = n_hyperplanes
        self.n_tables = n_tables

        # 为每张表生成随机超平面法向量
        # shape: (n_tables, n_hyperplanes, dim)
        self.hyperplanes = np.random.randn(n_tables, n_hyperplanes, dim)

        # 每张表的桶: table_id -> {hash_code -> [idx, ...]}
        self.tables: List[Dict[tuple, List[int]]] = [
            defaultdict(list) for _ in range(n_tables)
        ]
        self.data: Optional[np.ndarray] = None

    def _hash(self, vecs: np.ndarray, table_id: int) -> np.ndarray:
        """
        计算一批向量在指定表中的哈希码。

        vecs: (N, dim)
        返回: (N, n_hyperplanes) bool 数组
        """
        # 投影: (N, dim) x (dim, n_hyperplanes) -> (N, n_hyperplanes)
        projections = vecs @ self.hyperplanes[table_id].T
        return projections > 0  # sign: True/False

    def _hash_to_key(self, bits: np.ndarray) -> tuple:
        """将 bool 数组转为可哈希的 tuple（作为桶的 key）"""
        return tuple(bits.astype(int))

    def index(self, data: np.ndarray):
        """
        建立索引。

        data: (N, dim) 数据集
        """
        self.data = data.copy()
        N = data.shape[0]

        for t in range(self.n_tables):
            hash_codes = self._hash(data, t)  # (N, n_hyperplanes)
            for i in range(N):
                key = self._hash_to_key(hash_codes[i])
                self.tables[t][key].append(i)

        print(f"[CosineLSH] 索引完成: {N} 条数据, "
              f"{self.n_tables} 张表, 每表 {self.n_hyperplanes} 位")
        self._print_bucket_stats()

    def _print_bucket_stats(self):
        """打印桶的统计信息"""
        for t in range(self.n_tables):
            sizes = [len(v) for v in self.tables[t].values()]
            print(f"  Table {t}: {len(sizes)} 个桶, "
                  f"平均桶大小={np.mean(sizes):.1f}, "
                  f"最大桶={max(sizes)}")

    def query(self, q: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        查询最相似的 top_k 个向量。

        q      : (dim,) 查询向量
        返回   : [(idx, cosine_similarity), ...] 按相似度降序
        """
        q = q / (np.linalg.norm(q) + 1e-10)  # 归一化

        # 收集所有表中命中同一桶的候选
        candidates = set()
        for t in range(self.n_tables):
            bits = self._hash(q[np.newaxis, :], t)[0]
            key = self._hash_to_key(bits)
            candidates.update(self.tables[t].get(key, []))

        if not candidates:
            return []

        # 对候选集精确计算余弦相似度
        candidates = list(candidates)
        cand_vecs = self.data[candidates]  # (M, dim)
        norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-10
        cand_vecs_norm = cand_vecs / norms

        sims = cand_vecs_norm @ q  # (M,)

        # 排序取 top_k
        order = np.argsort(-sims)[:top_k]
        return [(candidates[i], float(sims[i])) for i in order]


# ============================================================
# Part 2: 随机投影 LSH（欧氏距离）
# ============================================================

class EuclideanLSH:
    """
    基于随机投影 + 分桶的 LSH，用于近似欧氏距离搜索。

    原理：
      - 随机方向 r，将向量投影到 r 上
      - 按宽度 w 均匀分桶: h = floor((r·v + b) / w)
      - 距离越近，落入同一桶的概率越高
    """

    def __init__(self, dim: int, n_hyperplanes: int, n_tables: int, bucket_width: float = 1.0):
        """
        参数：
            dim           : 向量维度
            n_hyperplanes : 每张表使用的随机投影数（哈希码长度）
            n_tables      : 哈希表数量
            bucket_width  : 分桶宽度 w（越大桶越宽，候选越多）
        """
        self.dim = dim
        self.n_hyperplanes = n_hyperplanes
        self.n_tables = n_tables
        self.w = bucket_width

        # 随机方向向量，shape: (n_tables, n_hyperplanes, dim)
        self.rand_vectors = np.random.randn(n_tables, n_hyperplanes, dim)
        # 随机偏移，shape: (n_tables, n_hyperplanes)
        self.rand_bias = np.random.uniform(0, bucket_width, (n_tables, n_hyperplanes))

        self.tables: List[Dict[tuple, List[int]]] = [
            defaultdict(list) for _ in range(n_tables)
        ]
        self.data: Optional[np.ndarray] = None

    def _hash(self, vecs: np.ndarray, table_id: int) -> np.ndarray:
        """
        计算欧氏 LSH 哈希码。

        vecs: (N, dim)
        返回: (N, n_hyperplanes) int 数组（桶编号）
        """
        # 投影: (N, n_hyperplanes)
        proj = vecs @ self.rand_vectors[table_id].T
        # 加偏移，除以桶宽，取整
        return np.floor((proj + self.rand_bias[table_id]) / self.w).astype(int)

    def index(self, data: np.ndarray):
        """建立索引"""
        self.data = data.copy()
        N = data.shape[0]

        for t in range(self.n_tables):
            hash_codes = self._hash(data, t)  # (N, n_hyperplanes)
            for i in range(N):
                key = tuple(hash_codes[i])
                self.tables[t][key].append(i)

        print(f"[EuclideanLSH] 索引完成: {N} 条数据, "
              f"{self.n_tables} 张表, 桶宽={self.w}")

    def query(self, q: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        查询最近的 top_k 个向量（按欧氏距离）。

        返回: [(idx, distance), ...] 按距离升序
        """
        candidates = set()
        for t in range(self.n_tables):
            bits = self._hash(q[np.newaxis, :], t)[0]
            key = tuple(bits)
            candidates.update(self.tables[t].get(key, []))

        if not candidates:
            return []

        candidates = list(candidates)
        cand_vecs = self.data[candidates]
        dists = np.linalg.norm(cand_vecs - q, axis=1)

        order = np.argsort(dists)[:top_k]
        return [(candidates[i], float(dists[i])) for i in order]


# ============================================================
# Part 3: 工具函数 — 暴力搜索（用于对比评估）
# ============================================================

def brute_force_cosine(data: np.ndarray, q: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    """精确余弦相似度搜索"""
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-10
    sims = (data / norms) @ q_norm
    order = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in order]


def brute_force_euclidean(data: np.ndarray, q: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    """精确欧氏距离搜索"""
    dists = np.linalg.norm(data - q, axis=1)
    order = np.argsort(dists)[:top_k]
    return [(int(i), float(dists[i])) for i in order]


def recall_at_k(lsh_results: List[Tuple], true_results: List[Tuple], k: int) -> float:
    """计算 Recall@K"""
    lsh_ids = set(r[0] for r in lsh_results[:k])
    true_ids = set(r[0] for r in true_results[:k])
    return len(lsh_ids & true_ids) / max(len(true_ids), 1)


# ============================================================
# Part 4: 演示
# ============================================================

def demo_cosine_lsh():
    print("=" * 60)
    print("演示 1: 余弦 LSH")
    print("=" * 60)

    np.random.seed(42)
    N, dim = 5000, 128
    data = np.random.randn(N, dim)

    # 构建索引
    # n_hyperplanes 较小 → 桶更宽 → 候选更多 → 召回率更高
    lsh = CosineLSH(dim=dim, n_hyperplanes=6, n_tables=10)
    lsh.index(data)

    # 随机查询
    q = np.random.randn(dim)
    top_k = 10

    lsh_results = lsh.query(q, top_k=top_k)
    true_results = brute_force_cosine(data, q, top_k=top_k)

    recall = recall_at_k(lsh_results, true_results, top_k)

    print(f"\n查询向量维度: {dim}")
    print(f"LSH  候选数量: {len(lsh_results)}")
    print(f"Recall@{top_k}: {recall:.2%}")

    print("\nLSH Top-5 结果 (idx, cosine_sim):")
    for idx, sim in lsh_results[:5]:
        print(f"  idx={idx:4d}, sim={sim:.4f}")

    print("\n真实 Top-5 结果 (idx, cosine_sim):")
    for idx, sim in true_results[:5]:
        print(f"  idx={idx:4d}, sim={sim:.4f}")


def demo_euclidean_lsh():
    print("\n" + "=" * 60)
    print("演示 2: 欧氏 LSH")
    print("=" * 60)

    np.random.seed(0)
    N, dim = 5000, 64
    data = np.random.randn(N, dim)

    lsh = EuclideanLSH(dim=dim, n_hyperplanes=4, n_tables=8, bucket_width=4.0)
    lsh.index(data)

    q = np.random.randn(dim)
    top_k = 10

    lsh_results = lsh.query(q, top_k=top_k)
    true_results = brute_force_euclidean(data, q, top_k=top_k)

    recall = recall_at_k(lsh_results, true_results, top_k)

    print(f"\n查询向量维度: {dim}")
    print(f"LSH  候选数量: {len(lsh_results)}")
    print(f"Recall@{top_k}: {recall:.2%}")

    print("\nLSH Top-5 结果 (idx, dist):")
    for idx, dist in lsh_results[:5]:
        print(f"  idx={idx:4d}, dist={dist:.4f}")

    print("\n真实 Top-5 结果 (idx, dist):")
    for idx, dist in true_results[:5]:
        print(f"  idx={idx:4d}, dist={dist:.4f}")


def demo_parameter_sensitivity():
    """
    演示参数对 Recall 的影响：
      - n_hyperplanes 越多 → 哈希码越长 → 桶越精细 → 召回率下降
      - n_tables 越多      → 多表并集   → 召回率上升
    """
    print("\n" + "=" * 60)
    print("演示 3: 参数敏感性分析")
    print("=" * 60)

    np.random.seed(1)
    N, dim, top_k = 3000, 64, 10
    data = np.random.randn(N, dim)
    queries = np.random.randn(20, dim)

    true_results_all = [brute_force_cosine(data, q, top_k) for q in queries]

    print(f"\n{'n_hyperplanes':>14} {'n_tables':>9} {'avg Recall@10':>14}")
    print("-" * 42)

    for n_hp in [4, 8, 12, 16]:
        for n_t in [3, 6]:
            lsh = CosineLSH(dim=dim, n_hyperplanes=n_hp, n_tables=n_t)
            lsh.index(data)
            recalls = [
                recall_at_k(lsh.query(q, top_k), tr, top_k)
                for q, tr in zip(queries, true_results_all)
            ]
            print(f"{n_hp:>14} {n_t:>9} {np.mean(recalls):>13.2%}")


if __name__ == "__main__":
    demo_cosine_lsh()
    demo_euclidean_lsh()
    demo_parameter_sensitivity()