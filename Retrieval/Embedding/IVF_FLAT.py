"""
从零手写 IVF_FLAT 向量索引
只依赖 numpy，不使用 faiss
"""

import numpy as np
import time


# 本质上就是 K-Means


# ─────────────────────────────────────────────
# K-Means 聚类（用于训练质心）
# ─────────────────────────────────────────────
def kmeans(vectors: np.ndarray, k: int, n_iter: int = 20) -> np.ndarray:
    """简单 K-Means，返回 k 个质心"""
    # 随机选 k 个向量作为初始质心
    indices = np.random.choice(len(vectors), k, replace=False)
    centroids = vectors[indices].copy()

    for i in range(n_iter):
        # 1. 分配：每个向量找最近质心
        assignments = assign_to_centroids(vectors, centroids)

        # 2. 更新：每个簇取均值作为新质心
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            members = vectors[assignments == c]
            if len(members) > 0:
                new_centroids[c] = members.mean(axis=0)
            else:
                new_centroids[c] = centroids[c]  # 空簇保留旧质心

        # 收敛检查
        if np.allclose(centroids, new_centroids):
            print(f"   K-Means 第 {i + 1} 轮收敛")
            break
        centroids = new_centroids

    return centroids


def assign_to_centroids(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """计算每个向量距离最近的质心编号（L2 距离）"""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a·b)
    # 利用广播避免 for 循环，shape: (N, k)
    a2 = (vectors ** 2).sum(axis=1, keepdims=True)  # (N, 1)
    b2 = (centroids ** 2).sum(axis=1, keepdims=True).T  # (1, k)
    ab = vectors @ centroids.T  # (N, k)
    dists = a2 + b2 - 2 * ab
    return np.argmin(dists, axis=1)


# ─────────────────────────────────────────────
# IVF_FLAT 索引类
# ─────────────────────────────────────────────
class IVFFlatIndex:
    def __init__(self, dim: int, nlist: int):
        """
        dim   : 向量维度
        nlist : 聚类数（桶的数量）
        """
        self.dim = dim
        self.nlist = nlist
        self.centroids = None  # shape: (nlist, dim)
        self.buckets = {}  # {簇id: [原始向量idx, ...]}
        self.vectors = None  # 所有向量（用于精确计算距离）
        self.is_trained = False

    # ── 训练：K-Means 找质心 ──────────────────
    def train(self, vectors: np.ndarray):
        print(f"   训练 K-Means (nlist={self.nlist}, n={len(vectors)})...")
        self.centroids = kmeans(vectors, self.nlist)
        self.is_trained = True
        print(f"   训练完成，得到 {self.nlist} 个质心")

    # ── 添加向量 ──────────────────────────────
    def add(self, vectors: np.ndarray):
        assert self.is_trained, "请先调用 train()"
        self.vectors = vectors

        # 初始化桶
        self.buckets = {i: [] for i in range(self.nlist)}

        # 每个向量分配到最近的质心
        assignments = assign_to_centroids(vectors, self.centroids)
        for idx, cluster_id in enumerate(assignments):
            self.buckets[cluster_id].append(idx)

        sizes = [len(v) for v in self.buckets.values()]
        print(f"   已添加 {len(vectors)} 条向量")
        print(f"   每桶平均 {np.mean(sizes):.0f} 条，最多 {max(sizes)} 条，最少 {min(sizes)} 条")

    # ── 搜索 ──────────────────────────────────
    def search(self, queries: np.ndarray, top_k: int, nprobe: int = 8):
        """
        queries : shape (Q, dim)
        top_k   : 返回最近的 K 个
        nprobe  : 查询几个桶
        返回 (distances, indices)，shape 均为 (Q, top_k)
        """
        assert self.vectors is not None, "请先调用 add()"
        nprobe = min(nprobe, self.nlist)

        all_dists = []
        all_ids = []

        for q in queries:
            # Step 1: 找最近的 nprobe 个质心
            centroid_dists = l2_distance(q, self.centroids)  # (nlist,)
            top_clusters = np.argsort(centroid_dists)[:nprobe]

            # Step 2: 收集候选向量的 idx
            candidate_ids = []
            for cid in top_clusters:
                candidate_ids.extend(self.buckets[cid])

            if len(candidate_ids) == 0:
                all_dists.append([float("inf")] * top_k)
                all_ids.append([-1] * top_k)
                continue

            candidate_ids = np.array(candidate_ids)

            # Step 3: 在候选集里精确计算 L2 距离
            candidates = self.vectors[candidate_ids]  # (M, dim)
            dists = l2_distance(q, candidates)  # (M,)

            # Step 4: 取 top_k
            k = min(top_k, len(dists))
            top_local = np.argsort(dists)[:k]

            result_ids = candidate_ids[top_local].tolist()
            result_dists = dists[top_local].tolist()

            # 不足 top_k 时补齐
            while len(result_ids) < top_k:
                result_ids.append(-1)
                result_dists.append(float("inf"))

            all_dists.append(result_dists)
            all_ids.append(result_ids)

        return np.array(all_dists), np.array(all_ids)


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def l2_distance(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """query (dim,) 到 targets (N, dim) 的 L2 距离"""
    diff = targets - query
    return (diff ** 2).sum(axis=1)


def brute_force_search(vectors, queries, top_k):
    """暴力搜索（FLAT），作为精度基准"""
    all_dists, all_ids = [], []
    for q in queries:
        dists = l2_distance(q, vectors)
        top = np.argsort(dists)[:top_k]
        all_ids.append(top.tolist())
        all_dists.append(dists[top].tolist())
    return np.array(all_dists), np.array(all_ids)


def recall_at_k(true_ids, pred_ids):
    hits = sum(len(set(t) & set(p)) for t, p in zip(true_ids, pred_ids))
    return hits / (true_ids.shape[0] * true_ids.shape[1])


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    DIM = 32
    N = 10_000
    N_QUERY = 20
    TOP_K = 5
    NLIST = 50
    NPROBE = 8

    print("=" * 55)
    print("         IVF_FLAT from scratch (pure numpy)")
    print("=" * 55)

    # 1. 生成数据
    print(f"\n📦 生成 {N} 条 {DIM} 维向量...")
    vectors = np.random.random((N, DIM)).astype("float32")
    queries = np.random.random((N_QUERY, DIM)).astype("float32")

    # 2. 暴力搜索（基准）
    print("\n🔍 暴力搜索（FLAT 基准）...")
    t0 = time.time()
    flat_dists, flat_ids = brute_force_search(vectors, queries, TOP_K)
    flat_time = time.time() - t0
    print(f"   耗时: {flat_time * 1000:.2f} ms")

    # 3. 构建 IVF_FLAT 索引
    print(f"\n🔨 构建 IVF_FLAT 索引 (nlist={NLIST})...")
    index = IVFFlatIndex(dim=DIM, nlist=NLIST)
    index.train(vectors)
    index.add(vectors)

    # 4. IVF 搜索
    print(f"\n🔍 IVF_FLAT 搜索 (nprobe={NPROBE})...")
    t0 = time.time()
    ivf_dists, ivf_ids = index.search(queries, TOP_K, nprobe=NPROBE)
    ivf_time = time.time() - t0
    print(f"   耗时: {ivf_time * 1000:.2f} ms")
    print(f"   速度提升: {flat_time / ivf_time:.1f}x")
    print(f"   召回率 Recall@{TOP_K}: {recall_at_k(flat_ids, ivf_ids) * 100:.1f}%")

    # 5. 结果对比
    print("\n📋 第一条查询结果对比：")
    print(f"{'Rank':<6} {'FLAT idx':<12} {'FLAT dist':<14} {'IVF idx':<12} {'IVF dist':<14} {'匹配?'}")
    print("-" * 62)
    for r in range(TOP_K):
        fi, fd = flat_ids[0][r], flat_dists[0][r]
        ii, id_ = ivf_ids[0][r], ivf_dists[0][r]
        match = "✅" if fi == ii else "❌"
        print(f"{r + 1:<6} {fi:<12} {fd:<14.4f} {ii:<12} {id_:<14.4f} {match}")

    # 6. nprobe 影响分析
    print(f"\n📊 nprobe 对召回率 & 速度的影响：")
    print(f"{'nprobe':<10} {'耗时(ms)':<14} {'速度提升':<12} {'Recall@5'}")
    print("-" * 48)
    for nprobe in [1, 5, 10, 20, NLIST]:
        t0 = time.time()
        _, ids = index.search(queries, TOP_K, nprobe=nprobe)
        elapsed = (time.time() - t0) * 1000
        rc = recall_at_k(flat_ids, ids)
        speedup = flat_time / (elapsed / 1000)
        print(f"{nprobe:<10} {elapsed:<14.2f} {speedup:<12.1f} {rc * 100:.1f}%")

    print("\n✅ 完成！")
