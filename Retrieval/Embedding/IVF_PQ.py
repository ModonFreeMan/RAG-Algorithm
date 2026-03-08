"""
从零手写 IVF_PQ 向量索引
只依赖 numpy，不使用 faiss

结构：
  - PQ (Product Quantization)：把向量切段 → 每段 K-Means → 用 code 代替原始值
  - IVF (Inverted File Index)：聚类分桶，只搜 nprobe 个桶
  - IVF_PQ = IVF 分桶 + 桶内存 PQ 编码 + ADC 查表计算距离
  query
  │
  ├─ [IVF] 比较64个粗质心 → 选8个桶 → 2500条候选
  │
  ├─ [PQ]  预算距离表 (8×256)        → 只算一次
  │
  ├─ [ADC] 2500条 × 8次查表          → 得到近似距离
  │
  └─ Top-K 排序 → 返回结果
"""

import numpy as np
import time


# ═══════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════

def l2_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    计算 A (N, d) 和 B (M, d) 之间的 L2 距离矩阵，返回 (N, M)
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a·b)
    """
    A2 = (A ** 2).sum(axis=1, keepdims=True)   # (N, 1)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T  # (1, M)
    AB = A @ B.T                                 # (N, M)
    return np.maximum(A2 + B2 - 2 * AB, 0)      # 避免浮点负数


def kmeans(vectors: np.ndarray, k: int, n_iter: int = 20) -> np.ndarray:
    """K-Means 聚类，返回 k 个质心"""
    idx = np.random.choice(len(vectors), k, replace=False)
    centroids = vectors[idx].copy()

    for _ in range(n_iter):
        dists = l2_distance_matrix(vectors, centroids)  # (N, k)
        assignments = np.argmin(dists, axis=1)

        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            members = vectors[assignments == c]
            new_centroids[c] = members.mean(axis=0) if len(members) > 0 else centroids[c]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids


# ═══════════════════════════════════════════════════════
# PQ：乘积量化
# ═══════════════════════════════════════════════════════

class ProductQuantizer:
    """
    将 dim 维向量切成 M 段，每段 dim/M 维
    每段用 K-Means 训练 ksub 个子质心（码本）
    每个向量编码为 M 个 uint8（每个是子质心编号 0~ksub-1）
    """

    def __init__(self, dim: int, M: int, ksub: int = 256):
        assert dim % M == 0, "dim 必须能被 M 整除"
        self.dim = dim
        self.M = M                    # 段数
        self.ksub = ksub              # 每段的子质心数（默认256，用 uint8 存）
        self.dsub = dim // M          # 每段的维度
        self.codebooks = None         # shape: (M, ksub, dsub)
        self.is_trained = False

    def train(self, vectors: np.ndarray):
        """对每段分别做 K-Means，训练 M 个码本"""
        print(f"   训练 PQ 码本 (M={self.M}, ksub={self.ksub}, dsub={self.dsub})...")
        self.codebooks = np.zeros((self.M, self.ksub, self.dsub), dtype="float32")

        for m in range(self.M):
            # 取第 m 段的所有子向量
            sub_vectors = vectors[:, m * self.dsub: (m + 1) * self.dsub]
            self.codebooks[m] = kmeans(sub_vectors, self.ksub)

        self.is_trained = True
        print(f"   PQ 码本训练完成")

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        将原始向量编码为 PQ codes
        输入: (N, dim)
        输出: (N, M) uint8
        """
        assert self.is_trained
        N = len(vectors)
        codes = np.zeros((N, self.M), dtype="uint8")

        for m in range(self.M):
            sub_vectors = vectors[:, m * self.dsub: (m + 1) * self.dsub]  # (N, dsub)
            dists = l2_distance_matrix(sub_vectors, self.codebooks[m])     # (N, ksub)
            codes[:, m] = np.argmin(dists, axis=1).astype("uint8")

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        将 PQ codes 还原为近似向量（有损）
        输入: (N, M) uint8
        输出: (N, dim) float32
        """
        N = len(codes)
        vectors = np.zeros((N, self.dim), dtype="float32")

        for m in range(self.M):
            vectors[:, m * self.dsub: (m + 1) * self.dsub] = self.codebooks[m][codes[:, m]]

        return vectors

    def compute_distance_table(self, query: np.ndarray) -> np.ndarray:
        """
        ADC（非对称距离计算）的核心：预计算查询向量各段到所有子质心的距离
        输入:  query (dim,)
        输出:  distance_table (M, ksub)  — 查距离时直接索引
        """
        table = np.zeros((self.M, self.ksub), dtype="float32")

        for m in range(self.M):
            sub_q = query[m * self.dsub: (m + 1) * self.dsub]  # (dsub,)
            diff = self.codebooks[m] - sub_q                     # (ksub, dsub)
            table[m] = (diff ** 2).sum(axis=1)                  # (ksub,)

        return table

    def adc_distance(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """
        用 ADC 查表计算 query 到一批 PQ 编码向量的近似距离
        输入: query (dim,), codes (N, M)
        输出: distances (N,)

        原理：
          每段距离 = distance_table[m, codes[n, m]]
          总距离   = 各段距离之和
        """
        table = self.compute_distance_table(query)  # (M, ksub)

        # 向量化查表：对每段 m，取出对应 code 的距离，累加
        # codes[:, m] shape: (N,) → table[m, codes[:, m]] shape: (N,)
        distances = np.zeros(len(codes), dtype="float32")
        for m in range(self.M):
            distances += table[m][codes[:, m]]

        return distances


# ═══════════════════════════════════════════════════════
# IVF_PQ 索引
# ═══════════════════════════════════════════════════════

class IVFPQIndex:
    """
    IVF_PQ = IVF 分桶（粗量化） + PQ 编码（精细量化）

    写入流程:
      train()  → ① IVF K-Means 找 nlist 个粗质心
                 ② PQ 在残差上训练码本（可选，这里在原始向量上训练）
      add()    → 每条向量分配到最近粗质心的桶，桶内存 PQ code（不存原始向量）

    搜索流程:
      search() → ① 找最近 nprobe 个桶
                 ② 对桶内所有 code 用 ADC 查表计算近似距离
                 ③ 返回 Top-K
    """

    def __init__(self, dim: int, nlist: int, M: int, ksub: int = 256):
        self.dim = dim
        self.nlist = nlist
        self.M = M
        self.ksub = ksub

        self.coarse_centroids = None          # IVF 粗质心 (nlist, dim)
        self.pq = ProductQuantizer(dim, M, ksub)

        # 倒排表：{cluster_id: {"ids": [...], "codes": np.ndarray(N, M)}}
        self.invlists = {}
        self.is_trained = False

    def train(self, vectors: np.ndarray):
        print(f"\n[IVF] 训练粗量化质心 (nlist={self.nlist})...")
        self.coarse_centroids = kmeans(vectors, self.nlist)
        print(f"[IVF] 粗质心训练完成")

        print(f"\n[PQ]  训练乘积量化码本...")
        self.pq.train(vectors)

        self.is_trained = True

    def add(self, vectors: np.ndarray):
        assert self.is_trained, "请先调用 train()"

        print(f"\n[ADD] 编码并写入倒排表...")
        # 1. 分配到粗质心
        dists = l2_distance_matrix(vectors, self.coarse_centroids)  # (N, nlist)
        assignments = np.argmin(dists, axis=1)                       # (N,)

        # 2. PQ 编码所有向量
        all_codes = self.pq.encode(vectors)  # (N, M)

        # 3. 写入倒排表
        self.invlists = {i: {"ids": [], "codes": []} for i in range(self.nlist)}
        for idx, cluster_id in enumerate(assignments):
            self.invlists[cluster_id]["ids"].append(idx)
            self.invlists[cluster_id]["codes"].append(all_codes[idx])

        # 转为 numpy array 方便后续批量查表
        for i in range(self.nlist):
            ids = self.invlists[i]["ids"]
            codes = self.invlists[i]["codes"]
            self.invlists[i]["ids"] = np.array(ids, dtype="int32")
            self.invlists[i]["codes"] = (
                np.array(codes, dtype="uint8") if len(codes) > 0
                else np.empty((0, self.M), dtype="uint8")
            )

        sizes = [len(self.invlists[i]["ids"]) for i in range(self.nlist)]
        print(f"[ADD] 完成。每桶平均 {np.mean(sizes):.0f} 条，"
              f"最多 {max(sizes)}，最少 {min(sizes)}")
        print(f"[ADD] 内存节省：原始 {vectors.nbytes/1024:.1f} KB → "
              f"编码后 {all_codes.nbytes/1024:.1f} KB "
              f"({all_codes.nbytes/vectors.nbytes*100:.1f}%)")

    def search(self, queries: np.ndarray, top_k: int, nprobe: int = 8):
        """
        queries: (Q, dim)
        返回 (distances, indices)，shape (Q, top_k)
        """
        nprobe = min(nprobe, self.nlist)
        all_dists, all_ids = [], []

        for q in queries:
            # Step 1: 找最近 nprobe 个粗质心
            coarse_dists = ((self.coarse_centroids - q) ** 2).sum(axis=1)
            top_clusters = np.argsort(coarse_dists)[:nprobe]

            # Step 2: 收集候选的 ids 和 codes
            cand_ids_list, cand_codes_list = [], []
            for cid in top_clusters:
                cand_ids_list.append(self.invlists[cid]["ids"])
                cand_codes_list.append(self.invlists[cid]["codes"])

            if all(len(x) == 0 for x in cand_ids_list):
                all_dists.append([float("inf")] * top_k)
                all_ids.append([-1] * top_k)
                continue

            cand_ids = np.concatenate(cand_ids_list)
            cand_codes = np.vstack([c for c in cand_codes_list if len(c) > 0])

            # Step 3: ADC 查表计算近似距离
            approx_dists = self.pq.adc_distance(q, cand_codes)  # (M_cand,)

            # Step 4: Top-K
            k = min(top_k, len(approx_dists))
            top_local = np.argsort(approx_dists)[:k]

            result_ids = cand_ids[top_local].tolist()
            result_dists = approx_dists[top_local].tolist()

            while len(result_ids) < top_k:
                result_ids.append(-1)
                result_dists.append(float("inf"))

            all_dists.append(result_dists)
            all_ids.append(result_ids)

        return np.array(all_dists), np.array(all_ids)


# ═══════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════

def brute_force(vectors, queries, top_k):
    all_dists, all_ids = [], []
    for q in queries:
        d = ((vectors - q) ** 2).sum(axis=1)
        top = np.argsort(d)[:top_k]
        all_ids.append(top.tolist())
        all_dists.append(d[top].tolist())
    return np.array(all_dists), np.array(all_ids)


def recall_at_k(true_ids, pred_ids):
    hits = sum(len(set(t) & set(p)) for t, p in zip(true_ids, pred_ids))
    return hits / (true_ids.shape[0] * true_ids.shape[1])


if __name__ == "__main__":
    np.random.seed(42)

    DIM = 64
    N = 20_000
    N_QUERY = 20
    TOP_K = 5
    NLIST = 64
    M = 8          # PQ 段数（DIM 必须能被 M 整除）
    KSUB = 256     # 每段子质心数
    NPROBE = 8

    print("=" * 58)
    print("         IVF_PQ from scratch (pure numpy)")
    print("=" * 58)

    print(f"\n📦 生成 {N} 条 {DIM} 维向量...")
    vectors = np.random.random((N, DIM)).astype("float32")
    queries = np.random.random((N_QUERY, DIM)).astype("float32")

    # ── 暴力搜索基准 ──────────────────────────
    print("\n🔍 暴力搜索（FLAT 基准）...")
    t0 = time.time()
    flat_dists, flat_ids = brute_force(vectors, queries, TOP_K)
    flat_time = time.time() - t0
    print(f"   耗时: {flat_time * 1000:.2f} ms")

    # ── 构建 IVF_PQ 索引 ──────────────────────
    print(f"\n🔨 构建 IVF_PQ (nlist={NLIST}, M={M}, ksub={KSUB})...")
    index = IVFPQIndex(dim=DIM, nlist=NLIST, M=M, ksub=KSUB)
    index.train(vectors)
    index.add(vectors)

    # ── 搜索 ──────────────────────────────────
    print(f"\n🔍 IVF_PQ 搜索 (nprobe={NPROBE})...")
    t0 = time.time()
    ivf_dists, ivf_ids = index.search(queries, TOP_K, nprobe=NPROBE)
    ivf_time = time.time() - t0
    print(f"   耗时: {ivf_time * 1000:.2f} ms")
    print(f"   速度提升: {flat_time / ivf_time:.1f}x")
    print(f"   召回率 Recall@{TOP_K}: {recall_at_k(flat_ids, ivf_ids) * 100:.1f}%")

    # ── 结果对比 ──────────────────────────────
    print("\n📋 第一条查询结果对比（FLAT vs IVF_PQ）：")
    print(f"{'Rank':<6} {'FLAT idx':<12} {'FLAT dist':<14} {'IVF_PQ idx':<14} {'近似dist':<14} {'匹配?'}")
    print("-" * 68)
    for r in range(TOP_K):
        fi, fd = flat_ids[0][r], flat_dists[0][r]
        ii, id_ = ivf_ids[0][r], ivf_dists[0][r]
        match = "✅" if fi == ii else "❌"
        print(f"{r+1:<6} {fi:<12} {fd:<14.4f} {ii:<14} {id_:<14.4f} {match}")

    # ── PQ 重建误差 ───────────────────────────
    print("\n🔬 PQ 重建误差分析（前5条向量）：")
    codes = index.pq.encode(vectors[:5])
    reconstructed = index.pq.decode(codes)
    for i in range(5):
        err = np.linalg.norm(vectors[i] - reconstructed[i])
        print(f"   向量 {i}: 重建误差 L2 = {err:.4f}")

    # ── nprobe 影响分析 ───────────────────────
    print(f"\n📊 nprobe 对召回率 & 速度的影响：")
    print(f"{'nprobe':<10} {'耗时(ms)':<14} {'速度提升':<12} {'Recall@5'}")
    print("-" * 48)
    for nprobe in [1, 4, 8, 16, 32, NLIST]:
        t0 = time.time()
        _, ids = index.search(queries, TOP_K, nprobe=nprobe)
        elapsed = (time.time() - t0) * 1000
        rc = recall_at_k(flat_ids, ids)
        speedup = flat_time / (elapsed / 1000)
        print(f"{nprobe:<10} {elapsed:<14.2f} {speedup:<12.1f} {rc*100:.1f}%")

    print("\n✅ 完成！")