"""
从零手写 PQ (Product Quantization) 乘积量化
只依赖 numpy

核心思想：
  1. 把高维向量切成 M 段
  2. 每段独立做 K-Means，训练码本 codebook[m]  (256个质心)
  3. 编码：每段子向量 → 找最近质心编号 (0~255) → 存 uint8
  4. 解码：uint8 编号 → 查码本 → 还原近似向量
  5. 距离：ADC 查表，不需要还原向量
"""

import numpy as np
import time


# ═══════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════

def l2_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A (N,d) 到 B (M,d) 的 L2 距离矩阵，返回 (N,M)"""
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T
    AB = A @ B.T
    return np.maximum(A2 + B2 - 2 * AB, 0)


def kmeans(vectors: np.ndarray, k: int, n_iter: int = 20) -> np.ndarray:
    """K-Means，返回 k 个质心"""
    idx = np.random.choice(len(vectors), k, replace=False)
    centroids = vectors[idx].copy()

    for _ in range(n_iter):
        dists = l2_distance_matrix(vectors, centroids)
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
# PQ 主类
# ═══════════════════════════════════════════════════════

class ProductQuantizer:

    def __init__(self, dim: int, M: int, ksub: int = 256):
        """
        dim  : 原始向量维度（必须能被 M 整除）
        M    : 切段数
        ksub : 每段质心数，默认 256 配合 uint8
        """
        assert dim % M == 0, f"dim={dim} 必须能被 M={M} 整除"
        assert ksub <= 256, f"ksub={ksub} 超过 uint8 上限 256"

        self.dim = dim
        self.M = M
        self.ksub = ksub
        self.dsub = dim // M  # 每段维度

        self.codebooks = None  # shape: (M, ksub, dsub)
        self.is_trained = False

    # ────────────────────────────────────
    # 训练
    # ────────────────────────────────────
    def train(self, vectors: np.ndarray):
        """
        对每段分别做 K-Means，训练 M 个码本

        vectors: (N, dim)
        """
        N = len(vectors)
        print(f"[PQ train] N={N}, dim={self.dim}, M={self.M}, "
              f"dsub={self.dsub}, ksub={self.ksub}")

        self.codebooks = np.zeros((self.M, self.ksub, self.dsub), dtype="float32")

        for m in range(self.M):
            # 取第 m 段的所有子向量: (N, dsub)
            sub = vectors[:, m * self.dsub: (m + 1) * self.dsub]
            self.codebooks[m] = kmeans(sub, self.ksub)
            print(f"  码本 {m:>2} 训练完成，质心 shape: {self.codebooks[m].shape}")

        self.is_trained = True
        mem = self.codebooks.nbytes
        print(f"[PQ train] 完成，码本占用内存: {mem} bytes ({mem / 1024:.1f} KB)\n")

    # ────────────────────────────────────
    # 编码
    # ────────────────────────────────────
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        原始向量 → PQ codes (有损压缩)

        输入:  (N, dim)   float32
        输出:  (N, M)     uint8

        压缩比: dim*4字节 → M*1字节

        例子 (dim=8, M=2, dsub=4, ksub=4):
          原始向量: [0.48, 0.58, 0.68, 0.78,  0.29, 0.19, 0.09, 0.88]
                     ↑────────第0段────────↑  ↑────────第1段────────↑

          第0段 [0.48,0.58,0.68,0.78] 到码本0各质心的距离:
            质心0 [0.1,0.2,0.3,0.4] → 距离 0.58
            质心1 [0.5,0.6,0.7,0.8] → 距离 0.001  ← 最近
            质心2 [0.9,0.1,0.2,0.3] → 距离 1.23
            质心3 [0.4,0.5,0.6,0.7] → 距离 0.12
            → code = 1

          第1段 [0.29,0.19,0.09,0.88] 到码本1各质心的距离:
            质心0 [0.8,0.7,0.6,0.5] → 距离 1.45
            质心1 [0.3,0.2,0.1,0.9] → 距离 0.001  ← 最近
            质心2 [0.6,0.5,0.4,0.3] → 距离 0.87
            质心3 [0.1,0.9,0.8,0.7] → 距离 1.02
            → code = 1

          最终 codes = [1, 1]
          原始 32字节 → 压缩为 2字节 (uint8)
        """
        assert self.is_trained, "请先 train()"
        N = len(vectors)
        codes = np.zeros((N, self.M), dtype="uint8")

        for m in range(self.M):
            sub = vectors[:, m * self.dsub: (m + 1) * self.dsub]  # (N, dsub)  取第m段
            dists = l2_distance_matrix(sub, self.codebooks[m])  # (N, ksub)  到每个质心的距离
            codes[:, m] = np.argmin(dists, axis=1).astype("uint8")  # 取最近质心的编号存为uint8

        return codes

    # ────────────────────────────────────
    # 解码
    # ────────────────────────────────────
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        PQ codes → 近似还原向量（有损）

        输入:  (N, M)     uint8
        输出:  (N, dim)   float32

        例子 (dim=8, M=2, dsub=4):
          codes = [1, 1]

          第0段 code=1 → 查码本0[1] → [0.5, 0.6, 0.7, 0.8]
          第1段 code=1 → 查码本1[1] → [0.3, 0.2, 0.1, 0.9]

          还原向量: [0.5, 0.6, 0.7, 0.8,  0.3, 0.2, 0.1, 0.9]
          原始向量: [0.48,0.58,0.68,0.78, 0.29,0.19,0.09,0.88]
                                                    ↑ 有误差，质心是近似值
          误差来源：编码时"四舍五入"到最近质心，质心坐标≠原始值
          ksub 越大（质心越多），误差越小，但压缩率越低
        """
        assert self.is_trained, "请先 train()"
        N = len(codes)
        vectors = np.zeros((N, self.dim), dtype="float32")

        for m in range(self.M):
            # codes[:, m] 是一批编号，直接作为下标索引码本，一次取出所有向量的第m段质心坐标
            vectors[:, m * self.dsub: (m + 1) * self.dsub] = self.codebooks[m][codes[:, m]]

        return vectors

    # ────────────────────────────────────
    # ADC 距离计算
    # ────────────────────────────────────
    def distance_table(self, query: np.ndarray) -> np.ndarray:
        """
        预计算 query 每段到所有子质心的距离表

        输入:  query (dim,)
        输出:  table (M, ksub)

        table[m][c] = query 第m段 到 码本m第c个质心 的 L2 距离
        """
        table = np.zeros((self.M, self.ksub), dtype="float32")
        for m in range(self.M):
            sub_q = query[m * self.dsub: (m + 1) * self.dsub]  # (dsub,)
            diff = self.codebooks[m] - sub_q  # (ksub, dsub)
            table[m] = (diff ** 2).sum(axis=1)
        return table

    def adc_distance(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """
        ADC (Asymmetric Distance Computation) 查表计算近似距离

        输入:  query (dim,),  codes (N, M)
        输出:  distances (N,)

        流程:
          1. 预算距离表 table (M, ksub)       — 只算一次
          2. 每条 code: dist = Σ table[m][code[m]]  — M次查表+加法
        """
        table = self.distance_table(query)  # (M, ksub)

        distances = np.zeros(len(codes), dtype="float32")
        for m in range(self.M):
            distances += table[m][codes[:, m]]  # 直接用 code 作为下标索引

        return distances


# ═══════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    DIM = 64
    M = 8  # 切 8 段，每段 8 维
    KSUB = 256
    N = 10_000
    N_Q = 5
    TOP_K = 5

    print("=" * 52)
    print("      Product Quantization from scratch")
    print("=" * 52)

    vectors = np.random.random((N, DIM)).astype("float32")
    queries = np.random.random((N_Q, DIM)).astype("float32")

    # ── 1. 训练 ──────────────────────────────
    pq = ProductQuantizer(dim=DIM, M=M, ksub=KSUB)
    pq.train(vectors)

    # ── 2. 编码 ──────────────────────────────
    print("=" * 52)
    print("[编码]")
    codes = pq.encode(vectors)
    print(f"  原始大小:   {vectors.nbytes / 1024:.1f} KB  (float32)")
    print(f"  压缩后大小: {codes.nbytes / 1024:.1f} KB  (uint8)")
    print(f"  压缩比:     {vectors.nbytes / codes.nbytes:.0f}x")
    print(f"  codes shape: {codes.shape}, dtype: {codes.dtype}")
    print(f"  前3条 codes:\n{codes[:3]}")

    # ── 3. 解码 & 重建误差 ───────────────────
    print("\n" + "=" * 52)
    print("[解码 & 重建误差]")
    reconstructed = pq.decode(codes[:5])
    for i in range(5):
        err = np.linalg.norm(vectors[i] - reconstructed[i])
        print(f"  向量 {i}: 重建 L2 误差 = {err:.4f}")

    # ── 4. ADC 距离 vs 精确距离 ──────────────
    print("\n" + "=" * 52)
    print("[ADC 近似距离 vs 精确 L2 距离]")
    q = queries[0]

    exact_dists = ((vectors - q) ** 2).sum(axis=1)
    approx_dists = pq.adc_distance(q, codes)

    # 取精确 Top-5
    exact_top = np.argsort(exact_dists)[:TOP_K]
    approx_top = np.argsort(approx_dists)[:TOP_K]

    print(f"\n  {'Rank':<6} {'精确idx':<10} {'精确dist':<14} {'近似idx':<10} {'近似dist':<14} {'匹配?'}")
    print("  " + "-" * 58)
    for r in range(TOP_K):
        ei, ed = exact_top[r], exact_dists[exact_top[r]]
        ai, ad = approx_top[r], approx_dists[approx_top[r]]
        match = "✅" if ei == ai else "❌"
        print(f"  {r + 1:<6} {ei:<10} {ed:<14.4f} {ai:<10} {ad:<14.4f} {match}")

    # ── 5. 速度对比 ───────────────────────────
    print("\n" + "=" * 52)
    print("[速度对比：精确 L2 vs ADC 查表]")

    t0 = time.time()
    for q in queries:
        _ = ((vectors - q) ** 2).sum(axis=1)
    exact_time = time.time() - t0

    t0 = time.time()
    for q in queries:
        _ = pq.adc_distance(q, codes)
    adc_time = time.time() - t0

    print(f"  精确 L2:  {exact_time * 1000:.2f} ms")
    print(f"  ADC 查表: {adc_time * 1000:.2f} ms")
    print(f"  速度提升: {exact_time / adc_time:.1f}x")

    # ── 6. M 对精度的影响 ────────────────────
    print("\n" + "=" * 52)
    print("[M（段数）对重建误差的影响]")
    print(f"  {'M':<6} {'dsub':<8} {'压缩比':<10} {'平均重建误差'}")
    print("  " + "-" * 40)
    for m in [1, 2, 4, 8, 16, 32, 64]:
        if DIM % m != 0:
            continue
        pq_tmp = ProductQuantizer(DIM, m, ksub=min(KSUB, 256))
        pq_tmp.train(vectors)
        c = pq_tmp.encode(vectors[:500])
        r = pq_tmp.decode(c)
        err = np.linalg.norm(vectors[:500] - r, axis=1).mean()
        ratio = (vectors.nbytes) / (codes.nbytes * DIM // (m * pq_tmp.dsub))
        print(f"  {m:<6} {DIM // m:<8} {vectors.nbytes // (m * 1) // N:<10} {err:.4f}")

    print("\n✅ 完成！")
