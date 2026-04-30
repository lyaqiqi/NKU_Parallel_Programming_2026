#!/usr/bin/env python3
"""
generate_pq.py — Build PQ codebook and codes for DEEP100K
Run on cluster (data lives at /anndata/).

Outputs:
  files/pq_codebook.bin  — 4 × 256 × 24 float32, raw binary (no header)
  files/pq_codes.bin     — N × 4   uint8,  raw binary (no header)
"""
import numpy as np
import os
from sklearn.cluster import KMeans

M      = 4
KS     = 256
D      = 96
SUBDIM = D // M   # 24

BASE_PATH = "/anndata/DEEP100K.base.100k.fbin"
OUT_DIR   = "files"
os.makedirs(OUT_DIR, exist_ok=True)

# --- read base vectors (fbin: int32 n, int32 d, then n*d float32) ---
with open(BASE_PATH, "rb") as f:
    n = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    d = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    base = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d).copy()
print(f"Loaded base: n={n}, d={d}")
assert d == D, f"Expected d={D}, got {d}"

codebook = np.zeros((M, KS, SUBDIM), dtype=np.float32)
codes    = np.zeros((n,  M),         dtype=np.uint8)

for m in range(M):
    sub = base[:, m * SUBDIM : (m + 1) * SUBDIM]
    print(f"Subspace {m}: running KMeans(k={KS}, subdim={SUBDIM}) ...")
    # KMeans 比 MiniBatchKMeans 精度更高（100K×24 维规模可接受）
    km = KMeans(
        n_clusters=KS, random_state=42,
        n_init=10, max_iter=500,
        algorithm="lloyd"
    )
    km.fit(sub)
    codebook[m] = km.cluster_centers_.astype(np.float32)
    codes[:, m] = km.labels_.astype(np.uint8)
    print(f"  done, inertia={km.inertia_:.2f}")

# --- save ---
cb_path    = os.path.join(OUT_DIR, "pq_codebook.bin")
codes_path = os.path.join(OUT_DIR, "pq_codes.bin")

codebook.tofile(cb_path)
codes.tofile(codes_path)

print(f"Saved codebook  ({M}x{KS}x{SUBDIM} f32): {cb_path}")
print(f"Saved codes     ({n}x{M} u8):            {codes_path}")
