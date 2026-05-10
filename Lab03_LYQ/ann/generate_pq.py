#!/usr/bin/env python3
"""
generate_pq.py — Build PQ codebook and codes for DEEP100K with optimized centroid indexing.

Outputs:
  files/pq_codebook.bin  — 4 × 256 × 24 float32, raw binary (no header)
  files/pq_codes.bin     — N × 4   uint8, AoS layout: codes[i * 4 + m]

Optimized centroid reindexing (required for FastScan efficiency):
  After KMeans, centroids within each subspace are sorted by their L2 norm,
  so consecutive groups of 16 centroids are spatially similar.
  This makes each 16-entry LUT portion cover a narrow IP-score range,
  giving tighter max-tables and stronger FastScan pruning (paper §4.2-4.3).
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
    print(f"Subspace {m}: KMeans(k={KS}, subdim={SUBDIM}) ...")
    km = KMeans(
        n_clusters=KS, random_state=42,
        n_init=10, max_iter=500,
        algorithm="lloyd"
    )
    km.fit(sub)
    centers = km.cluster_centers_.astype(np.float32)
    labels  = km.labels_.astype(np.int32)
    print(f"  KMeans done, inertia={km.inertia_:.2f}")

    # --- Optimized centroid reindexing for FastScan ---
    # Sort the 256 centroids by L2 norm. Centroids with similar norm tend to
    # have similar inner-product values with any query sub-vector, so consecutive
    # 16-centroid portions cover a narrow score range → tighter max-tables.
    # Simple sort guarantees exactly 16 centroids per portion (no KMeans needed).
    norms      = np.linalg.norm(centers, axis=1)       # shape (KS,)
    sort_order = np.argsort(norms)                     # sort_order[new_idx] = old_idx
    perm       = np.empty(KS, dtype=np.int32)
    perm[sort_order] = np.arange(KS, dtype=np.int32)  # perm[old_idx] = new_idx

    new_centers = centers[sort_order]   # new_centers[new_idx] = old centroid
    new_labels  = perm[labels]          # remap each vector's code
    print(f"  Centroid reindexing done.")

    codebook[m] = new_centers
    codes[:, m] = new_labels.astype(np.uint8)

# --- save (AoS layout: codes[i*M+m]) ---
cb_path    = os.path.join(OUT_DIR, "pq_codebook.bin")
codes_path = os.path.join(OUT_DIR, "pq_codes.bin")

codebook.tofile(cb_path)
codes.tofile(codes_path)

print(f"Saved codebook  ({M}x{KS}x{SUBDIM} f32): {cb_path}")
print(f"Saved codes     ({n}x{M} u8, AoS):        {codes_path}")
