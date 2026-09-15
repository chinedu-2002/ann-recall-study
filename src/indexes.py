"""Uniform wrappers over the three ANN index families under study.

Each wrapper exposes the same interface so the sweep driver does not need to
know which library it is talking to:
    build(base)            -> seconds
    set_query_param(v)     -> None
    search(queries, k)     -> (ids, seconds_per_query_list)
    memory_bytes()         -> int
    defaults()             -> the library's own default parameters
"""
import time
import numpy as np
import faiss
import hnswlib


class HNSWIndex:
    """hnswlib Hierarchical Navigable Small World graph."""
    family = "HNSW"
    # hnswlib's own documented defaults
    DEFAULTS = {"M": 16, "efConstruction": 200, "ef": 10}

    def __init__(self, dim, metric, M=16, efConstruction=200):
        self.dim, self.metric = dim, metric
        self.M, self.efConstruction = M, efConstruction
        space = "l2" if metric == "euclidean" else "ip"
        self.index = hnswlib.Index(space=space, dim=dim)
        self.ef = self.DEFAULTS["ef"]

    def build(self, base):
        t0 = time.perf_counter()
        self.index.init_index(max_elements=base.shape[0], ef_construction=self.efConstruction,
                              M=self.M, random_seed=42)
        self.index.set_num_threads(1)
        self.index.add_items(base, np.arange(base.shape[0]))
        return time.perf_counter() - t0

    def set_query_param(self, ef):
        self.ef = int(ef)
        self.index.set_ef(self.ef)

    def search(self, queries, k):
        self.index.set_num_threads(1)
        times = []
        out = np.empty((queries.shape[0], k), dtype=np.int32)
        for i in range(queries.shape[0]):
            q = queries[i:i + 1]
            t0 = time.perf_counter()
            ids, _ = self.index.knn_query(q, k=k)
            times.append(time.perf_counter() - t0)
            out[i] = ids[0]
        return out, times

    def memory_bytes(self):
        n = self.index.get_current_count()
        # graph: (M*2 links on layer 0) * 4B + id 8B, plus the stored vector
        return int(n * (self.M * 2 * 4 + 8 + self.dim * 4))

    def params(self):
        return {"M": self.M, "efConstruction": self.efConstruction, "ef": self.ef}


class IVFFlatIndex:
    """FAISS inverted file index with uncompressed (flat) residuals."""
    family = "IVF-Flat"
    DEFAULTS = {"nlist": 100, "nprobe": 1}

    def __init__(self, dim, metric, nlist=100):
        self.dim, self.metric, self.nlist = dim, metric, nlist
        m = faiss.METRIC_L2 if metric == "euclidean" else faiss.METRIC_INNER_PRODUCT
        quant = faiss.IndexFlatL2(dim) if metric == "euclidean" else faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quant, dim, nlist, m)
        self.nprobe = self.DEFAULTS["nprobe"]

    def build(self, base):
        faiss.omp_set_num_threads(1)
        t0 = time.perf_counter()
        self.index.train(base)
        self.index.add(base)
        return time.perf_counter() - t0

    def set_query_param(self, nprobe):
        self.nprobe = int(nprobe)
        self.index.nprobe = self.nprobe

    def search(self, queries, k):
        faiss.omp_set_num_threads(1)
        times = []
        out = np.empty((queries.shape[0], k), dtype=np.int32)
        for i in range(queries.shape[0]):
            t0 = time.perf_counter()
            _, ids = self.index.search(queries[i:i + 1], k)
            times.append(time.perf_counter() - t0)
            out[i] = ids[0]
        return out, times

    def memory_bytes(self):
        return int(self.index.ntotal * (self.dim * 4 + 8) + self.nlist * self.dim * 4)

    def params(self):
        return {"nlist": self.nlist, "nprobe": self.nprobe}


class IVFPQIndex:
    """FAISS inverted file index with product-quantized residuals."""
    family = "IVF-PQ"
    DEFAULTS = {"nlist": 100, "m": 8, "nbits": 8, "nprobe": 1}

    def __init__(self, dim, metric, nlist=100, m=8, nbits=8):
        self.dim, self.metric, self.nlist, self.m, self.nbits = dim, metric, nlist, m, nbits
        met = faiss.METRIC_L2 if metric == "euclidean" else faiss.METRIC_INNER_PRODUCT
        quant = faiss.IndexFlatL2(dim) if metric == "euclidean" else faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFPQ(quant, dim, nlist, m, nbits, met)
        self.nprobe = self.DEFAULTS["nprobe"]

    def build(self, base):
        faiss.omp_set_num_threads(1)
        t0 = time.perf_counter()
        self.index.train(base)
        self.index.add(base)
        return time.perf_counter() - t0

    def set_query_param(self, nprobe):
        self.nprobe = int(nprobe)
        self.index.nprobe = self.nprobe

    def search(self, queries, k):
        faiss.omp_set_num_threads(1)
        times = []
        out = np.empty((queries.shape[0], k), dtype=np.int32)
        for i in range(queries.shape[0]):
            t0 = time.perf_counter()
            _, ids = self.index.search(queries[i:i + 1], k)
            times.append(time.perf_counter() - t0)
            out[i] = ids[0]
        return out, times

    def memory_bytes(self):
        return int(self.index.ntotal * (self.m * self.nbits / 8 + 8) + self.nlist * self.dim * 4)

    def params(self):
        return {"nlist": self.nlist, "m": self.m, "nbits": self.nbits, "nprobe": self.nprobe}
