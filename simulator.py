import math
import numpy as np
import time

class KVCacheTransformer:
    def __init__(self, dim: int):
        self.dim = dim
        self.keys = []
        self.values = []
        self.total_tokens = 0

    def write(self, key: np.ndarray, value: np.ndarray):
        self.keys.append(key / (np.linalg.norm(key) + 1e-9))
        self.values.append(value)
        self.total_tokens += 1

    def read(self, query: np.ndarray) -> np.ndarray:
        if not self.keys:
            return np.zeros(self.dim)
        K = np.vstack(self.keys)
        V = np.vstack(self.values)
        q = query / (np.linalg.norm(query) + 1e-9)

        scores = np.dot(K, q) / math.sqrt(self.dim)
        exp_scores = np.exp(scores - np.max(scores))
        attn_weights = exp_scores / (np.sum(exp_scores) + 1e-9)
        return np.dot(attn_weights, V)

    def memory_bytes(self) -> int:
        return 2 * self.total_tokens * self.dim * 4


class BDHSynapticMemory:
    def __init__(self, dim: int, decay: float = 0.9995, learning_rate: float = 1.0):
        self.dim = dim
        self.decay = decay
        self.lr = learning_rate
        self.W = np.zeros((dim, dim), dtype=np.float32)
        self.total_tokens = 0

    def write(self, key: np.ndarray, value: np.ndarray):
        k = key / (np.linalg.norm(key) + 1e-9)
        hebbian_update = np.outer(value, k)
        self.W = self.decay * self.W + self.lr * hebbian_update
        self.total_tokens += 1

    def read(self, query: np.ndarray) -> np.ndarray:
        q = query / (np.linalg.norm(query) + 1e-9)
        return np.dot(self.W, q)

    def memory_bytes(self) -> int:
        return self.dim * self.dim * 4


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def run_synthetic_benchmark():
    np.random.seed(42)
    dim = 64
    num_facts = 25
    test_lengths = [100, 1000, 10000, 50000]

    print("=" * 80)
    print("THE MEMORY DUEL: KV CACHE vs. BDH SYNAPTIC MEMORY BENCHMARK")
    print(f"Dimension: {dim} | Key-Value Facts Stored: {num_facts}")
    print("=" * 80)
    print(f"{'Total Seq Len':<15} | {'KV Memory':<14} | {'BDH Memory':<14} | {'KV Recall':<10} | {'BDH Recall':<10}")
    print("-" * 80)

    for total_len in test_lengths:
        kv_model = KVCacheTransformer(dim=dim)
        bdh_model = BDHSynapticMemory(dim=dim, decay=0.9998)

        keys = [np.random.randn(dim) for _ in range(num_facts)]
        values = [np.random.randn(dim) for _ in range(num_facts)]

        step_interval = max(1, total_len // num_facts)
        fact_idx = 0
        for t in range(total_len):
            if fact_idx < num_facts and (t % step_interval == 0 or t == total_len - 1):
                k, v = keys[fact_idx], values[fact_idx]
                kv_model.write(k, v)
                bdh_model.write(k, v)
                fact_idx += 1
            else:
                dk = np.random.randn(dim) * 0.02
                dv = np.random.randn(dim) * 0.02
                kv_model.write(dk, dv)
                bdh_model.write(dk, dv)

        query = keys[0]
        true_val = values[0]

        kv_out = kv_model.read(query)
        bdh_out = bdh_model.read(query)

        kv_sim = cosine_similarity(kv_out, true_val)
        bdh_sim = cosine_similarity(bdh_out, true_val)

        kv_mem_kb = kv_model.memory_bytes() / 1024
        bdh_mem_kb = bdh_model.memory_bytes() / 1024

        print(f"{total_len:<15} | {kv_mem_kb:>9.1f} KB   | {bdh_mem_kb:>9.1f} KB   | {kv_sim:>8.3f}   | {bdh_sim:>8.3f}")

    print("=" * 80)
    print("EMPIRICAL VERIFICATION COMPLETE:")
    print("1. Transformer KV Cache scales O(T) without bound.")
    print("2. BDH Synaptic Memory remains strictly fixed at 16.0 KB.")
    print("=" * 80)

if __name__ == '__main__':
    run_synthetic_benchmark()
