import torch
import time
import attention

device = "cuda"

configs = [
    # (batch, num_heads, seq_len, head_dim, is_causal)
    (2, 20, 16, 64, False),    # Audio encoder style, small seq
    (2, 20, 64, 64, False),    # Audio encoder style, medium seq
    (2, 20, 128, 64, False),   # Audio encoder style, larger seq
    (2, 20, 256, 64, False),   # Audio encoder style, max seq
    (1, 28, 16, 128, True),    # Text decoder style, small seq
    (1, 28, 64, 128, True),    # Text decoder style, medium seq
    (1, 28, 128, 128, True),   # Text decoder style, larger seq
    (1, 28, 256, 128, True),   # Text decoder style, max seq
]

print("=== FlashAttention vs Naive 3-Kernel ===")
print(f"{'Config':>40}  {'Flash':>10}  {'3-Kernel':>10}  {'Speedup':>8}")
print("-" * 78)

for batch, heads, seq, hdim, causal in configs:
    q = torch.randn(batch, heads, seq, hdim, device=device, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, hdim, device=device, dtype=torch.float32)
    v = torch.randn(batch, heads, seq, hdim, device=device, dtype=torch.float32)

    label = f"B={batch} H={heads} S={seq} D={hdim} {'causal' if causal else 'full'}"

    # Warmup both paths
    attention.USE_FLASH_ATTENTION = True
    for _ in range(3):
        _ = attention.scaled_dot_product_attention(q, k, v, is_causal=causal)
    attention.USE_FLASH_ATTENTION = False
    for _ in range(3):
        _ = attention.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.cuda.synchronize()

    # FlashAttention
    attention.USE_FLASH_ATTENTION = True
    times_flash = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_flash = attention.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_flash.append((t1 - t0) * 1000)

    # Naive 3-kernel
    attention.USE_FLASH_ATTENTION = False
    times_naive = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_naive = attention.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_naive.append((t1 - t0) * 1000)

    f = sum(times_flash) / len(times_flash)
    n = sum(times_naive) / len(times_naive)

    # Correctness check
    diff = (out_flash - out_naive).abs().max().item()

    print(f"{label:>40}  {f:>9.3f}ms  {n:>9.3f}ms  {n/f:>7.2f}x  (max_diff={diff:.2e})")

# Memory comparison
print("\n=== Memory Usage Comparison (B=2 H=20 S=256 D=64) ===")
q = torch.randn(2, 20, 256, 64, device=device, dtype=torch.float32)
k = torch.randn(2, 20, 256, 64, device=device, dtype=torch.float32)
v = torch.randn(2, 20, 256, 64, device=device, dtype=torch.float32)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
mem_before = torch.cuda.max_memory_allocated()
attention.USE_FLASH_ATTENTION = True
for _ in range(5):
    _ = attention.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
mem_flash = torch.cuda.max_memory_allocated() - mem_before

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
mem_before = torch.cuda.max_memory_allocated()
attention.USE_FLASH_ATTENTION = False
for _ in range(5):
    _ = attention.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
mem_naive = torch.cuda.max_memory_allocated() - mem_before

print(f"  FlashAttention peak alloc: {mem_flash / 1024:.1f} KB")
print(f"  3-Kernel peak alloc:       {mem_naive / 1024:.1f} KB")
print(f"  Memory saved:              {(mem_naive - mem_flash) / 1024:.1f} KB")
