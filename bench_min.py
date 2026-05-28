"""Minimal benchmark: params + inference time only. No thop."""
import sys, time, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.yolo import Model

def bench_model(name, cfg, runs=20):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    model = Model(cfg).eval()
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_p:,}")
    
    x1 = torch.randn(1, 3, 640, 640)
    x2 = torch.randn(1, 3, 640, 640)
    
    # Warmup 5 runs
    with torch.no_grad():
        for _ in range(5):
            model(x1, x2)
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x1, x2)
            times.append((time.perf_counter() - t0) * 1000)
    
    times.sort()
    mean = sum(times) / len(times)
    med = times[len(times)//2]
    print(f"  Mean: {mean:.1f}ms | Median: {med:.1f}ms | Min: {times[0]:.1f}ms | Max: {times[-1]:.1f}ms")
    return total_p, mean, med

def bench_block(c, spatial, n, runs=50):
    from models.common import C3, C3k2
    x = torch.randn(1, c, spatial, spatial)
    
    print(f"\n  Block: c={c}, spatial={spatial}, n={n}")
    for name, cls, kw in [
        ("C3",              C3,   dict(c1=c, c2=c, n=n)),
        ("C3k2(c3k=False)", C3k2, dict(c1=c, c2=c, n=n, c3k=False)),
    ]:
        b = cls(**kw).eval()
        p = sum(pp.numel() for pp in b.parameters())
        with torch.no_grad():
            for _ in range(10): b(x)
        times = []
        with torch.no_grad():
            for _ in range(runs):
                t0 = time.perf_counter()
                b(x)
                times.append((time.perf_counter()-t0)*1000)
        times.sort()
        mean = sum(times)/len(times)
        med = times[len(times)//2]
        print(f"    {name:22s} | params={p:>8,} | mean={mean:.2f}ms | median={med:.2f}ms")

print("PyTorch:", torch.__version__)

# Full model benchmark
r = {}
for name, cfg in [
    ('C3',   str(ROOT/'models'/'NPS_uav_s.yaml')),
    ('C3k2', str(ROOT/'models'/'NPS_uav_s_v11.yaml')),
]:
    r[name] = bench_model(name, cfg, runs=20)

# Summary
p0,m0,md0 = r['C3']
p1,m1,md1 = r['C3k2']
print(f"\n{'='*50}")
print(f"  SUMMARY")
print(f"{'='*50}")
print(f"  {'':20s} | {'C3':>10s} | {'C3k2':>10s} | {'Diff':>12s}")
print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
print(f"  {'Params':20s} | {p0:>10,} | {p1:>10,} | {p1-p0:>+12,}")
print(f"  {'Mean (ms)':20s} | {m0:>10.1f} | {m1:>10.1f} | {m1-m0:>+8.1f}ms ({(m1-m0)/m0*100:+.1f}%)")
print(f"  {'Median (ms)':20s} | {md0:>10.1f} | {md1:>10.1f} | {md1-md0:>+8.1f}ms ({(md1-md0)/md0*100:+.1f}%)")

# Block-level benchmarks
print(f"\n{'='*50}")
print(f"  BLOCK-LEVEL BENCHMARKS")
print(f"{'='*50}")
for c,sp,n in [(64,80,1),(128,40,1),(256,20,3)]:
    bench_block(c, sp, n)

print(f"\nDONE!")
