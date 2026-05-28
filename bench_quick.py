"""
Quick benchmark C3 vs C3k2: params + FLOPs + inference time
Reduced iterations for CPU testing
"""
import sys, time, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.yolo import Model
from utils.torch_utils import select_device

try:
    import thop
except ImportError:
    thop = None


def bench_model(name, cfg, device, img_size=640, warmup=10, runs=30):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    model = Model(cfg).to(device).eval()
    
    # Params
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_p:,}")
    
    # FLOPs
    if thop:
        x1 = torch.zeros(1, 3, img_size, img_size).to(device)
        x2 = torch.zeros(1, 3, img_size, img_size).to(device)
        try:
            flops = thop.profile(model, inputs=(x1, x2), verbose=False)[0] / 1e9
            print(f"  GFLOPs: {flops:.2f}")
        except Exception as e:
            flops = None
            print(f"  GFLOPs: failed ({e})")
    else:
        flops = None
    
    # Inference
    x1 = torch.randn(1, 3, img_size, img_size).to(device)
    x2 = torch.randn(1, 3, img_size, img_size).to(device)
    
    with torch.no_grad():
        for _ in range(warmup):
            model(x1, x2)
    
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x1, x2)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    
    times.sort()
    trim = max(1, len(times) // 5)
    t_mean = sum(times[trim:-trim]) / len(times[trim:-trim])
    t_med = times[len(times) // 2]
    t_min = times[0]
    
    print(f"  Inference: mean={t_mean:.1f}ms, median={t_med:.1f}ms, min={t_min:.1f}ms")
    
    result = {'params': total_p, 'gflops': flops, 'mean': t_mean, 'median': t_med, 'min': t_min}
    del model
    return result


def bench_block_standalone(device, c=128, spatial=80, n=1, warmup=20, runs=100):
    from models.common import C3, C3k2
    
    x = torch.randn(1, c, spatial, spatial).to(device)
    
    print(f"\n  Block bench: c={c}, spatial={spatial}, n={n}")
    
    for name, cls, kw in [
        ("C3",              C3,   dict(c1=c, c2=c, n=n)),
        ("C3k2(c3k=False)", C3k2, dict(c1=c, c2=c, n=n, c3k=False)),
        ("C3k2(c3k=True)",  C3k2, dict(c1=c, c2=c, n=n, c3k=True)),
    ]:
        block = cls(**kw).to(device).eval()
        params = sum(p.numel() for p in block.parameters())
        
        flops_str = "N/A"
        if thop:
            try:
                f = thop.profile(block, inputs=(x.clone(),), verbose=False)[0] / 1e6
                flops_str = f"{f:.1f}M"
            except:
                pass
        
        with torch.no_grad():
            for _ in range(warmup):
                block(x)
        
        times = []
        with torch.no_grad():
            for _ in range(runs):
                t0 = time.perf_counter()
                block(x)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
        
        times.sort()
        trim = max(1, len(times) // 5)
        mean = sum(times[trim:-trim]) / len(times[trim:-trim])
        med = times[len(times) // 2]
        
        print(f"    {name:22s} | params={params:>8,} | FLOPs={flops_str:>10s} | mean={mean:.3f}ms | median={med:.3f}ms")
        del block


def main():
    device = select_device('cpu')
    print(f"Device: CPU")
    print(f"PyTorch: {torch.__version__}")
    
    cfgs = {
        'C3 (NPS_uav_s)':      str(ROOT / 'models' / 'NPS_uav_s.yaml'),
        'C3k2 (NPS_uav_s_v11)': str(ROOT / 'models' / 'NPS_uav_s_v11.yaml'),
    }
    
    results = {}
    for name, cfg in cfgs.items():
        results[name] = bench_model(name, cfg, device, warmup=10, runs=30)
    
    # Summary
    n = list(results.keys())
    r0, r1 = results[n[0]], results[n[1]]
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY: C3 vs C3k2")
    print(f"{'='*60}")
    print(f"  {'':25s} | {'C3':>12s} | {'C3k2':>12s} | {'Diff':>14s}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}")
    
    pd = r1['params'] - r0['params']
    print(f"  {'Params':25s} | {r0['params']:>12,} | {r1['params']:>12,} | {pd:>+14,}")
    
    if r0['gflops'] and r1['gflops']:
        fd = r1['gflops'] - r0['gflops']
        print(f"  {'GFLOPs':25s} | {r0['gflops']:>12.2f} | {r1['gflops']:>12.2f} | {fd:>+14.2f}")
    
    td = r1['mean'] - r0['mean']
    tp = td / r0['mean'] * 100
    print(f"  {'Mean inference (ms)':25s} | {r0['mean']:>12.1f} | {r1['mean']:>12.1f} | {td:>+10.1f}ms ({tp:+.1f}%)")
    
    td2 = r1['median'] - r0['median']
    tp2 = td2 / r0['median'] * 100
    print(f"  {'Median inference (ms)':25s} | {r0['median']:>12.1f} | {r1['median']:>12.1f} | {td2:>+10.1f}ms ({tp2:+.1f}%)")
    
    # Standalone block benchmarks
    print(f"\n{'='*60}")
    print(f"  STANDALONE BLOCK BENCHMARKS")
    print(f"{'='*60}")
    
    for c, sp, n in [(32, 160, 1), (64, 80, 1), (128, 40, 1), (128, 40, 3), (256, 20, 3)]:
        bench_block_standalone(device, c=c, spatial=sp, n=n, warmup=20, runs=100)
    
    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
