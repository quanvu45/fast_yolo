"""
Benchmark C3 (NPS_uav_s) vs C3k2 (NPS_uav_s_v11)
Measures: parameter count, GFLOPs, actual inference time
"""
import sys
import os
import time
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.yolo import Model
from utils.torch_utils import select_device, time_sync

try:
    import thop
except ImportError:
    thop = None
    print("WARNING: thop not installed, FLOPs will not be computed. Install with: pip install thop")


def count_params(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, device, img_size=640):
    """Count FLOPs using thop"""
    if thop is None:
        return None
    model.eval()
    x1 = torch.zeros(1, 3, img_size, img_size).to(device)
    x2 = torch.zeros(1, 3, img_size, img_size).to(device)
    try:
        flops = thop.profile(model, inputs=(x1, x2), verbose=False)[0]
        return flops / 1e9  # GFLOPs
    except Exception as e:
        print(f"  FLOPs computation failed: {e}")
        return None


def benchmark_inference(model, device, img_size=640, warmup=50, runs=200):
    """Benchmark actual inference time"""
    model.eval()
    x1 = torch.randn(1, 3, img_size, img_size).to(device)
    x2 = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x1, x2)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x1, x2)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
    
    times = sorted(times)
    # Remove top/bottom 10% outliers
    trim = max(1, len(times) // 10)
    times_trimmed = times[trim:-trim]
    
    return {
        'mean': sum(times_trimmed) / len(times_trimmed),
        'min': min(times),
        'max': max(times),
        'median': times[len(times) // 2],
        'p95': times[int(len(times) * 0.95)],
    }


def count_module_types(model):
    """Count occurrences of each module type"""
    from models.common import C3, C3k2, C2f, Bottleneck, Conv
    counts = {}
    for name, m in model.named_modules():
        type_name = type(m).__name__
        if type_name in ['C3', 'C3k2', 'C2f', 'Bottleneck', 'Conv', 'SPPF', 'CBAM', 'Concat3', 'Detect']:
            counts[type_name] = counts.get(type_name, 0) + 1
    return counts


def benchmark_block_standalone(device, c_in=128, c_out=128, n=1, img_size=80, warmup=100, runs=500):
    """Benchmark C3 vs C3k2 block in isolation"""
    from models.common import C3, C3k2
    
    print(f"\n{'='*70}")
    print(f"  STANDALONE BLOCK BENCHMARK (c_in={c_in}, c_out={c_out}, n={n}, spatial={img_size}x{img_size})")
    print(f"{'='*70}")
    
    x = torch.randn(1, c_in, img_size, img_size).to(device)
    
    results = {}
    for name, block_cls, kwargs in [
        ("C3",   C3,   dict(c1=c_in, c2=c_out, n=n)),
        ("C3k2 (c3k=False)", C3k2, dict(c1=c_in, c2=c_out, n=n, c3k=False)),
        ("C3k2 (c3k=True)",  C3k2, dict(c1=c_in, c2=c_out, n=n, c3k=True)),
    ]:
        block = block_cls(**kwargs).to(device).eval()
        params = sum(p.numel() for p in block.parameters())
        
        # FLOPs
        flops = None
        if thop:
            try:
                flops = thop.profile(block, inputs=(x,), verbose=False)[0] / 1e6  # MFLOPs
            except:
                pass
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = block(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timed
        times = []
        with torch.no_grad():
            for _ in range(runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = block(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
        
        times = sorted(times)
        trim = max(1, len(times) // 10)
        times_trimmed = times[trim:-trim]
        mean_ms = sum(times_trimmed) / len(times_trimmed)
        median_ms = times[len(times) // 2]
        
        results[name] = {'params': params, 'flops': flops, 'mean_ms': mean_ms, 'median_ms': median_ms}
        
        flops_str = f"{flops:.2f} MFLOPs" if flops else "N/A"
        print(f"  {name:25s} | Params: {params:>8,} | {flops_str:>16s} | Mean: {mean_ms:.3f} ms | Median: {median_ms:.3f} ms")
    
    return results


def main():
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    
    configs = {
        'C3 (NPS_uav_s)':     str(ROOT / 'models' / 'NPS_uav_s.yaml'),
        'C3k2 (NPS_uav_s_v11)': str(ROOT / 'models' / 'NPS_uav_s_v11.yaml'),
    }
    
    # =============================
    # PART 1: Full Model Comparison
    # =============================
    print(f"\n{'='*70}")
    print(f"  FULL MODEL COMPARISON")
    print(f"{'='*70}")
    
    all_results = {}
    for name, cfg_path in configs.items():
        print(f"\n--- {name} ---")
        print(f"  Config: {cfg_path}")
        
        try:
            model = Model(cfg_path).to(device)
        except Exception as e:
            print(f"  ERROR building model: {e}")
            continue
        
        # Params
        total_p, train_p = count_params(model)
        print(f"  Total params:     {total_p:>12,}")
        print(f"  Trainable params: {train_p:>12,}")
        
        # Module counts
        mcounts = count_module_types(model)
        print(f"  Module counts: {mcounts}")
        
        # FLOPs
        gflops = count_flops(model, device)
        if gflops is not None:
            print(f"  GFLOPs:           {gflops:>12.2f}")
        
        # Inference timing
        print(f"  Running inference benchmark (warmup=50, runs=200)...")
        timing = benchmark_inference(model, device, img_size=640, warmup=50, runs=200)
        print(f"  Inference time:   mean={timing['mean']:.2f} ms, median={timing['median']:.2f} ms, "
              f"p95={timing['p95']:.2f} ms, min={timing['min']:.2f} ms")
        
        all_results[name] = {
            'params': total_p,
            'gflops': gflops,
            'timing': timing,
        }
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Summary
    if len(all_results) == 2:
        names = list(all_results.keys())
        r0, r1 = all_results[names[0]], all_results[names[1]]
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Metric':<25s} | {'C3':>15s} | {'C3k2':>15s} | {'Diff':>15s}")
        print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}")
        
        p_diff = r1['params'] - r0['params']
        print(f"  {'Params':<25s} | {r0['params']:>15,} | {r1['params']:>15,} | {p_diff:>+15,}")
        
        if r0['gflops'] and r1['gflops']:
            f_diff = r1['gflops'] - r0['gflops']
            print(f"  {'GFLOPs':<25s} | {r0['gflops']:>15.2f} | {r1['gflops']:>15.2f} | {f_diff:>+15.2f}")
        
        t_diff = r1['timing']['mean'] - r0['timing']['mean']
        t_pct = (t_diff / r0['timing']['mean']) * 100
        print(f"  {'Inference mean (ms)':<25s} | {r0['timing']['mean']:>15.2f} | {r1['timing']['mean']:>15.2f} | {t_diff:>+12.2f} ms ({t_pct:+.1f}%)")
        
        t_diff_med = r1['timing']['median'] - r0['timing']['median']
        t_pct_med = (t_diff_med / r0['timing']['median']) * 100
        print(f"  {'Inference median (ms)':<25s} | {r0['timing']['median']:>15.2f} | {r1['timing']['median']:>15.2f} | {t_diff_med:>+12.2f} ms ({t_pct_med:+.1f}%)")
    
    # =============================
    # PART 2: Standalone Block Comparison
    # =============================
    for c, spatial, n in [(64, 160, 1), (128, 80, 1), (256, 40, 3), (512, 20, 3)]:
        benchmark_block_standalone(device, c_in=c, c_out=c, n=n, img_size=spatial)


if __name__ == '__main__':
    main()
