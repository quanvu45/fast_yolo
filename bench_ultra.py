"""Ultra-minimal benchmark: block-level only, small inputs."""
import sys, time, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.common import C3, C3k2, Bottleneck, Conv

print(f"PyTorch: {torch.__version__}")
print(f"Device: CPU\n")

# Test multiple configs
configs = [
    # (c_in, c_out, spatial, n, description)
    (16,  16,  160, 1, "Small (backbone start)"),
    (32,  32,  80,  1, "Medium-small"),
    (64,  64,  40,  1, "Medium n=1"),
    (128, 128, 20,  1, "Large n=1"),
    (128, 128, 20,  3, "Large n=3"),
    (256, 256, 10,  3, "Very large n=3"),
]

print(f"{'Config':30s} | {'Module':22s} | {'Params':>8s} | {'Mean ms':>8s} | {'Med ms':>8s}")
print(f"{'-'*30}-+-{'-'*22}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

for c_in, c_out, spatial, n, desc in configs:
    x = torch.randn(1, c_in, spatial, spatial)
    
    for name, cls, kw in [
        ("C3",              C3,   dict(c1=c_in, c2=c_out, n=n)),
        ("C3k2(c3k=False)", C3k2, dict(c1=c_in, c2=c_out, n=n, c3k=False)),
    ]:
        b = cls(**kw).eval()
        p = sum(pp.numel() for pp in b.parameters())
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                b(x)
        
        # Timed (30 runs)
        times = []
        with torch.no_grad():
            for _ in range(30):
                t0 = time.perf_counter()
                b(x)
                times.append((time.perf_counter() - t0) * 1000)
        
        times.sort()
        mean = sum(times) / len(times)
        med = times[len(times)//2]
        
        print(f"{desc:30s} | {name:22s} | {p:>8,} | {mean:>8.2f} | {med:>8.2f}")
    print()  # blank line between configs

# Also test full model param counts only (no inference)
print(f"\n{'='*50}")
print(f"  FULL MODEL PARAM COUNTS + GFLOPS")
print(f"{'='*50}")

from models.yolo import Model
for name, cfg in [
    ('C3 (NPS_uav_s)',      str(ROOT/'models'/'NPS_uav_s.yaml')),
    ('C3k2 (NPS_uav_s_v11)', str(ROOT/'models'/'NPS_uav_s_v11.yaml')),
]:
    model = Model(cfg).eval()
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {total_p:,} params")
    
    # Single forward pass timing
    x1 = torch.randn(1, 3, 320, 320)
    x2 = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        model(x1, x2)  # warmup
        t0 = time.perf_counter()
        for _ in range(5):
            model(x1, x2)
        elapsed = (time.perf_counter() - t0) * 1000 / 5
    print(f"  {name}: {elapsed:.1f} ms per inference (320x320, 5 runs avg)")
    del model

print("\nDONE!")
