import time
import torch
from models.yolo import Model

def benchmark(model_cfg, name, device, runs=30, warmup=5):
    model = Model(model_cfg).to(device)
    model.eval()
    
    # Mock inputs
    x1 = torch.randn(1, 3, 640, 640).to(device)
    x2 = torch.randn(1, 3, 640, 640).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x1, x2)
            
    # Measure time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x1, x2)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / runs) * 1000  # ms
    fps = 1000 / avg_latency
    return avg_latency, fps

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}\n")
    
    print("Benchmarking NPS_uav_s (YOLOv5-based)...")
    lat_v5, fps_v5 = benchmark('models/NPS_uav_s.yaml', 'NPS_uav_s', device)
    
    print("\nBenchmarking NPS_uav_s_v11 (YOLOv11-based)...")
    lat_v11, fps_v11 = benchmark('models/NPS_uav_s_v11.yaml', 'NPS_uav_s_v11', device)
    
    fps_diff = fps_v11 - fps_v5
    fps_percent = (fps_diff / fps_v5) * 100
    
    print("\n" + "="*50)
    print("SPEED BENCHMARK RESULTS")
    print("="*50)
    print(f"NPS_uav_s (YOLOv5):     Latency: {lat_v5:.2f} ms | FPS: {fps_v5:.2f}")
    print(f"NPS_uav_s_v11 (YOLOv11): Latency: {lat_v11:.2f} ms | FPS: {fps_v11:.2f}")
    print(f"FPS Increase:            +{fps_diff:.2f} FPS ({fps_percent:+.2f}%)")
    print("="*50)
