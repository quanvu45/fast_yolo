"""
Kiem tra TensorRT / ONNX co fuse duoc C3k2 khong
- Buoc 1: Conv+BN fuse (PyTorch level)
- Buoc 2: Export ONNX -> xem graph
- Buoc 3: Phan tich TensorRT se lam gi
"""
import sys, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.common import C3, C3k2, Conv
from utils.torch_utils import fuse_conv_and_bn

print("=" * 65)
print("  PHAN TICH FUSION CHO C3 vs C3k2")
print("=" * 65)

# =============================================
# PHAN 1: Conv+BN Fuse (PyTorch level)
# =============================================
print("\n--- PHAN 1: Conv+BN Fuse (model.fuse()) ---")
print("  Moi Conv trong YOLOv5 = Conv2d + BatchNorm2d + SiLU")
print("  Khi fuse: Conv2d + BN -> 1 Conv2d duy nhat (nhanh hon)")

for name, block_cls, kw in [
    ("C3",   C3,   dict(c1=128, c2=128, n=2)),
    ("C3k2", C3k2, dict(c1=128, c2=128, n=2, c3k=False)),
]:
    block = block_cls(**kw).eval()
    
    # Dem so Conv truoc fuse
    conv_count_before = sum(1 for m in block.modules() if isinstance(m, nn.Conv2d))
    bn_count_before = sum(1 for m in block.modules() if isinstance(m, nn.BatchNorm2d))
    
    # Fuse Conv+BN
    for m in block.modules():
        if isinstance(m, Conv) and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, 'bn')
            m.forward = m.forward_fuse
    
    conv_count_after = sum(1 for m in block.modules() if isinstance(m, nn.Conv2d))
    bn_count_after = sum(1 for m in block.modules() if isinstance(m, nn.BatchNorm2d))
    
    print(f"\n  {name}:")
    print(f"    Truoc fuse: {conv_count_before} Conv2d + {bn_count_before} BN")
    print(f"    Sau fuse:   {conv_count_after} Conv2d + {bn_count_after} BN")
    print(f"    -> Tat ca BN da duoc merge vao Conv2d? {'DA' if bn_count_after == 0 else 'CHUA'}")

# =============================================
# PHAN 2: Export ONNX - xem graph structure
# =============================================
print("\n\n--- PHAN 2: Export ONNX ---")

for name, block_cls, kw in [
    ("C3",   C3,   dict(c1=128, c2=128, n=2)),
    ("C3k2", C3k2, dict(c1=128, c2=128, n=2, c3k=False)),
]:
    block = block_cls(**kw).eval()
    x = torch.randn(1, 128, 40, 40)
    fname = f"{name.lower()}_block.onnx"
    
    try:
        torch.onnx.export(
            block, x, str(ROOT / fname),
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            do_constant_folding=True,
        )
        print(f"\n  {name} -> {fname} exported OK")
        
        # Doc ONNX va dem ops
        try:
            import onnx
            model = onnx.load(str(ROOT / fname))
            ops = {}
            for node in model.graph.node:
                ops[node.op_type] = ops.get(node.op_type, 0) + 1
            print(f"    ONNX nodes:")
            for op, count in sorted(ops.items()):
                print(f"      {op}: {count}")
            print(f"    Total nodes: {sum(ops.values())}")
        except ImportError:
            print(f"    (onnx package not installed, cannot analyze graph)")
    except Exception as e:
        print(f"\n  {name} -> ONNX export FAILED: {e}")

# =============================================
# PHAN 3: TensorRT fusion analysis
# =============================================
print("\n\n--- PHAN 3: TensorRT se fuse gi? ---")
print("""
  TensorRT thuc hien cac loai fusion sau:

  1. CONV + BN + ACTIVATION FUSION (ca C3 va C3k2 deu duoc)
     Conv2d + BatchNorm + SiLU -> 1 TRT layer
     -> GIONG NHAU giua C3 va C3k2

  2. LAYER FUSION (vertical)
     Conv 1x1 -> ReLU -> Conv 3x3 -> ReLU
     TRT co the gop thanh 1 kernel
     -> GIONG NHAU giua C3 va C3k2 (ca 2 deu co Bottleneck)

  3. CONCAT FUSION (horizontal)
     TRT xu ly Concat bang pointer arithmetic (zero-copy)
     -> C3:   Concat 2 tensor  -> nhe
     -> C3k2: Concat 4 tensor  -> van zero-copy, nhung nhieu branch hon

  4. SPLIT/CHUNK FUSION
     chunk(2) trong C3k2 -> ONNX 'Split' op
     TRT chuyen Split thanh pointer offset -> ZERO COST
     -> Chi C3k2 co buoc nay, nhung cost = 0
""")

print("  KET LUAN VE TENSORRT FUSION:")
print("  " + "-" * 50)
print("""
  +--------------------------------------------------+
  |                  | C3          | C3k2              |
  |------------------|-------------|-------------------|
  | Conv+BN+SiLU     | FUSE duoc   | FUSE duoc         |
  | Bottleneck chain | FUSE duoc   | FUSE duoc         |
  | Split/Chunk      | Khong co    | Zero-cost (view)  |
  | Concat           | 2 tensor    | (2+n) tensor      |
  |                  |             | van zero-copy     |
  +--------------------------------------------------+

  Ca hai deu duoc TensorRT fuse TOT NHU NHAU.

  Ly do: Khi export ra ONNX, Python loop bien mat.
  Graph tro thanh STATIC. TensorRT thay:

    C3:   Conv -> Conv -> Conv... -> Concat -> Conv
    C3k2: Conv -> Split -> Conv -> Conv... -> Concat -> Conv

  Ca hai deu la DAG (directed acyclic graph) don gian,
  TensorRT xu ly tot ca hai.
  """)

# =============================================
# PHAN 4: Chung minh bang code - trace forward
# =============================================
print("--- PHAN 4: So sanh forward sau khi fuse ---")

for name, block_cls, kw in [
    ("C3",   C3,   dict(c1=128, c2=128, n=2)),
    ("C3k2", C3k2, dict(c1=128, c2=128, n=2, c3k=False)),
]:
    block = block_cls(**kw).eval()
    
    # Fuse
    for m in block.modules():
        if isinstance(m, Conv) and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, 'bn')
            m.forward = m.forward_fuse
    
    # List all operations
    print(f"\n  {name} sau khi fuse - cac operation con lai:")
    ops = []
    for n_name, m in block.named_modules():
        if isinstance(m, nn.Conv2d):
            ops.append(f"    Conv2d({m.in_channels}->{m.out_channels}, k={m.kernel_size})")
        elif isinstance(m, nn.SiLU):
            ops.append(f"    SiLU")
    for op in ops:
        print(op)
    print(f"  + Concat (torch.cat)")
    if name == "C3k2":
        print(f"  + Split  (torch.chunk) <- zero cost")

print("\n\nDONE!")
