"""
Trace từng bước forward() của C3k2 với shape cụ thể.
Mục tiêu: bạn nhìn vào output là hiểu ngay.
"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.common import C3, C3k2, Bottleneck

print("=" * 60)
print("  VÍ DỤ CỤ THỂ: C3k2(c1=128, c2=128, n=2, c3k=False)")
print("=" * 60)

# Tạo block
block = C3k2(c1=128, c2=128, n=2, c3k=False).eval()

# Input giả: batch=1, 128 channels, 40x40 pixels
x = torch.randn(1, 128, 40, 40)
print(f"\n--- INPUT ---")
print(f"  x.shape = {list(x.shape)}")
print(f"  Ý nghĩa: 1 ảnh, 128 kênh, 40×40 pixel")

# ========================================
# BƯỚC 1: cv1 - một Conv 1×1 duy nhất
# ========================================
print(f"\n--- BƯỚC 1: cv1 (Conv 1×1) ---")
print(f"  cv1 = Conv(128 → {2 * block.c}, kernel=1×1)")
out_cv1 = block.cv1(x)
print(f"  cv1(x).shape = {list(out_cv1.shape)}")
print(f"  → Tăng từ 128 lên {2*block.c} kênh (gấp đôi hidden channels)")

# ========================================
# BƯỚC 2: chunk - CẮT đôi tensor
# ========================================
print(f"\n--- BƯỚC 2: chunk(2, dim=1) - CẮT đôi theo chiều channel ---")
chunks = out_cv1.chunk(2, 1)
y = list(chunks)
print(f"  chunk(2, dim=1) tạo ra 2 tensor:")
print(f"    y[0].shape = {list(y[0].shape)}  ← nửa trên (giữ nguyên, KHÔNG xử lý)")
print(f"    y[1].shape = {list(y[1].shape)}  ← nửa dưới (sẽ đưa vào Bottleneck)")
print(f"  ")
print(f"  ⚡ chunk() KHÔNG copy data, chỉ tạo 2 'view' → gần như miễn phí")

# ========================================
# BƯỚC 3: Bottleneck lần lượt
# ========================================
print(f"\n--- BƯỚC 3: Đưa qua từng Bottleneck ---")
print(f"  Block có {len(block.m)} Bottleneck (n=2)")

for i, m in enumerate(block.m):
    input_tensor = y[-1]  # luôn lấy tensor CUỐI CÙNG trong list
    output_tensor = m(input_tensor)
    y.append(output_tensor)
    print(f"\n  Bottleneck {i}:")
    print(f"    Input:  y[{len(y)-2}].shape = {list(input_tensor.shape)}")
    print(f"    Output: y[{len(y)-1}].shape = {list(output_tensor.shape)}")
    print(f"    Bên trong Bottleneck:")
    print(f"      cv1: Conv 1×1 ({block.c} → {block.c})  - giữ nguyên channels")
    print(f"      cv2: Conv 3×3 ({block.c} → {block.c})  - trích xuất spatial features")
    print(f"      + residual shortcut (x + output)")

# ========================================
# BƯỚC 4: Xem list y chứa gì
# ========================================
print(f"\n--- BƯỚC 4: Xem list y sau khi chạy xong ---")
print(f"  y chứa {len(y)} tensor:")
for i, t in enumerate(y):
    if i == 0:
        label = "nửa trên từ chunk (bypass, KHÔNG xử lý)"
    elif i == 1:
        label = "nửa dưới từ chunk (input cho Bottleneck 0)"
    else:
        label = f"output của Bottleneck {i-2}"
    print(f"    y[{i}].shape = {list(t.shape)}  ← {label}")

# ========================================
# BƯỚC 5: Concatenate TẤT CẢ
# ========================================
print(f"\n--- BƯỚC 5: torch.cat(y, dim=1) - NỐI TẤT CẢ lại ---")
cat_result = torch.cat(y, 1)
print(f"  Nối {len(y)} tensor × {block.c} kênh = {len(y)} × {block.c} = {len(y) * block.c} kênh")
print(f"  cat(y).shape = {list(cat_result.shape)}")

# ========================================
# BƯỚC 6: cv2 - Conv 1×1 cuối
# ========================================
print(f"\n--- BƯỚC 6: cv2 (Conv 1×1 cuối) ---")
print(f"  cv2 = Conv({len(y) * block.c} → 128, kernel=1×1)")
output = block.cv2(cat_result)
print(f"  cv2(cat).shape = {list(output.shape)}")
print(f"  → Nén từ {len(y) * block.c} kênh về 128 kênh (kích thước output)")

# ========================================
# TỔNG KẾT
# ========================================
print(f"\n{'=' * 60}")
print(f"  TỔNG KẾT LUỒNG DỮ LIỆU")
print(f"{'=' * 60}")
print(f"""
  Input [1, 128, 40, 40]
    │
    ▼
  cv1: Conv 1×1 (128 → {2*block.c})
    │
    ▼
  [1, {2*block.c}, 40, 40]
    │
    ├── chunk ──┐
    │           │
    ▼           ▼
  y[0]        y[1]
  [{block.c}]       [{block.c}]
  (bypass)    │
    │         ▼
    │     Bottleneck 0
    │         │
    │         ▼
    │       y[2]
    │       [{block.c}]
    │         │
    │         ▼
    │     Bottleneck 1
    │         │
    │         ▼
    │       y[3]
    │       [{block.c}]
    │         │
    ▼         ▼
  ┌─────────────────────┐
  │ cat(y[0],y[1],y[2],y[3]) │
  │ = [{block.c} + {block.c} + {block.c} + {block.c}]        │
  │ = [{4*block.c}] channels       │
  └─────────────────────┘
    │
    ▼
  cv2: Conv 1×1 ({4*block.c} → 128)
    │
    ▼
  Output [1, 128, 40, 40]
""")

# ========================================
# SO SÁNH VỚI C3
# ========================================
print(f"{'=' * 60}")
print(f"  SO SÁNH: C3 chỉ giữ 2 tensor, C3k2 giữ 4 tensor")
print(f"{'=' * 60}")

c3 = C3(c1=128, c2=128, n=2).eval()
print(f"""
  C3 forward:
    cv1(x) → Bottleneck 0 → Bottleneck 1 → output_processed
    cv2(x) → output_bypass
    cat(output_processed, output_bypass) → 2 tensor × {c3.cv1.conv.out_channels} = {2 * c3.cv1.conv.out_channels} kênh
    cv3({2 * c3.cv1.conv.out_channels} → 128)

  C3k2 forward:
    cv1(x) → chunk thành y[0], y[1]
    y[1] → Bottleneck 0 → y[2]
    y[2] → Bottleneck 1 → y[3]
    cat(y[0], y[1], y[2], y[3]) → 4 tensor × {block.c} = {4*block.c} kênh
    cv2({4*block.c} → 128)

  KHÁC BIỆT CỐT LÕI:
  ┌─────────────────────────────────────────────────┐
  │ C3:   chỉ giữ OUTPUT CUỐI của chuỗi Bottleneck │
  │       → concat 2 tensor                         │
  │                                                  │
  │ C3k2: giữ TẤT CẢ output trung gian              │
  │       → concat (2 + n) tensor                    │
  │       → mỗi Bottleneck đều đóng góp vào kết quả │
  └─────────────────────────────────────────────────┘
""")
