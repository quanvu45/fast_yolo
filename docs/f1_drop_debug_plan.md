# Debug: F1 sụt từ 0.94 → 0.55 khi chuyển PT → TensorRT

## Tình trạng

| | Training (PT, FP32) | Val exp9 (TensorRT, FP16) |
|---|---|---|
| **F1 max** | **0.94** @ conf=0.343 | **0.55** @ conf=0.178 |
| Weights | `ARD100_mask32-640_uavs/best.pt` | `best.engine` |
| Config | `NPS_uav_s.yaml` | Cùng model |
| Batch size | 24 (train) | 8 (val) |
| Precision | FP32 | FP16 |
| Resolution | 640 | 640 |

## Confusion Matrix (exp9)

- Recall = **0.45** (chỉ detect 45% drone)
- **55% drone bị miss** (False Negative)
- FP rất thấp → model predict đúng class nhưng confidence quá thấp → miss nhiều

## Nguyên nhân có thể

### 1. 🔴 FP16 Precision Loss trong Concat3 (khả năng cao nhất)

Phép chia trong `Concat3` ([common.py L129](file:///d:/Algo_test_python/fast_yolo/models/common.py#L129)):
```python
weight = weight1 / weight2   # ← Khi weight2 ≈ 0, FP16 overflow/NaN
x2 = weight * x2
x1 = x1 * (2 - weight)
```
FP16 chỉ có 10-bit mantissa → khi `weight2` gần 0, phép chia tạo giá trị cực lớn hoặc NaN → phá hỏng attention fusion.

**Fix**: Thêm epsilon trước khi export:
```python
weight = weight1 / (weight2 + 1e-6)   # tránh chia cho 0
```

### 2. 🟡 TensorRT batch size mismatch

Nếu engine build với batch=1 nhưng val batch=8 → shape mismatch.

### 3. 🟡 TensorRT binding order sai

Engine có 2 input: `images1` (RGB) + `images2` (mask). Nếu binding order bị đảo → model nhận sai input.

### 4. 🟢 Resolution khi export khác với khi val

Engine build ở imgsz khác 640 → kết quả sai.

---

## Kế hoạch debug (thứ tự ưu tiên)

### Bước 1: Xác nhận .pt vẫn OK
```bash
python val.py --weights runs/train/ARD100_mask32-640_uavs/weights/best.pt --imgsz 640 --batch-size 8 --device 0
```
**Kỳ vọng**: F1 ≈ 0.94. Nếu KHÔNG → vấn đề ở dataset/val script, không phải TensorRT.

### Bước 2: Export lại engine FP32 (loại trừ precision)
```bash
python export.py --weights runs/train/ARD100_mask32-640_uavs/weights/best.pt --imgsz 640 --batch-size 8 --include engine
```
Rồi val:
```bash
python val.py --weights runs/train/ARD100_mask32-640_uavs/weights/best.engine --imgsz 640 --batch-size 8 --device 0
```
**Nếu F1 ≈ 0.94**: Vấn đề là FP16 → đi bước 3  
**Nếu F1 vẫn thấp**: Vấn đề ở export script hoặc binding → đi bước 4

### Bước 3: Fix FP16 — thêm epsilon vào Concat3
Sửa [common.py L129](file:///d:/Algo_test_python/fast_yolo/models/common.py#L129):
```diff
-        weight = (weight1/weight2)
+        weight = weight1 / (weight2 + 1e-6)
```
Sau đó:
```bash
# Re-train hoặc load weights rồi export lại
python export.py --weights best.pt --imgsz 640 --batch-size 8 --half --include engine
python val.py --weights best.engine --imgsz 640 --batch-size 8 --device 0
```

### Bước 4: Kiểm tra binding order
Thêm debug print vào [common.py L669-684](file:///d:/Algo_test_python/fast_yolo/models/common.py#L669-L684):
```python
# Sau khi load engine, in ra tên bindings
for i in range(num):
    name = model.get_tensor_name(i)
    shape = tuple(model.get_tensor_shape(name))
    print(f"  Binding {i}: {name}  shape={shape}")
```
Xác nhận:
- `images1` = RGB, shape `(batch, 3, 640, 640)`
- `images2` = Mask, shape `(batch, 3, 640, 640)`
- `output` = predictions

### Bước 5 (nếu cần): Export với dynamic batch
```bash
python export.py --weights best.pt --imgsz 640 --dynamic --half --include engine
```

---

## Files liên quan

| File | Đường dẫn |
|------|-----------|
| Model config | [NPS_uav_s.yaml](file:///d:/Algo_test_python/fast_yolo/models/NPS_uav_s.yaml) |
| Concat3 fusion | [common.py L119-134](file:///d:/Algo_test_python/fast_yolo/models/common.py#L119-L134) |
| TRT binding | [common.py L661-768](file:///d:/Algo_test_python/fast_yolo/models/common.py#L661-L768) |
| Training F1 | [F1_curve.png](file:///d:/Algo_test_python/fast_yolo/runs/train/ARD100_mask32-640_uavs/F1_curve.png) |
| Val exp9 F1 | [F1_curve.png](file:///d:/Algo_test_python/fast_yolo/runs/val/exp9/F1_curve.png) |
| PT weights | `runs/train/ARD100_mask32-640_uavs/weights/best.pt` |
| Engine weights | `runs/train/ARD100_mask32-640_uavs/weights/best.engine` |
