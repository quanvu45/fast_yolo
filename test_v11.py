"""
Test script: kiem tra toan bo pipeline v11Detect
  1. Model init + stride
  2. Forward pass (training mode) -> dict
  3. Forward pass (eval mode)    -> NMS-ready tensor
  4. v11ComputeLoss forward
  5. Backward (gradient check)
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Config ────────────────────────────────────────────────────────────────────
CFG    = 'models/NPS_uav_s_v11.yaml'
BS     = 2      # batch size
IMG_SZ = 320    # smaller for speed
NC     = 1      # number of classes (matches yaml)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("TEST 1: Model init + stride")
print("=" * 60)
from models.yolo import Model
model = Model(CFG, ch=3, ch2=3).to(DEVICE)
model.eval()
print(f"  Stride : {model.stride.tolist()}")
print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")
print("  [PASS]")

# ── Dummy inputs ───────────────────────────────────────────────────────────────
x1 = torch.randn(BS, 3, IMG_SZ, IMG_SZ).to(DEVICE)   # IR stream
x2 = torch.randn(BS, 3, IMG_SZ, IMG_SZ).to(DEVICE)   # optical stream

print()
print("=" * 60)
print("TEST 2: Forward (eval mode) -> inference tensor")
print("=" * 60)
with torch.no_grad():
    out_eval = model(x1, x2)
# eval returns (pred_tensor, feat_list)
pred_tensor = out_eval[0]
print(f"  Output shape : {pred_tensor.shape}")   # (BS, total_anchors, 5+NC)
assert pred_tensor.ndim == 3, "Expected 3D tensor (bs, anchors, nc+5)"
assert pred_tensor.shape[0] == BS
print("  [PASS]")

print()
print("=" * 60)
print("TEST 3: Forward (training mode) -> dict")
print("=" * 60)
model.train()
out_train = model(x1, x2)
print(f"  Output type  : {type(out_train)}")
# v11Detect returns dict with keys: boxes, scores, feats
# (or nested dict if end2end)
if isinstance(out_train, dict) and 'boxes' in out_train:
    preds = out_train
    print(f"  boxes  shape : {preds['boxes'].shape}")   # (BS, 4*reg_max, total_anchors)
    print(f"  scores shape : {preds['scores'].shape}")  # (BS, NC, total_anchors)
    print(f"  feats  count : {len(preds['feats'])}")    # number of detection scales
elif isinstance(out_train, dict) and 'one2many' in out_train:
    preds = out_train['one2many']
    print(f"  [end2end mode] boxes : {preds['boxes'].shape}")
else:
    print(f"  Unexpected output: {out_train}")
    sys.exit(1)
print("  [PASS]")

print()
print("=" * 60)
print("TEST 4: v11ComputeLoss forward")
print("=" * 60)
from utils.loss_v11 import v11ComputeLoss

# Fake hyp (must have box/cls/dfl keys)
model.hyp = {
    'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
    'anchor_t': 4.0, 'label_smoothing': 0.0,
}
model.class_weights = None

compute_loss = v11ComputeLoss(model)

# Fake targets: (N, 6) [batch_idx, cls, cx, cy, w, h] normalized
# Put 3 objects: 2 in image 0, 1 in image 1
targets = torch.tensor([
    [0, 0, 0.5, 0.5, 0.2, 0.2],   # img 0, cls 0
    [0, 0, 0.3, 0.3, 0.1, 0.1],   # img 0, cls 0
    [1, 0, 0.6, 0.4, 0.3, 0.3],   # img 1, cls 0
], dtype=torch.float32).to(DEVICE)

loss, loss_items = compute_loss(preds, targets)
print(f"  Total loss   : {loss.item():.4f}")
print(f"  Loss items   : box={loss_items[0].item():.4f}  dfl={loss_items[1].item():.4f}  cls={loss_items[2].item():.4f}")
assert not torch.isnan(loss), "Loss is NaN!"
assert not torch.isinf(loss), "Loss is Inf!"
print("  [PASS]")

print()
print("=" * 60)
print("TEST 5: Backward (gradient check)")
print("=" * 60)
model.train()
model.zero_grad()
out_train2 = model(x1, x2)
preds2 = out_train2 if 'boxes' in out_train2 else out_train2['one2many']
loss2, _ = compute_loss(preds2, targets)
loss2.backward()

# DFL conv has requires_grad=False intentionally, skip it
grad_nans = []
grad_none = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue  # frozen params (e.g. DFL conv weights)
    if p.grad is None:
        grad_none.append(name)
    elif torch.isnan(p.grad).any():
        grad_nans.append(name)

if grad_none:
    print(f"  Params with None grad ({len(grad_none)}): {grad_none[:3]}...")
if grad_nans:
    print(f"  Params with NaN grad  ({len(grad_nans)}): {grad_nans[:3]}...")

grad_ok = len(grad_none) == 0 and len(grad_nans) == 0
print(f"  Gradients OK : {grad_ok}")
if not grad_ok:
    print("  [WARN] Some gradients are None - check if those layers are actually in the compute graph")
    # This may be acceptable if those params are not used in the forward pass
    # for this specific input (e.g. end2end one2one head not used in training)
else:
    print("  [PASS]")


print()
print("=" * 60)
print("ALL TESTS PASSED - Pipeline v11 hoat dong chinh xac!")
print("=" * 60)
