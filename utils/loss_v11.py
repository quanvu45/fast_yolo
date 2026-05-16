import torch
import torch.nn as nn
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox
from ultralytics.utils.ops import xywh2xyxy


class v11ComputeLoss:
    """
    Anchor-free loss dung TaskAlignedAssigner + DFL (CIoU + BCE cls).
    Tuong thich voi train.py cua YOLOv5 (tra ve (loss_sum, loss_3items)).
    """
    def __init__(self, model):
        device = next(model.parameters()).device
        h = model.hyp  # hyperparameters

        m = model.model[-1]  # v11Detect module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        targets: (N, 6) tensor [batch_idx, cls, cx, cy, w, h] normalized
        Returns: (batch_size, max_boxes, 5) [cls, x1, y1, x2, y2]
        """
        nl = targets.shape[0]
        if nl == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)

        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        for j in range(batch_size):
            matches = batch_idx == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]  # [cls, cx, cy, w, h]
        # Scale and convert xywh normalized -> xyxy pixel
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode DFL distribution -> xyxy bbox."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, targets):
        """
        preds: dict {'boxes', 'scores', 'feats'} from v11Detect (training=True).
        targets: (N, 6) [batch_idx, cls, cx, cy, w, h] normalized.
        Returns: (total_loss * bs, Tensor[lbox, ldfl, lcls]) matching YOLOv5 train.py.
        """
        lbox = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)
        ldfl = torch.zeros(1, device=self.device)

        # Permute to (bs, anchors, channels)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        # Build anchor points & stride tensor
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        anchor_points = anchor_points.to(self.device)
        stride_tensor = stride_tensor.to(self.device)

        imgsz = torch.tensor(
            preds["feats"][0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]

        # Preprocess targets
        gt = self.preprocess(
            targets.to(self.device), batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels = gt[..., :1]
        gt_bboxes = gt[..., 1:]
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode predicted boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # TaskAlignedAssigner
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss (BCE)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        lcls = bce_loss.sum() / target_scores_sum

        # Box + DFL loss
        if fg_mask.sum():
            lbox_val, ldfl_val = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            lbox = lbox_val
            ldfl = ldfl_val

        # Apply hyp gains
        box_gain = self.hyp.get('box', 7.5)
        cls_gain = self.hyp.get('cls', 0.5)
        dfl_gain = self.hyp.get('dfl', 1.5)

        lbox = lbox * box_gain
        lcls = lcls * cls_gain
        ldfl = ldfl * dfl_gain

        total = lbox + lcls + ldfl
        # Ensure all are 1D tensors for torch.cat
        items = torch.stack([lbox.view(1), ldfl.view(1), lcls.view(1)]).squeeze()
        # YOLOv5 train.py logs 3 items: box, obj_slot(dfl), cls
        return total * batch_size, items.detach()
