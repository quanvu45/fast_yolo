import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
from ultralytics.utils.ops import xywh2xyxy

class v11ComputeLoss(v8DetectionLoss):
    def __init__(self, model):
        # We manually initialize without calling super() to bypass model.args requirement
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        m = model.model[-1]  # v11Detect module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

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
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        targets: (N, 6) [batch_idx, cls, x, y, w, h] normalized
        scale_tensor: [w, h, w, h] to scale to pixel space
        """
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            
            for j in range(batch_size):
                matches = batch_idx == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
                    
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def __call__(self, preds, targets):
        """
        preds: dict containing 'boxes', 'scores', 'feats' from v11Detect
        targets: (N, 6) [batch_idx, cls, x, y, w, h]
        """
        # Formulate batch dict expected by get_assigned_targets_and_loss
        batch = {
            "batch_idx": targets[:, 0],
            "cls": targets[:, 1],
            "bboxes": targets[:, 2:6]
        }
        
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        
        # loss_detach is (box_loss, cls_loss, dfl_loss)
        # YOLOv5 train.py logs 3 elements, so we just return them as expected
        batch_size = preds["boxes"].shape[0]
        
        # YOLOv5 averages loss by batch_size before backward?
        # In YOLOv5, ComputeLoss returns (loss, torch.cat((lbox, lobj, lcls)).detach())
        # The returned loss is already scaled by batch_size in ComputeLoss of YOLOv5!
        # Actually YOLOv5 returns loss * batch_size.
        
        return loss.sum() * batch_size, loss_detach
