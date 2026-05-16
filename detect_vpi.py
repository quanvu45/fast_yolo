"""
detect_vpi.py  –  Real-time dual-stream detection with GPU-accelerated motion mask.

Instead of loading pre-computed masks from disk, this script:
  1. Reads a VIDEO source (or image sequence)
  2. Computes the motion mask ON-THE-FLY using GMC_VPI (NVIDIA VPI on CUDA)
  3. Feeds both RGB + motion mask into the dual-stream YOLO model

Usage:
    python detect_vpi.py --weights best.pt --source video.mp4 --device 0
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

torch.backends.cudnn.benchmark = True

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (LOGGER, check_img_size, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args,
                           scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.gmc_vpi import create_gmc


# ══════════════════════════════════════════════════════════════
#  Video / image-sequence loader with frame buffering for GMC
# ══════════════════════════════════════════════════════════════
class VideoStreamGMC:
    """
    Reads frames from a video file and maintains a sliding window
    of past frames needed for the FD5-style motion mask
    (frames at t-4, t-2, t).
    """

    def __init__(self, source, img_size=640, stride=32, buf_size=5):
        self.source   = str(source)
        self.img_size = img_size if isinstance(img_size, int) else img_size[0]
        self.stride   = int(stride)
        self.buf_size = buf_size

        self.cap = cv2.VideoCapture(self.source)
        assert self.cap.isOpened(), f'Cannot open video: {self.source}'

        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n      = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w      = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h      = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.count  = 0

        # Ring buffer of grayscale frames for GMC
        self._buf = deque(maxlen=buf_size)

        LOGGER.info(
            f'VideoStreamGMC: {self.source}  '
            f'{self.w}×{self.h} @ {self.fps:.1f} FPS  ({self.n} frames)'
        )

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame_bgr = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration

        self.count += 1
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._buf.append(frame_gray)

        return frame_bgr, frame_gray

    @property
    def ready_for_gmc(self):
        """Need at least 5 frames (t-4 … t) for FD5-style mask."""
        return len(self._buf) >= self.buf_size

    def get_triplet(self):
        """
        Return the 3 frames used by FD5_mask:
          frame1 = t-4  (lastFrame1)
          frame2 = t-2  (lastFrame3)
          frame3 = t    (currentFrame)
        """
        buf = list(self._buf)
        return buf[-5], buf[-3], buf[-1]

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def __len__(self):
        return self.n


def preprocess_frame(img0_bgr, img_size, stride):
    """Letterbox + CHW + contiguous – same logic as LoadDualStream._preprocess."""
    h0, w0 = img0_bgr.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        img = cv2.resize(img0_bgr, (int(w0 * r), int(h0 * r)), interpolation=interp)
    else:
        img = img0_bgr

    h, w = img.shape[:2]
    img_lb, ratio, pad = letterbox(img, img_size, stride=stride, auto=False, scaleup=False)
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img_chw = np.ascontiguousarray(img_lb.transpose((2, 0, 1))[::-1])
    return img_chw, shapes


def mask_gray_to_3ch(mask_gray, img_size, stride):
    """Convert single-channel motion diff to 3-channel + letterbox for model input."""
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    img_chw, shapes = preprocess_frame(mask_bgr, img_size, stride)
    return img_chw, shapes


# ══════════════════════════════════════════════════════════════
#  Main inference loop
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def run(
    weights        = ROOT / 'runs/train/ARD100_mask32-640_uavs/weights/best.engine',
    source         = ROOT / 'test_video.mp4',
    batch_size     = 1,         # VPI-based GMC is frame-by-frame
    imgsz          = (640, 640),
    conf_thres     = 0.25,
    iou_thres      = 0.45,
    max_det        = 1000,
    device         = '',
    view_img       = True,
    save_txt       = False,
    save_conf      = False,
    save_crop      = False,
    nosave         = False,
    classes        = None,
    agnostic_nms   = False,
    augment        = False,
    visualize      = False,
    update         = False,
    project        = ROOT / 'runs/detect',
    name           = 'exp_vpi',
    exist_ok       = False,
    line_thickness = 3,
    hide_labels    = False,
    hide_conf      = False,
    half           = False,
    dnn            = False,
    gmc_gpu        = True,       # True → VPI/CUDA,  False → CPU fallback
    gmc_scale      = 2,
    show_mask      = False,      # Debug: also display computed motion mask
):
    source  = str(source)
    save_img = not nosave

    # ── Directories ──────────────────────────────────────────
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # ── Load YOLO model ─────────────────────────────────────
    device = select_device(device)
    model  = DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz  = check_img_size(imgsz, s=stride)

    if pt:
        half &= device.type != 'cpu'
        model.model.half() if half else model.model.float()

    # ── GMC engine ───────────────────────────────────────────
    gmc = create_gmc(prefer_gpu=gmc_gpu, scale=gmc_scale)
    LOGGER.info(f'GMC engine: {type(gmc).__name__}')

    # ── Video stream ─────────────────────────────────────────
    stream = VideoStreamGMC(source, img_size=imgsz, stride=stride, buf_size=5)
    model.warmup(imgsz=(1 if pt else batch_size, 3, *imgsz))

    # ── State ────────────────────────────────────────────────
    dt             = [0.0, 0.0, 0.0, 0.0]   # preprocess, gmc, inference, nms
    seen           = 0
    WINDOW         = 'Drone Detection (VPI)'
    vid_writer     = None
    vid_path_saved = None
    save_path      = str(save_dir / 'output.mp4')

    # ══════════════════════════════════════════════════════════
    #  Frame loop
    # ══════════════════════════════════════════════════════════
    for frame_bgr, frame_gray in stream:
        if not stream.ready_for_gmc:
            continue   # accumulate initial buffer

        # ── 1. Compute motion mask on GPU ────────────────────
        t0 = time_sync()
        f1, f2, f3 = stream.get_triplet()
        motion_mask = gmc.compute_fd5_mask(f1, f2, f3)   # (H, W) uint8
        t1 = time_sync()
        dt[1] += t1 - t0

        # ── 2. Preprocess RGB + mask ─────────────────────────
        img_rgb_chw, shapes = preprocess_frame(frame_bgr, imgsz[0], stride)
        img_mask_chw, _     = mask_gray_to_3ch(motion_mask, imgsz[0], stride)

        im  = torch.from_numpy(np.ascontiguousarray(img_rgb_chw[None])).to(device, non_blocking=True)
        im2 = torch.from_numpy(np.ascontiguousarray(img_mask_chw[None])).to(device, non_blocking=True)
        im  = im.half()  if half else im.float()
        im2 = im2.half() if half else im2.float()
        im  /= 255.0
        im2 /= 255.0
        t2 = time_sync()
        dt[0] += t2 - t1

        # ── 3. Inference ─────────────────────────────────────
        pred = model(im, im2, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[2] += t3 - t2

        # ── 4. NMS ───────────────────────────────────────────
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t4 = time_sync()
        dt[3] += t4 - t3

        seen += 1
        infer_ms = (dt[2] / seen) * 1e3
        gmc_ms   = (dt[1] / seen) * 1e3
        cur_fps  = 1000.0 / infer_ms if infer_ms > 0 else 0.0

        # ── 5. Draw ──────────────────────────────────────────
        det = pred[0]
        im0 = frame_bgr.copy()
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        s = f'frame {stream.count} — '

        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for c in det[:, -1].unique():
                n  = (det[:, -1] == c).sum()
                s += f'{n} {names[int(c)]}{"s" * (n > 1)}, '

            for *xyxy, conf, cls in reversed(det):
                if save_txt:
                    gn   = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    txt_path = str(save_dir / 'labels' / f'frame_{stream.count:06d}')
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                c_int = int(cls)
                label = None if hide_labels else (names[c_int] if hide_conf else f'{names[c_int]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c_int, True))

                if save_crop:
                    save_one_box(xyxy, im0.copy(),
                                 file=save_dir / 'crops' / names[c_int] / f'frame_{stream.count:06d}.jpg', BGR=True)

        im0 = annotator.result()

        # Overlay stats
        cv2.putText(im0, f'FPS: {cur_fps:.1f}  |  GMC: {gmc_ms:.1f}ms',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # ── 6. Display ───────────────────────────────────────
        if view_img:
            cv2.imshow(WINDOW, im0)
            if show_mask:
                cv2.imshow('Motion Mask (GMC)', motion_mask)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                LOGGER.info('User requested exit.')
                break

        # ── 7. Save video ────────────────────────────────────
        if save_img:
            if vid_path_saved != save_path:
                vid_path_saved = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                h_out, w_out = im0.shape[:2]
                vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*'mp4v'),
                    stream.fps, (w_out, h_out)
                )
            vid_writer.write(im0)

        LOGGER.info(f'{s}Done. (GMC {gmc_ms:.1f}ms | Infer {infer_ms:.1f}ms | FPS {cur_fps:.1f})')

    # ── Cleanup ───────────────────────────────────────────────
    stream.release()
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
    cv2.destroyAllWindows()

    # ── Final stats ───────────────────────────────────────────
    if seen:
        t = tuple(x / seen * 1e3 for x in dt)
        LOGGER.info(f'Speed: {t[0]:.1f}ms preprocess | {t[1]:.1f}ms GMC | '
                    f'{t[2]:.1f}ms inference | {t[3]:.1f}ms NMS  (per frame)')
        LOGGER.info(f'FPS inference: {1000/t[2]:.1f}  |  FPS full pipeline: {1000/sum(t):.1f}')
    if save_txt or save_img:
        LOGGER.info(f'Results saved to {colorstr("bold", save_dir)}')
    if update:
        strip_optimizer(weights)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'runs/train/ARD100_mask32-1280_uavs/weights/best.pt')
    parser.add_argument('--source',  type=str, default=ROOT / 'test_video.mp4')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280])
    parser.add_argument('--conf-thres',     type=float, default=0.25)
    parser.add_argument('--iou-thres',      type=float, default=0.45)
    parser.add_argument('--max-det',        type=int,   default=1000)
    parser.add_argument('--device',         default='')
    parser.add_argument('--view-img',       action='store_true', default=True)
    parser.add_argument('--save-txt',       action='store_true')
    parser.add_argument('--save-conf',      action='store_true')
    parser.add_argument('--save-crop',      action='store_true')
    parser.add_argument('--nosave',         action='store_true')
    parser.add_argument('--classes',        nargs='+', type=int)
    parser.add_argument('--agnostic-nms',   action='store_true')
    parser.add_argument('--augment',        action='store_true')
    parser.add_argument('--visualize',      action='store_true')
    parser.add_argument('--update',         action='store_true')
    parser.add_argument('--project',        default=ROOT / 'runs/detect')
    parser.add_argument('--name',           default='exp_vpi')
    parser.add_argument('--exist-ok',       action='store_true')
    parser.add_argument('--line-thickness', type=int, default=3)
    parser.add_argument('--hide-labels',    action='store_true')
    parser.add_argument('--hide-conf',      action='store_true')
    parser.add_argument('--half',           action='store_true')
    parser.add_argument('--dnn',            action='store_true')
    # GMC-specific options
    parser.add_argument('--gmc-gpu',        action='store_true', default=True,
                        help='Use VPI/CUDA for motion mask (default: True)')
    parser.add_argument('--gmc-cpu',        action='store_true',
                        help='Force CPU fallback for motion mask')
    parser.add_argument('--gmc-scale',      type=int, default=2,
                        help='Internal GMC processing scale factor')
    parser.add_argument('--show-mask',      action='store_true',
                        help='Display motion mask window for debugging')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    if opt.gmc_cpu:
        opt.gmc_gpu = False
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**{k: v for k, v in vars(opt).items() if k != 'gmc_cpu'})


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
