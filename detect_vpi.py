import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import threading
from queue import Queue

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

IMG_FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')


# ══════════════════════════════════════════════════════════════
#  Helper Functions
# ══════════════════════════════════════════════════════════════
def preprocess_frame(img0_bgr, img_size, stride):
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
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    img_chw, shapes = preprocess_frame(mask_bgr, img_size, stride)
    return img_chw, shapes


# ══════════════════════════════════════════════════════════════
#  Image Sequence Loader (DataLoader with num_workers)
# ══════════════════════════════════════════════════════════════
class LoadGMCStream(Dataset):
    def __init__(self, path_rgb, img_size=640, stride=32, gmc_gpu=False, gmc_scale=0.5):
        self.img_size = img_size if isinstance(img_size, int) else img_size[0]
        self.stride   = int(stride)
        self.gmc_gpu  = gmc_gpu
        self.gmc_scale= gmc_scale
        self.im_files = self._load_paths(path_rgb)
        self.n = len(self.im_files)
        LOGGER.info(f'LoadGMCStream: {self.n} frames loaded from {path_rgb}')

    @staticmethod
    def _load_paths(path):
        p = Path(path)
        files = []

        if p.is_file() and p.suffix.lower() == '.txt':
            with open(p) as f:
                lines = f.read().strip().splitlines()
            parent = str(p.parent) + os.sep
            for x in lines:
                x = x.strip()
                if not x: continue
                if x.startswith('./'):
                    x = x.replace('./', parent, 1)
                files.append(x)
        elif p.is_dir():
            files = glob.glob(str(p / '**' / '*.*'), recursive=True)
        elif '%' in str(p):
            dir_path = p.parent
            files = glob.glob(str(dir_path / '**' / '*.*'), recursive=True)
        else:
            raise FileNotFoundError(f'Không tồn tại: {path}')

        im_files = sorted(
            x.replace('/', os.sep) for x in files
            if x.split('.')[-1].lower() in IMG_FORMATS
        )
        assert im_files, f'Không tìm thấy ảnh nào trong: {path}'
        return im_files

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        # To match generate_mask5.py logic exactly:
        #   mask[N] = FD5(f_{N-2}, f_N, f_{N+2})  → reference is f_N (middle)
        # Real-time equivalent (no future frame):
        #   mask computed from (f_{N-4}, f_{N-2}, f_N) → reference is f_{N-2}
        #   → pair this mask with RGB frame at index N-2
        # So: use current index as the NEWEST frame, display frame at index-2
        idx_newest = index                    # f_N  (newest, used for optical flow)
        idx_mid    = max(0, index - 2)        # f_{N-2}  (middle / reference)
        idx_oldest = max(0, index - 4)        # f_{N-4}  (oldest)

        f1 = cv2.imread(self.im_files[idx_oldest], cv2.IMREAD_GRAYSCALE)
        f2 = cv2.imread(self.im_files[idx_mid],    cv2.IMREAD_GRAYSCALE)
        f3 = cv2.imread(self.im_files[idx_newest], cv2.IMREAD_GRAYSCALE)

        # RGB frame to DISPLAY / INFER is the middle frame (reference)
        img0_bgr = cv2.imread(self.im_files[idx_mid])

        # Initialize GMC context per worker
        gmc = create_gmc(prefer_gpu=self.gmc_gpu, scale=self.gmc_scale, silent=True)
        t0 = time.time()
        motion_mask = gmc.compute_fd5_mask(f1, f2, f3)
        gmc_time = (time.time() - t0) * 1000

        img_rgb_chw, shapes = preprocess_frame(img0_bgr, self.img_size, self.stride)
        img_mask_chw, _ = mask_gray_to_3ch(motion_mask, self.img_size, self.stride)

        return (self.im_files[idx_mid], img_rgb_chw, img0_bgr, shapes, img_mask_chw, motion_mask, gmc_time)

    @staticmethod
    def collate_fn(batch):
        paths, img_rgb, im0_rgb, shapes, img_mask, motion_mask, gmc_time = zip(*batch)
        img_rgb  = np.stack(img_rgb, 0)
        img_mask = np.stack(img_mask, 0)
        return (list(paths), img_rgb, list(im0_rgb), list(shapes), img_mask, list(motion_mask), list(gmc_time))


# ══════════════════════════════════════════════════════════════
#  Video Stream Loader (Async Threading for videos)
# ══════════════════════════════════════════════════════════════
class VideoStreamGMC:
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
        self._buf = deque(maxlen=buf_size)

        LOGGER.info(f'VideoStreamGMC: {self.source} {self.w}x{self.h} @ {self.fps:.1f} FPS')

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
        return len(self._buf) >= self.buf_size

    def get_triplet(self):
        buf = list(self._buf)
        return buf[-5], buf[-3], buf[-1]

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

class AsyncGMCStream:
    def __init__(self, stream, gmc, img_size, stride):
        self.stream = stream
        self.gmc = gmc
        self.img_size = img_size
        self.stride = stride
        self.queue = Queue(maxsize=3)
        self.stopped = False
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        for frame_bgr, frame_gray in self.stream:
            if self.stopped: break
            if not self.stream.ready_for_gmc: continue
            
            f1, f2, f3 = self.stream.get_triplet()
            t0 = time.time()
            motion_mask = self.gmc.compute_fd5_mask(f1, f2, f3)
            gmc_time = (time.time() - t0) * 1000
            
            img_rgb_chw, shapes = preprocess_frame(frame_bgr, self.img_size, self.stride)
            img_mask_chw, _ = mask_gray_to_3ch(motion_mask, self.img_size, self.stride)

            if not self.stopped:
                self.queue.put((
                    [f'frame_{self.stream.count:06d}'],
                    np.stack([img_rgb_chw], 0),
                    [frame_bgr],
                    [shapes],
                    np.stack([img_mask_chw], 0),
                    [motion_mask],
                    [gmc_time]
                ))
        if not self.stopped:
            self.queue.put(None)
            
    def __iter__(self):
        return self
        
    def __next__(self):
        res = self.queue.get()
        if res is None:
            raise StopIteration
        return res
        
    def release(self):
        self.stopped = True
        self.stream.release()

# ══════════════════════════════════════════════════════════════
#  Main inference loop
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def run(
    weights        = ROOT / 'runs/train/ARD100_mask32-640_uavs/weights/best.engine',
    source         = ROOT / 'test_video.mp4',
    batch_size     = 8,
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
    gmc_gpu        = True,
    gmc_scale      = 2.0,
    show_mask      = False,
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

    # ── Input stream / Dataloader ────────────────────────────
    is_video = Path(source).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')
    if is_video:
        LOGGER.info('Source is a video file. Using AsyncGMCStream (batch_size=1)')
        gmc = create_gmc(prefer_gpu=gmc_gpu, scale=gmc_scale)
        base_stream = VideoStreamGMC(source, img_size=imgsz[0], stride=stride, buf_size=5)
        dataloader = AsyncGMCStream(base_stream, gmc, imgsz[0], stride)
        vid_fps = base_stream.fps
    else:
        LOGGER.info('Source is an image sequence. Using LoadGMCStream (DataLoader)')
        dataset = LoadGMCStream(source, img_size=imgsz[0], stride=stride, gmc_gpu=gmc_gpu, gmc_scale=gmc_scale)
        nw = min(os.cpu_count(), 8)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=True, collate_fn=LoadGMCStream.collate_fn)
        vid_fps = 30.0
        
    model.warmup(imgsz=(1 if pt else batch_size, 3, *imgsz))

    # ── State ────────────────────────────────────────────────
    dt             = [0.0, 0.0, 0.0, 0.0]   # prep, gmc, infer, nms
    seen           = 0
    WINDOW         = 'Drone Detection (VPI - Parallel)'
    vid_writer     = None
    vid_path_saved = None
    save_path      = str(save_dir / 'output.mp4')
    stop_infer     = False
    t_start        = time_sync()

    # ══════════════════════════════════════════════════════════
    #  Batch loop
    # ══════════════════════════════════════════════════════════
    for batch_i, (paths, img_rgb_batch, im0s_rgb, shapes_batch, img_mask_batch, motion_masks, gmc_times) in enumerate(dataloader):
        nb = len(paths)
        
        # In batch processing, GMC times are concurrent across workers, we just record the average per frame for logging
        dt[1] += (sum(gmc_times) / (nb if not is_video else 1)) / 1000.0

        # ── 1. Contiguous + Device Transfer ──────────────────
        t1 = time_sync()
        im  = torch.from_numpy(np.ascontiguousarray(img_rgb_batch)).to(device, non_blocking=True)
        im2 = torch.from_numpy(np.ascontiguousarray(img_mask_batch)).to(device, non_blocking=True)
        im  = im.half()  if half else im.float()
        im2 = im2.half() if half else im2.float()
        im  /= 255.0
        im2 /= 255.0
        t2 = time_sync()
        dt[0] += t2 - t1

        # ── 2. Inference ─────────────────────────────────────
        pred = model(im, im2, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[2] += t3 - t2

        # ── 3. NMS ───────────────────────────────────────────
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t4 = time_sync()
        dt[3] += t4 - t3

        seen += nb
        infer_ms = (dt[2] / seen) * 1e3
        gmc_ms   = (dt[1] / seen) * 1e3
        cur_fps  = 1000.0 / infer_ms if infer_ms > 0 else 0.0

        # ── 4. Xử lý từng frame trong batch ──────────────────
        for si in range(nb):
            path    = paths[si]
            im0     = im0s_rgb[si].copy()
            mask_img= motion_masks[si]
            det     = pred[si]

            s = f'batch {batch_i+1} frame {si+1}/{nb} ({Path(path).stem}) — '
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

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
                        txt_path = str(save_dir / 'labels' / Path(path).stem)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    c_int = int(cls)
                    label = None if hide_labels else (names[c_int] if hide_conf else f'{names[c_int]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c_int, True))

                    if save_crop:
                        save_one_box(xyxy, im0.copy(),
                                     file=save_dir / 'crops' / names[c_int] / f'{Path(path).stem}.jpg', BGR=True)

            im0 = annotator.result()
            
            # Overlay stats
            cv2.putText(im0, f'FPS (Infer): {cur_fps:.1f}  |  GMC/fr: {gmc_ms:.1f}ms',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if view_img:
                cv2.imshow(WINDOW, im0)
                if show_mask:
                    cv2.imshow('Motion Mask (GMC)', mask_img)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    LOGGER.info('User requested exit.')
                    stop_infer = True
                    break

            if save_img:
                if vid_path_saved != save_path:
                    vid_path_saved = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    h_out, w_out = im0.shape[:2]
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*'mp4v'),
                        vid_fps, (w_out, h_out)
                    )
                vid_writer.write(im0)

        LOGGER.info(f'{s}Done. ({(t4-t1)*1000:.1f}ms loop)')
        if stop_infer: break

    # ── Cleanup ───────────────────────────────────────────────
    if is_video: dataloader.release()
    if isinstance(vid_writer, cv2.VideoWriter): vid_writer.release()
    cv2.destroyAllWindows()

    if seen:
        t = tuple(x / seen * 1e3 for x in dt)
        t_total = (time_sync() - t_start)
        LOGGER.info(f'Speed: {t[0]:.1f}ms prep | {t[1]:.1f}ms GMC | '
                    f'{t[2]:.1f}ms infer | {t[3]:.1f}ms NMS  (per frame)')
        LOGGER.info(f'FPS inference: {1000/t[2]:.1f} | FPS total pipeline: {seen/t_total:.1f}')
    if save_txt or save_img:
        LOGGER.info(f'Results saved to {colorstr("bold", save_dir)}')
    if update:
        strip_optimizer(weights)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/ARD100_mask32-1280_uavs/weights/best.pt')
    parser.add_argument('--source',  type=str, default=ROOT / 'test_video.mp4')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280])
    parser.add_argument('--conf-thres',     type=float, default=0.25)
    parser.add_argument('--iou-thres',      type=float, default=0.45)
    parser.add_argument('--max-det',        type=int,   default=1000)
    parser.add_argument('--device',         default='')
    parser.add_argument('--view-img',       action='store_true', help='Display results')
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
    parser.add_argument('--gmc-gpu',        action='store_true', default=True, help='Use VPI/CUDA for motion mask')
    parser.add_argument('--gmc-cpu',        action='store_true', help='Force CPU fallback for motion mask')
    parser.add_argument('--gmc-scale',      type=float, default=2.0, help='GMC scale factor (match MOD_Functions: 960*scale x 540*scale)')
    parser.add_argument('--show-mask',      action='store_true', help='Display mask window')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    if opt.gmc_cpu: opt.gmc_gpu = False
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**{k: v for k, v in vars(opt).items() if k != 'gmc_cpu'})

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
