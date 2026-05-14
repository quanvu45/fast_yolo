import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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

IMG_FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')


# ══════════════════════════════════════════════════════════════
#  Dual-stream frame loader
# ══════════════════════════════════════════════════════════════
class LoadDualStream(Dataset):
    def __init__(self, path_rgb, path_mask, img_size=640, stride=32):
        self.img_size = img_size if isinstance(img_size, int) else img_size[0]
        self.stride   = int(stride)

        self.im_files  = self._load_paths(path_rgb)
        self.im_files2 = self._load_paths(path_mask)

        assert len(self.im_files) == len(self.im_files2), (
            f'Số ảnh RGB ({len(self.im_files)}) ≠ số ảnh mask ({len(self.im_files2)})'
        )
        assert len(self.im_files) > 0, 'Không tìm thấy ảnh nào!'

        self.n     = len(self.im_files)
        self.index = 0
        LOGGER.info(f'LoadDualStream: {self.n} frames  |  RGB : {path_rgb}  |  Mask: {path_mask}')

    @staticmethod
    def _load_paths(path):
        import glob
        p = Path(path)
        files = []

        if p.is_file():
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
        else:
            raise FileNotFoundError(f'Không tồn tại: {path}')

        im_files = sorted(
            x.replace('/', os.sep) for x in files
            if x.split('.')[-1].lower() in IMG_FORMATS
        )
        assert im_files, f'Không tìm thấy ảnh nào trong: {path}'
        return im_files

    def _preprocess(self, f):
        img0_bgr = cv2.imread(f)
        assert img0_bgr is not None, f'Không đọc được: {f}'

        h0, w0 = img0_bgr.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img0_bgr, (int(w0 * r), int(h0 * r)), interpolation=interp)
        else:
            img = img0_bgr
        h, w = img.shape[:2]

        img_lb, ratio, pad = letterbox(img, self.img_size, stride=self.stride, auto=False, scaleup=False)
        shapes = (h0, w0), ((h / h0, w / w0), pad)
        img_chw = np.ascontiguousarray(img_lb.transpose((2, 0, 1))[::-1])

        return img_chw, img0_bgr, shapes

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        f_rgb  = self.im_files [index]
        f_mask = self.im_files2[index]
        img_rgb,  im0_rgb,  shapes_rgb  = self._preprocess(f_rgb)
        img_mask, im0_mask, shapes_mask = self._preprocess(f_mask)
        return (f_rgb, f_mask, img_rgb, im0_rgb, shapes_rgb, img_mask, im0_mask, shapes_mask)

    @staticmethod
    def collate_fn(batch):
        f_rgb, f_mask, img_rgb, im0_rgb, shapes_rgb, img_mask, im0_mask, shapes_mask = zip(*batch)
        img_rgb  = np.stack(img_rgb,  0)
        img_mask = np.stack(img_mask, 0)
        return (list(f_rgb), list(f_mask), img_rgb, list(im0_rgb), list(shapes_rgb), img_mask, list(im0_mask), list(shapes_mask))


# ══════════════════════════════════════════════════════════════
#  Main inference loop
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def run(
    weights        = ROOT / 'runs/train/ARD100_mask32-1280_uavs/weights/best.pt',
    source         = ROOT / 'datasets/rgb_list.txt',
    source2        = ROOT / 'datasets/mask_list.txt',
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
    name           = 'exp',
    exist_ok       = False,
    line_thickness = 3,
    hide_labels    = False,
    hide_conf      = False,
    half           = False,
    dnn            = False,
):
    source  = str(source)
    source2 = str(source2)
    save_img = not nosave

    # ── Directories ──────────────────────────────────────────
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────
    device = select_device(device)
    model  = DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz  = check_img_size(imgsz, s=stride)

    # FIX TENSORRT HALF PRECISION
    if pt:
        half &= device.type != 'cpu'
        model.model.half() if half else model.model.float()
    else:
        # Giữ nguyên cờ half cho TensorRT engine
        pass

    # ── Dataset ──────────────────────────────────────────────
    dataset = LoadDualStream(source, source2, img_size=imgsz, stride=stride)
    nw = min(os.cpu_count(), 8)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=True, collate_fn=LoadDualStream.collate_fn)
    model.warmup(imgsz=(1 if pt else batch_size, 3, *imgsz))

    # ── Biến theo dõi ─────────────────────────────────────────
    dt             = [0.0, 0.0, 0.0]
    seen           = 0
    WINDOW         = 'Drone Detection'
    vid_writer     = None
    vid_path_saved = None
    save_path      = str(save_dir / 'output.mp4')
    stop_infer     = False

    # ══════════════════════════════════════════════════════════
    #  Batch loop
    # ══════════════════════════════════════════════════════════
    for batch_i, (paths, paths2, img_rgb_batch, im0s_rgb, shapes_rgb_batch, img_mask_batch, im0s_mask, shapes_mask_batch) in enumerate(dataloader):
        nb = len(paths)

        # 1. FIX CONTIGUOUS MEMORY VÀ KIỂU DỮ LIỆU CHUẨN
        t1 = time_sync()
        im_np  = np.ascontiguousarray(img_rgb_batch)
        im2_np = np.ascontiguousarray(img_mask_batch)

        im  = torch.from_numpy(im_np).to(device, non_blocking=True)
        im2 = torch.from_numpy(im2_np).to(device, non_blocking=True)

        im  = im.half()  if half else im.float()
        im2 = im2.half() if half else im2.float()
        im  /= 255.0
        im2 /= 255.0
        t2 = time_sync()
        dt[0] += t2 - t1

        # 2. Inference
        # Thêm vào detect.py để profile
        with torch.autograd.profiler.emit_nvtx():
            # pred = model(im, im2)
        # stream = torch.cuda.Stream()
        # with torch.cuda.stream(stream):
            pred = model(im, im2, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # 3. NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t4 = time_sync()
        dt[2] += t4 - t3

        seen += nb
        infer_ms = (dt[1] / seen) * 1e3
        cur_fps  = 1000.0 / infer_ms if infer_ms > 0 else 0.0

        # 4. Xử lý từng frame trong batch
        for si in range(nb):
            path    = paths[si]
            im0_rgb = im0s_rgb[si]
            shapes  = shapes_rgb_batch[si]
            det     = pred[si]

            s = f'batch {batch_i+1} frame {si+1}/{nb} — '

            im0 = im0_rgb.copy()
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
                        save_one_box(xyxy, im0.copy(), file=save_dir / 'crops' / names[c_int] / f'{Path(path).stem}.jpg', BGR=True)

            im0 = annotator.result()

            cv2.putText(im0, f'FPS: {cur_fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 5. Hiển thị trực tiếp trên luồng chính
            if view_img:
                cv2.imshow(WINDOW, im0)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    LOGGER.info('Thoát theo yêu cầu người dùng.')
                    stop_infer = True
                    break

            if save_img:
                if vid_path_saved != save_path:
                    vid_path_saved = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    h_out, w_out = im0.shape[:2]
                    vid_writer   = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w_out, h_out))
                vid_writer.write(im0)

        LOGGER.info(f'{s}Done. ({(t4-t2)*1000:.1f}ms batch | FPS: {cur_fps:.1f})')

        # Dừng hẳn việc lặp qua batch nếu người dùng bấm Q
        if stop_infer:
            break

    # ── Dọn dẹp ──────────────────────────────────────────────
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
    cv2.destroyAllWindows()

    # ── Thống kê cuối ─────────────────────────────────────────
    if seen:
        t = tuple(x / seen * 1e3 for x in dt)
        LOGGER.info(f'Speed: {t[0]:.1f}ms pre-process | {t[1]:.1f}ms inference | {t[2]:.1f}ms NMS  (per frame)')
        LOGGER.info(f'FPS inference: {1000/t[1]:.1f}  |  FPS full pipeline: {1000/sum(t):.1f}')
    if save_txt or save_img:
        LOGGER.info(f'Results saved to {colorstr("bold", save_dir)}')
    if update:
        strip_optimizer(weights)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/ARD100_mask32-1280_uavs/weights/best.pt')
    parser.add_argument('--source',  type=str, default=ROOT / 'datasets/rgb_list.txt')
    parser.add_argument('--source2', type=str, default=ROOT / 'datasets/mask_list.txt')
    parser.add_argument('--batch-size', type=int, default=8)
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
    parser.add_argument('--name',           default='exp')
    parser.add_argument('--exist-ok',       action='store_true')
    parser.add_argument('--line-thickness', type=int, default=3)
    parser.add_argument('--hide-labels',    action='store_true')
    parser.add_argument('--hide-conf',      action='store_true')
    parser.add_argument('--half',           action='store_true')
    parser.add_argument('--dnn',            action='store_true')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)