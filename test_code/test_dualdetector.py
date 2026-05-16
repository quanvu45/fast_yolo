
import os
import sys
# Fix path logic to work from any directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import cv2
import time
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

class Yolov5Detector():
    def __init__(self, weights='', imgsz=640, device='0'):
        self.device = select_device(device)
        self.half = self.device.type != 'cpu' 
        
        self.model = attempt_load(weights, map_location=self.device) 
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=self.stride)
        if self.half:
            self.model.half()
        
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())),
                       torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

    def imgdeal(self, img):
        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def run(self, img1, img2, conf_thres=0.1, iou_thres=0.4, classes=None):
        img1 = self.imgdeal(img1)
        img2 = self.imgdeal(img2)
        pred = self.model(img1, img2, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)
        det = pred[0]

        if len(det):
            img0_shape = torch.from_numpy(np.array([1080, 1920])).to(self.device)
            boxes = scale_coords(img1.shape[2:], det[:, :4], img0_shape).round().cpu().numpy()
            labels = [self.names[int(cls)] for cls in det[:, -1]]
            scores = [float('%.2f' % conf) for conf in det[:, -2]]
            return labels, scores, boxes
        else:
            return [], [], np.array([])

if __name__ == '__main__':
    weights = os.path.join(ROOT, 'runs/train/ARD100_mask32-640_uavs/weights/best.pt')
    img1_path = os.path.join(ROOT, 'datasets/images/phantom144/phantom144_0968.jpg')
    img2_path = os.path.join(ROOT, 'datasets/ARD100_mask32/phantom144/phantom144_0968.jpg')
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not read images.\nRGB: {img1_path}\nMask: {img2_path}")
        sys.exit(1)
        
    detector = Yolov5Detector(weights=weights, imgsz=640, device='0')
    
    # Warmup
    for _ in range(10):
        detector.run(img1, img2, classes=[0])
    
    iters = 100
    t1 = time.time()
    for _ in range(iters):
        labels, scores, boxes = detector.run(img1, img2, classes=[0]) 
    t2 = time.time()
    
    avg_time = (t2 - t1) / iters
    print(f'Average Time per frame: {avg_time*1000:.2f} ms')
    print(f'FPS: {1.0/avg_time:.2f}\n')

    print('Labels: ', labels)
    print('Scores: ', scores)
    print('Boxes: ', boxes)
