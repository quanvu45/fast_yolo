
import cv2
import numpy as np
import time

from yolov5_dualdetector import Yolov5Detector
from FD2_mask import FD2_mask


set5 = ['phantom61', 'phantom133']


path1 = '/home/user-guo/data/drone-dataset/phantom-dataset/images/phantom133/phantom133_0126.jpg'
path2 = '/home/user-guo/data/drone-dataset/phantom-dataset/mask/phantom133/phantom133_0126.jpg'

frame = cv2.imread(path1)
frame_show = frame.copy()
mask = cv2.imread(path2)

detector = Yolov5Detector(weights='/home/user-guo/Documents/YOLOMG/runs/train/ARD100_dual_uav2-1280/weights/best.pt')
labels, scores, boxes = detector.run(frame, mask, classes=[0, 1, 2, 3, 4])  # drone

if len(boxes) != 0:
    for j in range(len(boxes)):
        x11, y11, x22, y22 = int(boxes[j][0]), int(boxes[j][1]), int(boxes[j][2]), int(boxes[j][3])
        conf = scores[j]

        # 画出边框和标签
        color = (255, 0, 0)
        cv2.rectangle(frame_show, (x11, y11), (x22, y22), color, 1, lineType=cv2.LINE_AA)
        cv2.putText(frame_show, str(f"{conf:.2f}"), (x11, y11 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

cv2.imshow('Detection', frame_show)
cv2.imwrite('./yolomg-1280-2.jpg', frame_show)


