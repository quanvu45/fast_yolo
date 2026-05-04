import cv2
import numpy as np
import time

from yolov5_dualdetector import Yolov5Detector
from FD2_mask import FD2_mask


# all test videos
set0 = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22', 'phantom39',
        'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54', 'phantom55', 'phantom56',
        'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom64', 'phantom73', 'phantom79',
        'phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom102', 'phantom110',
        'phantom113', 'phantom119', 'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144']

# domain adaptation
set1 = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom47', 'phantom50',
        'phantom54', 'phantom55', 'phantom56', 'phantom57', 'phantom58', 'phantom60']

# light adaptation
set2 = ['phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom133', 'phantom135', 'phantom136']

# test on similar scenes
set3 = ['phantom08', 'phantom22', 'phantom39', 'phantom41', 'phantom45', 'phantom61', 'phantom64',
        'phantom73', 'phantom79', 'phantom102', 'phantom110', 'phantom113', 'phantom119', 'phantom141', 'phantom144']

set4 = ['phantom05', 'phantom04', 'phantom54', 'phantom61', 'phantom79', 'phantom135', 'phantom94', 'phantom141', 'phantom144']

set5 = ['phantom61', 'phantom133']

for i in range(len(set5)):
    video_name = set5[i]
    cap = cv2.VideoCapture('./videos/' + video_name + '.mp4')

    count = 0
    prveframe = None

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    filename = "./output/" + video_name + ".mp4"
    vw = cv2.VideoWriter(filename, fourcc, int(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count = count + 1
        file_count = str(count)

        if prveframe is None:
            print('first frame input')
            prveframe = frame
            continue

        frame_show = frame.copy()
        # mask = FD2_mask(prveframe, frame)
        # color_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        path = '/home/user-guo/data/drone-dataset/phantom-dataset/mask/' + video_name + '/'
        name = video_name + '_' + str(count).zfill(4) + '.jpg'
        filename = path + name
        mask = cv2.imread(filename)

        detector = Yolov5Detector(weights='/home/user-guo/Documents/YOLOMG/runs/train/ARD100_dual_uav2-1280/weights/best.pt')
        labels, scores, boxes = detector.run(frame, mask, classes=[0, 1, 2, 3, 4])  # drone

        obj_num = 0

        if len(boxes) != 0:
            for j in range(len(boxes)):
                x11, y11, x22, y22 = int(boxes[j][0]), int(boxes[j][1]), int(boxes[j][2]), int(boxes[j][3])
                conf = scores[j]

                # 画出边框和标签
                color = (255, 0, 0)
                cv2.rectangle(frame_show, (x11, y11), (x22, y22), color, 2, lineType=cv2.LINE_AA)
                cv2.putText(frame_show, str(f"{conf:.2f}"), (x11, y11 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                obj_num = obj_num + 1

        print('video name: ', video_name, end=' ')
        print('frame count: %d obj_num: %d' % (count, obj_num))
        prveframe = frame

        cv2.putText(frame_show, "Frame: {}".format(count), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Detection', frame_show)
        vw.write(frame_show)

        key = cv2.waitKey(10) & 0xff

        if key == 27 or key == ord('q'):
            break

    cap.release()

