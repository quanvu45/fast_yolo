import cv2
import numpy as np
from FD2_mask import FD2_mask

# domain adaptation
set1 = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom47', 'phantom50',
        'phantom54', 'phantom55', 'phantom56', 'phantom57', 'phantom58', 'phantom60']

# light adaptation
set2 = ['phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom133', 'phantom135', 'phantom136']

# train videos
sets = ['phantom09', 'phantom10', 'phantom14', 'phantom17', 'phantom19', 'phantom20', 'phantom28', 'phantom29', 'phantom30', 'phantom32',
        'phantom36', 'phantom40', 'phantom42', 'phantom43', 'phantom44', 'phantom46', 'phantom63', 'phantom65', 'phantom66', 'phantom68',
        'phantom70', 'phantom71', 'phantom74', 'phantom75', 'phantom76', 'phantom77', 'phantom78', 'phantom80', 'phantom81', 'phantom82',
        'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89', 'phantom90', 'phantom101', 'phantom103', 'phantom104', 'phantom105',
        'phantom106', 'phantom107', 'phantom108', 'phantom109', 'phantom111', 'phantom112', 'phantom114', 'phantom115', 'phantom116', 'phantom117',
        'phantom118', 'phantom120', 'phantom132', 'phantom137', 'phantom138', 'phantom139', 'phantom140', 'phantom142', 'phantom143', 'phantom145',
        'phantom146', 'phantom147', 'phantom148', 'phantom149', 'phantom150']

# test videos
set0 = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22', 'phantom39',
        'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54', 'phantom55', 'phantom56',
        'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom64', 'phantom73', 'phantom79',
        'phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom102', 'phantom110',
        'phantom113', 'phantom119', 'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144']

Drone_vs_Bird = ['gopro_000', 'gopro_003', 'gopro_007', 'gopro_008', 'GOPR5848_004', 'GOPR5847_003',
                 'GOPR5846_005', 'GOPR5843_005', 'GOPR5845_001', 'GOPR5842_002', '2019_08_19_GOPR5869_1530_phantom',
                 '2019_09_02_GOPR5871_1058_solo']

Drone_name = ['DvsB27', 'DvsB56', 'DvsB38', 'DvsB44', 'DvsB36', 'DvsB08',
              'DvsB46', 'DvsB18', 'DvsB47', 'DvsB04', 'DvsB68', 'DvsB60']

vi_id = 0

for video_sets in Drone_vs_Bird:
    video_name = video_sets
    frame_name = Drone_name[vi_id]
    # cap = cv2.VideoCapture('/home/user-guo/data/ARD-MAV/test_videos/' + video_name + '.mp4')
    cap = cv2.VideoCapture('/home/user-guo/data/drone-videos/Drone-vs-Bird/videos/' + video_name + '.mp4')
    lastFrame1 = None
    lastFrame2 = None
    count = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        currentFrame = frame
        count = count + 1

        # frame = cv2.resize(frame, (1280, 1280), dst=None, interpolation=cv2.INTER_CUBIC)

        if lastFrame2 is None:
            if lastFrame1 is None:
                print(' first frame input')
                lastFrame1 = currentFrame
            else:
                print(' second frame input')
                lastFrame2 = currentFrame
            continue

        obj_num = FD2_mask(lastFrame1, currentFrame, count, frame_name)

        print('video name: ', video_name, end=' ')
        print('frame count: %d obj_num: %d' % (count, obj_num))

        lastFrame1 = lastFrame2
        lastFrame2 = currentFrame

    cap.release()
    vi_id = vi_id + 1


