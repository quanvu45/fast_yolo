import cv2
import os
import numpy as np
from MOD_Functions import motion_compensate
from MOD_Functions import enlargebox
import imgviz

kernel_size = 3
def motion_compensate(frame1, frame2):
    # grid-based KLT tracking
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))

    # 创建随机生成的颜色
    # color = np.random.randint(0, 255, (3000, 3))
    width = frame2.shape[1]
    height = frame2.shape[0]
    scale = 2

    frame1_grid = cv2.resize(frame1, (960 * scale, 540 * scale), dst=None, interpolation=cv2.INTER_CUBIC)
    frame2_grid = cv2.resize(frame2, (960 * scale, 540 * scale), dst=None, interpolation=cv2.INTER_CUBIC)

    width_grid = frame2_grid.shape[1]
    height_grid = frame2_grid.shape[0]
    gridSizeW = 32 * 2
    gridSizeH = 24 * 2
    p1 = []
    grid_numW = int(width_grid / gridSizeW - 1)
    grid_numH = int(height_grid / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1_grid, frame2_grid, pts_prev, None, **lk_params)

    # 选择good points
    good_new = pts_cur[st == 1]  # 当前帧中的跟踪点
    good_old = pts_prev[st == 1]  # 前一帧中的跟踪点

    # points_new = []
    # points_old = []

    # 绘制跟踪框
    # mask0 = np.zeros_like(frame2)  # 为绘制创建掩码图片
    motion_distance = []
    translate_x = []
    translate_y = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        motion_distance0 = np.sqrt((a - c) * (a - c) + (b - d) * (b - d))

        if motion_distance0 > 50:
            continue

        translate_x0 = a - c
        translate_y0 = b - d

        # point_new = np.array([a, b])
        # point_old = np.array([c, d])
        # points_new.append(point_new)
        # points_old.append(point_old)

        motion_distance.append(motion_distance0)
        translate_x.append(translate_x0)
        translate_y.append(translate_y0)
        # mask0 = cv2.line(mask0, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 3)
        # cv2.circle(frame2, (int(a), int(b)), 3, color[i].tolist(), -1)
    motion_dist = np.array(motion_distance)
    motion_x = np.mean(np.array(translate_x))
    motion_y = np.mean(np.array(translate_y))

    avg_dst = np.mean(motion_dist)

    # points_new = np.array(points_new)
    # points_old = np.array(points_old)
    # img_optflow = cv2.add(frame2, mask0)
    # cv2.imwrite('./output/drone1_grid/frame_' + str(frameCount) + '.jpg', img_optflow)
    # cv2.imshow('frame with optical flow ', img_optflow)

    # homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)
    # homography_matrix, status = cv2.findHomography(points_new, points_old, cv2.RANSAC, 3.0)
    # matchesMask = status.ravel().tolist()
    if len(good_old) < 15:
        homography_matrix = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]])
    else:
        homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)

    # 根据变换矩阵计算变换之后的图像
    compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # 计算掩膜
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    homo_inv = np.linalg.inv(homography_matrix)
    vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
    vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
    im = np.zeros(frame1.shape[:2], dtype='uint8')
    cv2.polylines(im, vertex_transformed, 1, 255)
    cv2.fillPoly(im, vertex_transformed, 255)
    mask = 255 - im

    return compensated, mask, avg_dst, motion_x, motion_y, homography_matrix

def FD5_mask(lastFrame1, lastFrame2, currentFrame, video_name, frame_count):
    lastFrame1 = cv2.GaussianBlur(lastFrame1, (11, 11), 0)
    lastFrame1 = cv2.cvtColor(lastFrame1, cv2.COLOR_BGR2GRAY)

    lastFrame2 = cv2.GaussianBlur(lastFrame2, (11, 11), 0)
    lastFrame2 = cv2.cvtColor(lastFrame2, cv2.COLOR_BGR2GRAY)

    currentFrame = cv2.GaussianBlur(currentFrame, (11, 11), 0)
    currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)

    img_compensate1, mask1, avg_dist1, motion_x1, motion_y1, homo_matrix = motion_compensate(lastFrame1, lastFrame2)
    frameDiff1 = cv2.absdiff(lastFrame2, img_compensate1)
    # fix_coef1 = np.mean(frameDiff1)
    # fix_coef1 = int(fix_coef1)
    # T_1 = 4 + fix_coef1
    # _, thresh1 = cv2.threshold(frameDiff1, T_1, 255, cv2.THRESH_BINARY)
    # thresh = thresh1 - mask1
    # thresh = cv2.medianBlur(thresh, 5)

    img_compensate2, mask2, avg_dist2, motion_x2, motion_y2, homo_matrix2 = motion_compensate(currentFrame, lastFrame2)
    frameDiff2 = cv2.absdiff(lastFrame2, img_compensate2)
    # fix_coef2 = np.mean(frameDiff2)
    # fix_coef2 = int(fix_coef2)
    # T_2 = 4 + fix_coef2
    # _, thresh2 = cv2.threshold(frameDiff2, T_2, 255, cv2.THRESH_BINARY)
    # thresh2 = thresh2 - mask2
    #
    # thresh = cv2.bitwise_or(thresh1, thresh2)

    frameDiff = (frameDiff1 + frameDiff2) / 2

    # _, thresh3 = cv2.threshold(np.uint8(frameDiff), 5, 255, cv2.THRESH_BINARY)
    # thresh3 = thresh3 - mask1
    # thresh = thresh3 - mask2
    #
    # width = frameDiff.shape[1]
    # height = frameDiff.shape[0]

    # if width == 1920:
    #     frameDiff[620:720, 0:240] = 0
    # else:
    #     frameDiff[540:640, 0:140] = 0

    # if width == 1920:
    #     thresh[620:720, 0:240] = 0
    # else:
    #     thresh[540:640, 0:140] = 0

    # frame_depth = thresh.astype(np.float)
    # frame_viz = imgviz.depth2rgb(frame_depth, min_value=5, max_value=30)

    # 对阈值图像进行开操作，减少噪声
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # open_demo = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)

    # 对开操作之后的图像做闭操作，减少孔洞，填充空隙
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # close_demo = cv2.morphologyEx(open_demo, cv2.MORPH_CLOSE, kernel2, iterations=3)
    # # cv2.imshow('Morphological Operation', close_demo)

    save_path = '/home/ccne/Documents/YOLOMG/datasets/ARD100_mask32/' + video_name

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_path + '/' + video_name + '_' + str(frame_count).zfill(4) + '.jpg', frameDiff)

    # save_path = '/home/user-guo/data/drone-videos/NPS-Dataset/mask5_thresh/' + video_name
    #
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # cv2.imwrite(save_path + '/' + video_name + '_' + str(frame_count).zfill(4) + '.jpg', close_demo)
    #
    # contours, hierarchy = cv2.findContours(close_demo.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # traverse contours
    # rect_list = []
    # for contour in contours:
    #     # if contour is too small or too big, ignore it
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     # area = cv2.contourArea(contour)
    #     area = w * h
    #     ratio = w / h
    #     if 25 < area < 3000 and 0.3 < ratio < 3.0:
    #         rect = (x, y, w, h)
    #         rect_list.append(rect)
    #
    # # rect_merge = box_select(np.array(rect_list))
    # rect_merge = rect_list
    # obj_num = len(rect_merge)

    return 0
