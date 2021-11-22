import cv2
import math
import numpy as np
from math import sqrt
from scipy.cluster.vq import kmeans, vq
from modules.utils.common import poly


def visual_box(bbox):
    box_np = np.array(bbox).astype(np.int32).reshape(-1, 1, 2)
    minsize = np.amin(box_np, axis=0)
    box_np = box_np - minsize
    rect = cv2.minAreaRect(box_np)
    box_enhan = cv2.boxPoints(rect)
    x, y, w, h = cv2.boundingRect(box_enhan)
    blank_image = np.zeros((h + 20, w + 20, 3), np.uint8)
    box_np = box_np + 10
    pts = box_np.reshape((-1, 1, 2))
    blank_image = cv2.polylines(blank_image, [pts], True, (255, 255, 255), 1)
    cv2.imshow('ct', blank_image)
    cv2.waitKey()


def visual_box_full_image(image, bboxes, debug=False):
    for bbox in bboxes:
        box_np = np.array(bbox).astype(np.int32).reshape(-1, 1, 2)
        pts = box_np.reshape((-1, 1, 2))
        image = cv2.polylines(image, [pts], True, (0, 255, 0), 2)
    if debug:
        cv2.imshow('ct', image)
        cv2.waitKey()
    return image


def drop_box(boxlist, drop_gap=(.5, 2), debug=False):
    new_boxlist = []
    # print('------ DROP BOX ------')
    for ide, box_data in enumerate(boxlist):
        if isinstance(box_data, dict):
            box = box_data['coors'] # [x1, y1, x2, y2, x3, y3, x4, y4]
        else:
            box = box_data
        box_np = np.array(box).astype(np.int32).reshape(-1, 1, 2)   # shape: [4, 1, 2]: 4 pts [x, y]
        rect = cv2.minAreaRect(box_np)  # Find the minimum area rectangle contains all point in box (nearly the same because pts are approximately rectangle)
        # print('Rect: ', ide)
        # print(rect)
        w, h = rect[1]
        if debug:
            visual_box(box)
        if min(drop_gap) < w / h < max(drop_gap):
            continue
        new_boxlist.append(box_data)
    # print('-----------')
    return new_boxlist


def stat(lst):
    """Calculate mean and std deviation from the input list."""
    n = float(len(lst))
    mean = sum(lst) / n
    a = (sum(x * x for x in lst) / n)
    b = (mean * mean)
    # print()
    # print()
    if a > b:
        stdev = sqrt((sum(x * x for x in lst) / n) - (mean * mean))
    else:
        stdev = 0
    print(stdev)
    return mean, stdev


def parse(lst, n):
    cluster = []
    for i in lst:
        if len(cluster) <= 1:  # the first two values are going directly in
            cluster.append(i)
            continue

        mean, stdev = stat(cluster)
        if abs(mean - i) > n * stdev:  # check the "distance"
            yield cluster
            cluster[:] = []  # reset cluster to the empty list

        cluster.append(i)
    yield cluster  # yield the last cluster


def get_mean_horizontal_angle(img_name, img, boxlist, debug=False, cluster=True):  # box_list: [{'coors': [x1, y1, x2, y2, x3, y3, x4, y4], 'data': None}, ...]
    if not boxlist:
        return 0
    all_box_angle = []
    tmp = []
    print('\n==== get_horizontal_angle: ')
    for ide, box_data in enumerate(boxlist):    # Với mỗi box, tìm ra angle để xoay về chiều ngang

        if isinstance(box_data, dict):
            box = box_data['coors']
        else:
            box = box_data
        # box: return [x1, y1, x2, y2, x3, y3, x4, y4]
        pol = poly(box)
        # print('Box {}: '.format(ide))
        angle_with_horizontal_line = pol.get_horizontal_angle()

        # print('End get_horizontal_angle\n')
        # print('\n==== Test poly ====')
        # print('Box: ', box)
        # print('drawing...')

        # (x1, y1, x2, y2, x3, y3, x4, y4) = tuple(box)
        # imgrect = cv2.polylines(img, pts=[np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)
        # cv2.imwrite('/mnt/e/bkai/MC_OCR/modules/3test_imgs/{}_box_{}.png'.format(img_name,ide), imgrect)
        # print('angle to rotate horizontal: ', angle_with_horizontal_line)  # góc để xoay ảnh về chiều ngang
        # print('\n')
        # if 45 < abs(angle_with_horizontal_line) < 135:
        #     continue
        # print(angle_with_horizontal_line)
        tmp.append(angle_with_horizontal_line)
        if angle_with_horizontal_line >= 0:
            angle_with_horizontal_line = 180 - angle_with_horizontal_line + 90
        else:
            angle_with_horizontal_line = math.fabs(angle_with_horizontal_line) - 90
        all_box_angle.append(angle_with_horizontal_line)
        if debug:
            visual_box(box)
    # print('Before observation: ', tmp)
    # print('Before calfiltered: ', all_box_angle)
    # print('========\n')

    if cluster: # Chỉ giữ lại phần lớn các box cùng đi theo 1 hướng
        # print('==== Begin filter outliers angle by kmeans')
        all_box_angle = filter_outliers_angle(all_box_angle)    # Loại đi các box có angle khác biệt hẳn so với phần lớn các angle
        # print('==== End filter')
    # all_box_angle
    mean_angle = np.array(all_box_angle).mean()
    mean_angle = mean_angle - 90
    # print('mean angle: ', mean_angle)
    return mean_angle


def filter_outliers_angle(list_angle, thresh=45):
    all_box_angle = np.array(list_angle)
    all_box_angle = np.absolute(all_box_angle)
    if all_box_angle.max() - all_box_angle.min() > thresh:
        # distortion: The mean (non-squared) Euclidean distance between the observations passed and the centroids generated

        codebook, distortion = kmeans(all_box_angle, 2)  # three clusters???
        # print('After_ Observation: ', all_box_angle)
        # print('Codebook: ', codebook)
        # print('Distortion: ', distortion)
        cluster_indices, _ = vq(all_box_angle, codebook)
        # print('Cluster indices:', cluster_indices)
        clas = set(cluster_indices)
        ret = {c: [] for c in clas}
        for idx, v in enumerate(all_box_angle):
            ret[cluster_indices[idx]].append(v)
        ret = list(ret.values())
        ret = sorted(ret, key=lambda e: len(e))
        list_angle = ret[-1]
    return list_angle

    # pass


def filter_90_box(boxlist, debug=False, thresh=45):
    if not boxlist:
        return 0
    all_box_angle = []
    for ide, box_data in enumerate(boxlist):
        if isinstance(box_data, dict):
            box = box_data['coors']
        else:
            box = box_data
        pol = poly(box)
        angle_with_horizontal_line = pol.get_horizontal_angle()
        # if 45 < abs(angle_with_horizontal_line) < 135:
        #     continue
        # print(angle_with_horizontal_line)
        # if angle_with_horizontal_line >= 0:
        #     angle_with_horizontal_line = 180 - angle_with_horizontal_line + 90
        # else:
        #     angle_with_horizontal_line = math.fabs(angle_with_horizontal_line) - 90
        all_box_angle.append(angle_with_horizontal_line)
        if debug:
            visual_box(box)

    # if cluster:
    all_box_angle = np.array(all_box_angle)
    all_box_angle_abs = np.absolute(all_box_angle)
    print(all_box_angle_abs.max() - all_box_angle_abs.min())
    if all_box_angle_abs.max() - all_box_angle_abs.min() > thresh:
        codebook, _ = kmeans(all_box_angle_abs, 2)  # three clusters
        cluster_indices, _ = vq(all_box_angle_abs, codebook)
        clas = set(cluster_indices)
        ret = {c: [] for c in clas}
        for idx, v in enumerate(all_box_angle_abs):
            ret[cluster_indices[idx]].append([v, boxlist[idx]])
        ret = list(ret.values())
        ret = sorted(ret, key=lambda e: len(e))
        list_angle_box = ret[-1]

        boxlist = []
        for ide, box_data in enumerate(list_angle_box):
            boxlist.append(box_data[1])
    return boxlist
