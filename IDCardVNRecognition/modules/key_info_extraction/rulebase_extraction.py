import numpy as np
import pandas as pd
import cv2
import Levenshtein
import re
from scipy.cluster.vq import kmeans, vq

cls_out_txt = '/mnt/e/bkai/MC_OCR/modules/output_test/text_classifier/mc_ocr_private_test/txt/cccd_{}{}.txt'
cls_out_path = '/mnt/e/bkai/MC_OCR/modules/data/mc_ocr_private_test/cccd_{}{}.png'
ext_out_csv = '/mnt/e/bkai/MC_OCR/modules/output_test/key_info_extraction/mc_ocr_private_test/result.csv'

def string_similarity(input, target):
    return 1 - 1.0*Levenshtein.distance(input, target)/max(len(input), len(target))

def similar(input, target):
    return (target in input) or (Levenshtein.distance(input, target) <= 2) or (string_similarity(input, target) > 0.8)

def read_anno_from_path(path):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        annos = f.readlines()

    for anno in annos:
        info = anno.strip().split(',')
        tmp = []
        for _, coor in enumerate(info[:8]):
            tmp.append(int(coor))
        text = ','.join(info[8:])
        tmp += [text, 1]    # prob = 1
        res.append(tmp)
    return res

def calculate_thickness(boxes, image):
    # Convert image to blackwhite image and inverted image
    # Convert to grayscale -> Convert to blackwhite
    img = image.copy()
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, black_white = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(black_white)
    # cv2.imwrite('/mnt/e/bkai/MC_OCR/modules/text_classifier/grayscale.png', grayscale)
    # cv2.imwrite('/mnt/e/bkai/MC_OCR/modules/text_classifier/black_white.png', black_white)
    # cv2.imwrite('/mnt/e/bkai/MC_OCR/modules/text_classifier/inverted.png', inverted_img)
    # quit(0)
    thickness = []
    for idx, box in enumerate(boxes):
        thickness.append([calculate_box_thickness(idx, box, black_white, inverted), box[8]])
    return thickness


def calculate_box_thickness(idx, box, bw_img, ivt_img):  # box: x1, y1, _, _, x3, y3
    ivt_img = ivt_img[ymin:ymax, xmin:xmax]
    # cv2.imwrite('/mnt/e/bkai/MC_OCR/modules/text_classifier/output_test{}.png'.format(idx), ivt_img)

    bw_img = bw_img[box[1]:box[5], box[0]:box[4]]
    # use_invert = True  # Sử dụng invert khi background: white (255), font: black (0)
    # Sử dụng img (không convert) khi background: black (0), font: white (255)
    bw_img = np.array(bw_img)
    cnt_white_col = 0
    cnt_black_col = 0
    for col in bw_img.T:
        white_col = all([p == 255 for p in col])
        black_col = all([p == 0 for p in col])
        cnt_white_col += 1 if white_col else 0
        cnt_black_col += 1 if black_col else 0

    use_invert = cnt_white_col >= cnt_black_col
    cvt_img = ivt_img if use_invert else bw_img
    px_compare = 255 if use_invert else 0
    thinned = cv2.ximgproc.thinning(cvt_img)
    print(use_invert)
    # cv2.imwrite('/mnt/e/bkai/MC_OCR/modules/text_classifier/thin{}.png'.format(idx), thinned)
    num_thin, num_invert = 0, 0
    for r in cvt_img:
        for pixel in r:
            num_invert += 1 if pixel == px_compare else 0
    for r in thinned:
        for pixel in r:
            num_thin += 1 if pixel == px_compare else 0
    # Compute thickness:
    thickness = (num_invert - num_thin) / num_thin if num_thin else 0
    # Normalize:
    # thickness /= 255
    return thickness

def convert_to_rect(coors):  # np: (-1, 4, 2) -> (-1, 2, 2): [[[xmin, ymin], [xmax, ymax]], ...]
    boxes = []
    for coor in coors:
        min_coor = np.min(coor, axis=0)
        max_coor = np.max(coor, axis=0)
        boxes.append([min_coor, max_coor])
    return np.array(boxes, dtype=np.int32)

def find_nearest_right(box_id, boxes, thres=0.2):
    box = boxes[box_id]
    xmin, ymin, xmax, ymax = box[0][0], box[0][1], box[1][0], box[1][1]
    candidates = []
    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = b[0][0], b[0][1], b[1][0], b[1][1]
        if x2 <= xmin or idx == box_id:
            continue
        if ymax < y1 or ymin > y2:
            iou = 0
        else:
            y = sorted([ymin, ymax, y1, y2])
            iou = (y[2] - y[1]) / (y[3] - y[0])
            if iou >= thres:
                candidates.append((idx,iou))
    sorted(candidates, key=lambda x: x[1])
    return candidates[0][0] if candidates is not None else None

def find_nearest_below(box_ids, boxes, thres=0.15):
    box = boxes[box_ids[0]]
    box_ = boxes[box_ids[1]]
    _, ymin, _, ymax = box[0][0], box[0][1], box[1][0], box[1][1]
    candidates = []
    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = b[0][0], b[0][1], b[1][0], b[1][1]
        if x2 <= box_[1][0] or idx <= box_ids[0]:   # Only take box has x > 'Quê quán/Nơi thường trú' box and below its contents
            continue
        if (ymin < y1) and (1.0*(y1-ymax)/(max(ymax-ymin, y2-y1)) < thres):
            candidates.append((idx, y1-ymax))
    sorted(candidates, key=lambda x: x[1])
    if len(candidates):
        return candidates[0][0]
    return None

def extract_extra_key(idx, key, keys, texts, boxes, expected=None):
    match = similar(texts[idx], key)
    if not match:
        return None
    ext_box = None
    if ':' in texts[idx] and texts[idx][-1] != ':':
        out = texts[idx].split(':')[-1].strip()
        ext_box = (idx, idx)
    else:
        right_idx = find_nearest_right(idx, boxes)
        if right_idx is None:
            return None
        out = texts[right_idx]
        for k in keys:
            if k in out:
                return None
        ext_box = (right_idx, idx)
    if key in ['Quê quán', 'Nơi thường trú']:
        thres = 0.25 if key == 'Quê quán' else 9999
        below_idx = find_nearest_below(ext_box, boxes, thres=thres)
        if below_idx is not None:
            txt = texts[below_idx]
            print(out)
            print(txt)
            out = out + ', ' + txt
    if expected is not None:
        for e in expected:
            if e == out:
                return e
    return out

extract_keys = ['ID', 'Họ và tên', 'Ngày sinh', 'Giới tính', 'Quốc tịch', 'Quê quán', 'Dân tộc', 'Nơi thường trú']


def rule_base_extract(anno_path=None, anno_info=None):    # anno_info = [[x1, y1... x4, y4, Text, Prob], ...]
    if anno_info is None:
        anno_info = read_anno_from_path(anno_path)
    anno_info = sorted(anno_info, key=lambda x: (x[1],x[0]))
    coors = np.reshape([info[:8] for info in anno_info], newshape=(-1, 4, 2))
    boxes = convert_to_rect(coors)
    texts = [info[8] for info in anno_info]
    probs = [info[9] for info in anno_info]
    global extract_keys
    info_extract = {key:None for key in extract_keys}
    flag_key = {key:False for key in extract_keys}
    flag_box = [False]*len(coors)

    id_pattern = r'([\d]{9}[\d]*)$'
    date_pattern = r'(\d+/\d+/\d{4})'

    for i in range(len(boxes)):
        box = boxes[i]
        text = texts[i]

        id_match = re.search(id_pattern, text)
        if id_match is not None and not flag_key['ID']:
            info_extract['ID'] = id_match.group(1)
            flag_key['ID'] = True
            continue

        if not flag_key['Họ và tên']:
            name_match = similar(text, 'Họ và tên')
            if name_match:
                if ':' in text and text[-1] != ':':
                    info_extract['Họ và tên'] = text.split(':')[-1]
                else:
                    right_idx = find_nearest_right(i, boxes, thres=0)
                    if right_idx is not None:
                        txt = texts[right_idx]
                        info_extract['Họ và tên'] = txt if txt.upper() == txt else info_extract['Họ và tên']
                flag_key['Họ và tên'] = True
                continue

        birthday_match = [(s in text) for s in ['Ngày', 'tháng', 'năm sinh']].count(True) >= 2
        if birthday_match and not flag_key['Ngày sinh']:
            check_contain_date1 = re.search(date_pattern, text)
            if check_contain_date1 is not None:
                res = check_contain_date1.group(1)
            else:
                right_idx = find_nearest_right(i, boxes)
                if right_idx is None:
                    continue
                right_txt = texts[right_idx]
                check_contain_date2 = re.search(date_pattern, right_txt)
                res = check_contain_date2.group(1) if check_contain_date2 is not None else None
            info_extract['Ngày sinh'] = res
            flag_key['Ngày sinh'] = True
            continue

        for key in ['Giới tính', 'Quốc tịch', 'Dân tộc', 'Quê quán', 'Nơi thường trú']:
            expected = ['Nam', 'Nữ'] if key == 'Giới tính' else None
            if not flag_key[key]:
                info = extract_extra_key(i, key, extract_keys, texts, boxes, expected=expected)
                if info is not None:
                    info_extract[key] = info
                    flag_key[key] = True
                    break
    return info_extract

def rule_base_batch_extract(anno_paths):
    results = []
    for anno_path in anno_paths:
        res_extract = rule_base_extract(anno_path)
        results.append(list(res_extract.values()))
    return results

def make_list_path(b=0, e=20):
    paths = []
    for i in range(b, e):
        pad_zeros = ''
        for _ in range(3-len(str(i))):
            pad_zeros += '0'
        path = cls_out_txt.format(pad_zeros, i)
        paths.append(path)
    return paths

if __name__ == '__main__':
    # b = 0
    # e = 20
    # results = rule_base_batch_extract(anno_paths=make_list_path(b, e))
    # df = pd.DataFrame(results, columns=extract_keys)
    # df.to_csv(ext_out_csv, encoding='utf-8')
    # print(df)
    # exg = np.array([1, 1, 0, 0, 0, 0, 0, 2, 2, 2], dtype=float)
    # # codebook, distortion = kmeans(exg, 2)
    # # print(codebook)
    # # print(distortion)
    # print(set(exg))

    x = np.array([[0, 1], [1, 1], [2, 0], [0, 0]])
    y = cv2.minAreaRect(x)
    print(y)