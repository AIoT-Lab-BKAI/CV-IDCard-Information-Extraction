import numpy as np
import cv2, os, time
from datetime import datetime
from modules.utils.common import get_list_file_in_folder
from modules.utils.visualize import viz_icdar
from modules.rotation_corrector.predict import init_box_rectify_model
from modules.rotation_corrector.utils.utils import rotate_image_bbox_angle
from modules.rotation_corrector.filter import drop_box, get_mean_horizontal_angle, filter_90_box
from modules.rotation_corrector.utils.line_angle_correction import rotate_and_crop
from modules.config import rot_drop_thresh, rot_visualize, rot_model_path, gpu

os.environ["PYTHONIOENCODING"] = "utf-8"
pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')

# vietocr = True
# get bbox
crop_method = 2  # overall method 2 is better than method 1 FOR CLASSIFY
classifier_batch_sz = 4
worker = 1
write_rotated_img = False
write_file = False
visualize = rot_visualize
# extend_bbox = True  # extend bbox when crop or not
debug = False

# box rotation classifier
weight_path = rot_model_path
classList = ['0', '180']

if gpu is None or debug:
    classifier_batch_sz = 1
    worker = 0


def write_output(list_boxes, result_file_path):
    result = ''
    for idx, box_data in enumerate(list_boxes):
        if isinstance(box_data, dict):
            box = box_data['coors']
            s = [str(i) for i in box]
            line = ','.join(s) + box_data['data']
        else:
            box = box_data
            s = [str(i) for i in box]
            line = ','.join(s) + ','
        result += line + '\n'
    result = result.rstrip('\n')
    with open(result_file_path, 'w', encoding='utf8') as res:
        res.write(result)


def get_boxes_data(img_data, boxes):
    boxes_data = []
    for idx, box_data in enumerate(boxes):
        if isinstance(box_data, dict):
            box_loc = box_data['coors']
        else:
            box_loc = box_data
        box_loc = np.array(box_loc).astype(np.int32).reshape(-1, 1, 2)
        box_data = rotate_and_crop(idx, img_data, box_loc, debug=False, extend=True,
                                   extend_x_ratio=0.0001,
                                   extend_y_ratio=0.0001,
                                   min_extend_y=2, min_extend_x=1)
        boxes_data.append(box_data)
    return boxes_data


def calculate_page_orient(box_rectify, img_rotated, boxes_list):
    boxes_data = get_boxes_data(img_rotated, boxes_list)
    rotation_state = {'0': 0, '180': 0}
    for it, img in enumerate(boxes_data):
        _, degr = box_rectify.inference(img, debug=False)
        rotation_state[degr[0]] += 1
    print(rotation_state)
    if rotation_state['0'] >= rotation_state['180']:
        ret = 0
    else:
        ret = 180
    return ret


def convert_suitable_form(boxes_data):  # boxes_data: (-1, 4, 2)
    boxes_list = []
    for box in boxes_data:
        box = np.array(box, dtype=np.int32).reshape(-1).tolist()
        boxes_list.append({'coors': box, 'data': ','})
    return boxes_list


def rotate_horizontal_image(boxes_data, image, img_name):
    box_rectify = init_box_rectify_model(weight_path)

    begin = time.time()
    # print('\n', 'Inference ', img_name)

    begin_detector = time.time()
    boxes_list = convert_suitable_form(boxes_data)

    # rot_drop_thresh = [.5, 2.]
    # usage: remove all boxes have 0.5 < w/h < 2
    boxes_list = drop_box(boxes_list, drop_gap=rot_drop_thresh)

    # return: Góc để xoay ảnh theo chiều ngang của cạnh dài hơn(Góc > 0: tính theo chiều kim đồng hồ)
    rotation = get_mean_horizontal_angle(boxes_list, False)

    print('\nAngle: {}\n'.format(rotation))

    # @image: origin image
    # @boxes_list: ['coors': [x1, y1, ... x4, y4], 'data': None]
    # @rotation: angle to rotate horizontal (either reverse or not reverse)
    img_rotated, boxes_list = rotate_image_bbox_angle(image, boxes_list, rotation)

    # For each transformed box, predict direction of that one (class: [0(degree), 18(degree)])
    # Return: argmax(len(boxes: 0degree), len(boxes: 180degree))
    degre = calculate_page_orient(box_rectify, img_rotated, boxes_list)

    # Rotate image with yet calculated degree
    img_rotated, boxes_list = rotate_image_bbox_angle(img_rotated, boxes_list, degre)

    # print('Image rotate\n', img_rotated)
    # print('boxes_list\n', boxes_list)

    # Filter boxes have h > w (boxes is rotated vertically)
    boxes_list = filter_90_box(boxes_list)
    end_detector = time.time()
    print('get boxes from icdar time:', end_detector - begin_detector, 'seconds')

    draw_im = draw_text_rot_res(dt_boxes=boxes_list, src_im=img_rotated.copy())
    return img_rotated, boxes_list, draw_im


def draw_text_rot_res(dt_boxes, src_im):  # img_path: only file
    for box in dt_boxes:
        box = np.array(box['coors']).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(0, 0, 255), thickness=2)
    return src_im


def get_list_boxes_from_icdar(anno_path):
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno_txt = f.readlines()
    list_boxes = []
    for anno in anno_txt:
        anno = anno.rstrip('\n')

        idx = -1
        for i in range(0, 8):
            idx = anno.find(',', idx + 1)

        coordinates = anno[:idx]
        coors = [int(f) for f in coordinates.split(',')]
        list_boxes.append({'coors': coors, 'data': anno[idx:]})
    return list_boxes
