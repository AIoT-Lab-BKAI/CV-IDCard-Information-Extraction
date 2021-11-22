import numpy as np
import cv2, os, time
from datetime import datetime
from modules.utils.common import get_list_file_in_folder
from modules.utils.visualize import viz_icdar
from modules.rotation_corrector.predict import init_box_rectify_model
from modules.rotation_corrector.utils.utils import rotate_image_bbox_angle
from modules.rotation_corrector.filter import drop_box, get_mean_horizontal_angle, filter_90_box
from modules.rotation_corrector.utils.line_angle_correction import rotate_and_crop
from modules.config import rot_out_img_dir, rot_out_txt_dir, rot_out_viz_dir, det_out_txt_dir, raw_img_dir
from modules.config import rot_drop_thresh, rot_visualize, rot_model_path
import pandas as pd
import math

os.environ["PYTHONIOENCODING"] = "utf-8"
pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
os.environ['DISPLAY'] = ':0'

gpu = '0'

img_dir = raw_img_dir
anno_dir = det_out_txt_dir

output_txt_dir = rot_out_txt_dir
output_viz_dir = rot_out_viz_dir
output_rotated_img_dir = rot_out_img_dir

# vietocr = True
# get bbox
crop_method = 2  # overall method 2 is better than method 1 FOR CLASSIFY
classifier_batch_sz = 4
worker = 1
write_rotated_img = True
write_file = True
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
    for box_data in boxes:
        if isinstance(box_data, dict):
            box_loc = box_data['coors']
        else:
            box_loc = box_data
        box_loc = np.array(box_loc).astype(np.int32).reshape(-1, 1, 2)
        box_data = rotate_and_crop(img_data, box_loc, debug=False, extend=True,
                                   extend_x_ratio=0.0001,
                                   extend_y_ratio=0.0001,
                                   min_extend_y=2, min_extend_x=1)

        boxes_data.append(box_data)
    return boxes_data


def calculate_page_orient(img_name, box_rectify, img_rotated, boxes_list):
    boxes_data = get_boxes_data(img_rotated, boxes_list)
    rotation_state = {'0': 0, '180': 0}
    print('\n===TEST RECTIFY')
    if not os.path.exists('/mnt/e/bkai/MC_OCR/modules/6test_imgs/{}'.format(img_name.split('.')[0])):
        os.makedirs('/mnt/e/bkai/MC_OCR/modules/6test_imgs/{}'.format(img_name.split('.')[0]))

    for it, img in enumerate(boxes_data):
        _, degr = box_rectify.inference(img, debug=False)
        print('Drawing box {}....'.format(it))
        cv2.imwrite('/mnt/e/bkai/MC_OCR/modules/6test_imgs/{}/box__{}__rectified.png'.format(img_name.split('.')[0], it), img)
        print('Degree:\n', degr)
        rotation_state[degr[0]] += 1
    print(rotation_state)
    print('=== END\n')
    if rotation_state['0'] >= rotation_state['180']:
        ret = 0
    else:
        ret = 180
    return ret

def main():

    global anno_path
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)
    if not os.path.exists(output_viz_dir):
        os.makedirs(output_viz_dir)
    if not os.path.exists(output_rotated_img_dir):
        os.makedirs(output_rotated_img_dir)

    begin_init = time.time()
    box_rectify = init_box_rectify_model(weight_path)
    end_init = time.time()
    print('Init models time:', end_init - begin_init, 'seconds')
    begin = time.time()

    list_img_path = get_list_file_in_folder(img_dir)
    list_img_path = sorted(list_img_path)       # List of image_names: [cccd_002.png, ...]
    print('=== list_img_path:\n', list_img_path)
    # print('=== Image_dir', img_dir)   return: /mnt/e/bkai/MC_OCR/modules/data/mc_ocr_private_test
    # print('=== Anno_dir', anno_dir)   #return: text file: /mnt/e/bkai/MC_OCR/modules/output_test/text_detector/mc_ocr_private_test/txt
    index = 2
    img_name = 'cccd_{}{}.png'
    for idx, img_name in enumerate(list_img_path):
        print('\n',idx,'Inference', img_name)
        test_img = cv2.imread(os.path.join(img_dir, img_name))  # Original image

        begin_detector = time.time()
        anno_path = os.path.join(anno_dir, img_name.replace(img_name.split('.')[-1], 'txt'))
        boxes_list = get_list_boxes_from_icdar(anno_path)

        # print('\n--- Boxes list get from anno to run ROTATION CORRECTOR ---')
        # print(boxes_list)   # [{'coors': [x1, y1, x2, y2, x3, y3, x4, y4], 'data': None}, ...]
        # print('drawing....')
        # draw_box_in_image(test_img, boxes_list, 1, img_name.split('.')[0], 'after_text_detector')
        # print('-----------\n')
        # boxes_list: [{'coors': [x1, y1, x2, y2, x3, y3, x4, y4], 'data': None}, ...], rot_drop_thresh = [.5, 2.]
        # usage: remove all boxes have 0.5 < w/h < 2
        boxes_list = drop_box(boxes_list, drop_gap=rot_drop_thresh) # Chắc là bỏ đi các box ko đúng

        # print('\n--- Boxes list after drop_box ---')
        # print(boxes_list)
        # draw_box_in_image(test_img, boxes_list, 2, img_name.split('.')[0], 'after_use_dropbox_fnc')
        # print('-----------\n')

        # quit(0)
        rotation = get_mean_horizontal_angle(img_name, test_img, boxes_list, False) # Góc để xoay ảnh: Tính theo chiều kim đồng hồ

        print('\n--- Rotation after get_mean_horizontal_angle ---')
        print('Angle = ', rotation)
        # print('math:', math.atan2(-9, -967))
        print('-----------\n')
        # @test_img: origin image
        # @boxes_list: ['coors': [x1, y1, ... x4, y4], 'data': None]
        # @rotation: angle to rotate horizontal (either reverse or not reverse)
        img_rotated, boxes_list = rotate_image_bbox_angle(True, idx, img_name, test_img, boxes_list, rotation)   # Xoay ảnh về theo chiều ngang, tuy nhiên chưa biết là ngược hay đúng theo chiều dọc

        # print('\n--- After rotate image with param (test_img, boxes_list, rotation) ---')
        # print('Image rotated: \n', img_rotated)
        # print('Boxes list : \n', boxes_list)
        # draw_box_in_image(img_rotated, boxes_list, 3, img_name.split('.')[0], 'after_call_rotate_fnc_with_rotation')
        # print('-----------\n')

        degre = calculate_page_orient(img_name,box_rectify, img_rotated, boxes_list) # Sử dụng box_rectify để xoay về cho đúng

        # print('Calculate degree: ', degre)

        img_rotated, boxes_list = rotate_image_bbox_angle(False, idx, img_name, img_rotated, boxes_list, degre)

        # print('\n--- After rotate image with param (img_rotated, boxes_list, degre) ---')
        # print('Image rotated: \n', img_rotated)
        # print('Boxes list : \n', boxes_list)
        # draw_box_in_image(img_rotated, boxes_list, 4, img_name.split('.')[0], 'after_call_rotate_fnc_with_degree')
        # print('-----------\n')

        boxes_list = filter_90_box(boxes_list)

        # print('\n--- Filter 90 box ---')
        # print('Boxes list : \n', boxes_list)
        # draw_box_in_image(img_rotated, boxes_list, 4, img_name.split('.')[0], 'after_rotation_corrector')
        # print('-----------\n')

        end_detector = time.time()
        print('get boxes from icdar time:', end_detector - begin_detector, 'seconds')

        output_txt_path = os.path.join(output_txt_dir, os.path.basename(img_name).split('.')[0] + '.txt')
        output_viz_path = os.path.join(output_viz_dir, os.path.basename(img_name))
        output_rotated_img_path = os.path.join(output_rotated_img_dir, os.path.basename(img_name).split('.')[0]+'(1).png')
        if write_rotated_img:
            cv2.imwrite(output_rotated_img_path, img_rotated)
        if write_file:
            write_output(boxes_list, output_txt_path)
        if visualize:
            viz_icdar(img_rotated, output_txt_path, output_viz_path)
            end_visualize = time.time()
            print('Visualize time:', end_visualize - end_detector, 'seconds')

    end = time.time()
    speed = (end - begin) / len(list_img_path)
    print('Processing time:', end - begin, 'seconds. Speed:', round(speed, 4), 'second/image')
    print('Done')
    # df = pd.DataFrame([[1, 2], [2, 3], [3, 4]], columns=['A', 'B'])
    # print(df)

def draw_box_in_image(img, boxes_list, order, index, img_type):     # img: read from cv2, coords: [x1, y1, x2, y2, x3, y3, x4, y4]
    coords = []
    imgcp = img.copy()
    for box in boxes_list:
        coords.append(box['coors'])
    for idx, coord in enumerate(coords):
        (x1, y1, x2, y2, x3, y3, x4, y4) = tuple(coord)
        print(coord)
        imgcp = cv2.polylines(imgcp, pts=[np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)
        imgcp = cv2.putText(imgcp, text='Box {}'.format(idx+1), org=(x1, y1-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(255, 0, 0), thickness=3)
    cv2.imwrite(filename='/mnt/e/bkai/MC_OCR/modules/1test_imgs/{}_{}_{}.png'.format(index, order, img_type), img=imgcp)

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


if __name__ == '__main__':
    # os.environ["DISPLAY"] = ":12.0"
    main()
