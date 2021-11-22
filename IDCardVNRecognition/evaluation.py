import time

import cv2
import numpy as np

from vietocr.tool.config import Cfg
from cropper import Cropper
from detector import Detector
from reader import OCR
from utils import download_weights, Config
from modules.config import *
from modules.text_detector.PaddleOCR.tools.infer.predict_det import detect_text
from modules.rotation_corrector.inference import rotate_horizontal_image
from modules.key_info_extraction.rulebase_extraction import string_similarity
import csv
import json
import Levenshtein as Lv
import re
import torch

# dir_path = os.path.dirname(os.path.realpath(__file__))
# static_path = os.path.join(dir_path, 'static')

# Model AI
cfg = Config.load_config()

cropper = Cropper(config_path=download_weights(cfg['cropper']['cfg']),
                  weight_path=download_weights(cfg['cropper']['weight']))

detector = Detector(config_path=download_weights(cfg['detector']['cfg']),
                    weight_path=download_weights(cfg['detector']['weight']))

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = cfg['reader']['weight']
config['cnn']['pretrained'] = False
config['device'] = torch.device('cpu') if gpu is None else 'cuda:'+str(gpu)
config['predictor']['beamsearch'] = False
reader = OCR(config)

ROOT_DIR = os.getcwd()
img_paths = os.path.join(ROOT_DIR, 'data/cccd_images')
ann_paths = os.path.join(ROOT_DIR, 'data/cccd_annos')


def correct(extracted, ann, rulebase=False):
    for e in extracted:
        if e is None:
            return False
    ext_name, ann_name = extracted[0].lower(), ann[0].lower()
    ext_birth, ann_birth = extracted[1], ann[1]
    ext_id, ann_id = extracted[2], ann[2]
    ext_gender, ann_gender = extracted[3], ann[3]

    if not rulebase:
        if (ext_id == ann_id) and (Lv.distance(ext_name, ann_name) <= 2) and (Lv.distance(ext_birth, ann_birth) <= 2) and (ext_gender == ann_gender):
            return True
    else:
        if (ext_id == ann_id) and (string_similarity(ext_name, ann_name) >= 0.7) and (string_similarity(ext_birth, ann_birth) >= 0.7) and (ext_gender == ann_gender):
            return True
    return False


keys = ['File name', 'Time Extraction 1', 'Time Extraction 2',
        'ID', 'Họ và tên', 'Ngày sinh', 'Giới tính', 'Quốc tịch', 'Dân tộc', 'Quê quán', 'Nơi thường trú', 'Ngày hết hạn']


def make_data_pathlist(imgs, anns):
    file_paths = []
    anno_paths = []
    for file_name in os.listdir(imgs):
        file_id = file_name.split('.')[0]
        file_paths.append(os.path.join(imgs, file_name))
        anno_paths.append(os.path.join(anns, file_id + '.json'))
    return file_paths, anno_paths


def extract(image_paths, ann_paths):
    file_paths, anno_paths = make_data_pathlist(image_paths, ann_paths)

    num_uncropped = 0
    num_unextracted = 0

    txt_dir_time = 0
    yolo_det_time = 0
    total_time = 0

    unable_to_extract = []
    wrong_extractions = []
    results = []

    for idx in range(len(file_paths)):
        # Read path
        img_path = file_paths[idx]
        ann_path = anno_paths[idx]
        _, file_name = os.path.split(file_paths[idx])
        print('\n ================{}/{}   Information Extraction: '.format(idx+1, len(file_paths)), file_name)
        # Read img
        begin = time.time()
        image = cv2.imread(img_path)
        is_card, is_id_card, warped = cropper.process(image=image)
        #
        use_text_direction = False
        if (is_card is False and is_id_card is None) or (is_id_card is not None and warped is None):
            # Text_detection
            print('Text detection')
            dt_boxes, _ = detect_text(img_path, image)
            # Rotate image
            warped, _, _ = rotate_horizontal_image(dt_boxes, image, file_name)
            use_text_direction = True
            num_uncropped += 1

        # print('{}/{}   File: '.format(idx+1, len(file_paths)), file_name, '         unable_to_crop: ', num_uncropped)
        # Detect obj keys:
        info_images, _ = detector.process(warped)
        if info_images is None:
            num_unextracted += 1
            unable_to_extract.append(file_name)
            print('Unable to extract\n')
            continue
        info = dict()
        # print('Info_images:\n', info_images)
        for id in list(info_images.keys()):
            # 7 is id of portrait class
            if id == 7:
                continue
            label = detector.i2label_cc[id]
            if isinstance(info_images[id], np.ndarray):
                info[label] = reader.predict(info_images[id])
            elif isinstance(info_images[id], list):
                info[label] = []
                for i in range(len(info_images[id])):
                    info[label].append(reader.predict(info_images[id][i]))

        info['nationality'] = 'Việt Nam'
        if 'sex' in info.keys():
            if 'Na' in info['sex']:
                info['sex'] = 'Nam'
            else:
                info['sex'] = 'Nữ'

        k = list(info.keys())
        info['File name'] = file_name
        info['ID'] = info.pop('id') if 'id' in k else None
        info['Họ và tên'] = info.pop('full_name') if 'full_name' in k else None
        info['Ngày sinh'] = info.pop('date_of_birth') if 'date_of_birth' in k else None
        info['Giới tính'] = info.pop('sex') if 'sex' in k else None
        info['Quốc tịch'] = info.pop('nationality')
        info['Dân tộc'] = info.pop('nation').replace('Dân tộc', '').replace(':', '').strip() if 'nation' in k else None
        info['Quê quán'] = ' '.join(info.pop('place_of_birth')) if 'place_of_birth' in k else None
        info['Nơi thường trú'] = ' '.join(info.pop('place_of_residence')) if 'place_of_residence' in k else None
        info['Ngày hết hạn'] = None
        if 'duration' in k:
            expire_str = info.pop('duration')
            pattern = r'[^\d]*([\d\/\s]+).?$'
            match = re.search(pattern, expire_str)
            info['Ngày hết hạn'] = match.group(1) if match is not None else expire_str

        end = time.time()
        duration = end-begin
        info['Time Extraction 1'] = None
        info['Time Extraction 2'] = None
        if use_text_direction:
            info['Time Extraction 1'] = round(duration, 5)
            txt_dir_time += duration
        else:
            info['Time Extraction 2'] = round(duration, 5)
            yolo_det_time += duration
        total_time += duration

        results.append(info)
        print('Extracted information: \n')
        indented = json.dumps(info, indent=2, ensure_ascii=False)
        print(indented)
        print('Writing to file...')
        with open(os.path.join(OUTPUT_ROOT, 'output1.csv'), 'a') as f:
            dictwriter_object = csv.DictWriter(f, fieldnames=keys)
            dictwriter_object.writerow(info)
            f.close()

        # Evaluate with annotation file:
        with open(ann_path, 'r') as f:
            ann = json.load(f)
            ann_info = [ann['fullName'], ann['birthdate'], ann['idNumber'], ann['gender']]
            corrected = correct([info['Họ và tên'], info['Ngày sinh'], info['ID'], info['Giới tính']], ann_info)
            if not corrected:
                wrong_extractions.append(info)
                with open(os.path.join(OUTPUT_ROOT, 'wrong_result.csv'), 'a') as f:
                    dictwriter_object = csv.DictWriter(f, fieldnames=keys)
                    dictwriter_object.writerow(info)
                    f.close()
        print('================================================\n')

    # Write result to file:
    with open(os.path.join(OUTPUT_ROOT, 'output2.csv'), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    print('Unable to crop: ', num_uncropped)
    print('Unable to extract: ', num_unextracted)
    with open(os.path.join(OUTPUT_ROOT, 'unable_to_extract.csv'), 'w') as f:
        for fn in unable_to_extract:
            f.write("%s\n" % fn)
    print('Average {} text direction extract: {}'.format(num_uncropped, txt_dir_time/num_uncropped))
    print('Average {} yolo detection extract: {}'.format(len(file_paths)-num_uncropped, 1.0*yolo_det_time/(len(file_paths)-num_uncropped)))
    print('Average {} extract: {}'.format(len(file_paths) - num_unextracted, 1.0*total_time/(len(file_paths) - num_unextracted)))
    print('Wrong result: ', len(wrong_extractions))
    wrong_fn = [info['File name'] for info in wrong_extractions]
    for fn in wrong_fn:
        print(fn)


def extract_information(image_path, ann_path):
    _, file_name = os.path.split(image_path)
    # Read img
    image = cv2.imread(image_path)
    is_card, is_id_card, warped = cropper.process(image=image)

    if (is_card is False and is_id_card is None) or (is_id_card is not None and warped is None):
        # # Add rotation corrector here
        dt_boxes = detect_text(image_path, image)
        warped, _ = rotate_horizontal_image(dt_boxes, image, file_name)

        cv2.imwrite(os.path.join(OUTPUT_ROOT, 'uncropped_one/{}'.format(file_name)), warped)
    print('File: ', file_name, '         Not cropped: ')
    info_images = detector.process(warped)
    if info_images is None:
        return None
    info = dict()
    # print('Info_images:\n', info_images)
    for id in list(info_images.keys()):
        # 7 is id of portrait class
        if id == 7:
            continue
        label = detector.i2label_cc[id]
        if isinstance(info_images[id], np.ndarray):
            info[label] = reader.predict(info_images[id])
        elif isinstance(info_images[id], list):
            info[label] = []
            for i in range(len(info_images[id])):
                info[label].append(reader.predict(info_images[id][i]))

    info['nationality'] = 'Việt Nam'
    if 'sex' in info.keys():
        if 'Na' in info['sex']:
            info['sex'] = 'Nam'
        else:
            info['sex'] = 'Nữ'
    print(json.dumps(info, indent=2, ensure_ascii=False))


# yolo_extract(img_paths, ann_paths)
# yolo_extract(os.path.join(img_paths, '001302025778.png'), None)

def rule_base_extract_img(img_path, image=None):
    info = {}
    # Read image:
    if image is None:
        image = cv2.imread(img_path)
    # Filename
    _, file_name = os.path.split(img_path)
    # Add rotation corrector
    dt_boxes = detect_text(os.path.join(raw_img_dir, file_name), image)
    # Rotate
    rotated_img, boxes_list = rotate_horizontal_image(dt_boxes, image, file_name)   # boxes_list = [{'coors':[], 'data': ','}...]

    # Text recognition:
    boxes = []  # return: [[x1, y1... x4, y4, Text, Prob], ...]
    for box in boxes_list:
        if isinstance(box, dict):
            box_data = box['coors']
        else:
            box_data = box
    return info


def rule_base_batch_extract(image_paths, anno_paths):
    file_paths, anno_paths = make_data_pathlist(image_paths, ann_paths)

    total_time = 0

    wrong_extractions = []
    results = []

    for idx in range(len(file_paths)):
        # Read path
        img_path = file_paths[idx]
        ann_path = anno_paths[idx]
        _, file_name = os.path.split(file_paths[idx])
        print('\n ================{}/{}   Information Extraction: '.format(idx + 1, len(file_paths)), file_name)
        # Read img
        begin = time.time()
        image = cv2.imread(img_path)
        # Extract:
        info = rule_base_extract_img(img_path, image)
        duration = time.time() - begin
        info['Time Extraction'] = duration

        results.append(info)
        print('Extracted information: \n')
        indented = json.dumps(info, indent=2, ensure_ascii=False)
        print(indented)
        print('Writing to file...')
        with open(os.path.join(OUTPUT_ROOT, 'output3.csv'), 'a') as f:
            dictwriter_object = csv.DictWriter(f, fieldnames=keys)
            dictwriter_object.writerow(info)
            f.close()

        # Evaluate with annotation file:
        with open(ann_path, 'r') as f:
            ann = json.load(f)
            ann_info = [ann['fullName'], ann['birthdate'], ann['idNumber'], ann['gender']]
            corrected = correct([info['Họ và tên'], info['Ngày sinh'], info['ID'], info['Giới tính']], ann_info)
            if not corrected:
                wrong_extractions.append(info)
                with open(os.path.join(OUTPUT_ROOT, 'wrong_result_rulebase.csv'), 'a') as f:
                    dictwriter_object = csv.DictWriter(f, fieldnames=keys)
                    dictwriter_object.writerow(info)
                    f.close()
        print('================================================\n')


if __name__ == "__main__":
    extract(img_paths, ann_paths)



