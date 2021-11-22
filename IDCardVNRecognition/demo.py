# -*- coding: utf-8 -*-

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
import json
import re
import streamlit as st
from PIL import ImageFont, ImageDraw, Image

cfg = Config.load_config()

cropper = Cropper(config_path=download_weights(cfg['cropper']['cfg']),
                  weight_path=download_weights(cfg['cropper']['weight']))

detector = Detector(config_path=download_weights(cfg['detector']['cfg']),
                    weight_path=download_weights(cfg['detector']['weight']))

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = cfg['reader']['weight']
config['cnn']['pretrained'] = False
config['device'] = 'cpu' if gpu is None else 'cuda:'+str(gpu)
config['predictor']['beamsearch'] = False
reader = OCR(config)

ROOT_DIR = os.getcwd()
img_paths = os.path.join(ROOT_DIR, 'data/cccd_images')
ann_paths = os.path.join(ROOT_DIR, 'data/cccd_annos')

keys = ['Tên file', 'Time Extraction',
        'ID', 'Họ và tên', 'Ngày sinh', 'Giới tính', 'Quốc tịch', 'Dân tộc', 'Quê quán', 'Nơi thường trú', 'Ngày hết hạn']

mapping = {
    'id': 'ID',
    'full_name': 'Họ và tên',
    'date_of_birth': 'Ngày sinh',
    'sex': 'Giới tính',
    'nationality': 'Quốc tịch',
    'nation': 'Dân tộc',
    'portrait': 'portrait',
    'duration': 'Ngày hết hạn',
    'place_of_birth': 'Quê quán',
    'place_of_residence': 'Nơi thường trú'
}

######
#    @:return extracted information from image. Return None if cannot extract
#    @:param:
#        image: image read from opencv
#        file_name: name of the file
#        vis_res (optional): visualized images from extraction processing
######

def extract(image, file_name, vis_res=None):
    global keys, mapping
    if vis_res is not None and isinstance(vis_res, dict):
        vis_res.update({
            'cropped_image': None,
            'text_detection': None,
            'rectified_image': None,
            'detect_key': None,
            'text_recognition': None
        })
    print('\n ================   Information Extraction: ', file_name)
    begin = time.time()
    use_rectify_text = False
    is_card, is_id_card, warped = cropper.process(image=image)

    if (is_card is False and is_id_card is None) or (is_id_card is not None and warped is None):
        # # Add rotation corrector here
        dt_boxes, vis_det_img = detect_text(None, image)
        warped, _, vis_rot_img = rotate_horizontal_image(dt_boxes, image, file_name)
        use_rectify_text = True

    info_images, info_box = detector.process(warped)
    for k, v in info_box.items():
        info_box[k] = v.tolist()

    # print('Info images', info_images)
    if info_images is None:
        return None

    info = dict()
    # print('Info_images:\n', info_images)
    for id in list(info_images.keys()):
        # 7 is id of portrait class
        label = detector.i2label_cc[id]
        info_box[label] = info_box.pop(id)
        if id == 7:
            continue

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

    ext_key = list(info.keys())
    tmp = dict()

    for k, v in mapping.items():
        if k in ext_key:
            if k == 'nation':
                info[k] = info[k].replace('Dân tộc', '').replace(':', '').strip()
            if k == 'place_of_birth' or k == 'place_of_residence':
                info[k] = ' '.join(info[k])
            if k == 'duration':
                txt = info[k]
                date_pattern = r'[^\d]*([\d\/\s]+).?$'
                match = re.search(date_pattern, txt)
                info[k] = match.group(1) if match is not None else txt
            tmp.update({k: info[k]})
            info[v] = info.pop(k)

    info['Tên file'] = file_name
    info['Extraction Time'] = str(round(time.time() - begin, 3)) + '(s)'

    if vis_res is not None:
        if not use_rectify_text:
            vis_res['cropped_image'] = warped
            # cv2.imwrite('/mnt/e/bkai/IDCardVNRecognition/crop.jpg', warped)
        else:
            vis_res['text_detection'] = vis_det_img
            vis_res['rectified_image'] = vis_rot_img
            # cv2.imwrite('/mnt/e/bkai/IDCardVNRecognition/detection.jpg', vis_det_img)
            # cv2.imwrite('/mnt/e/bkai/IDCardVNRecognition/rectify.jpg', vis_rot_img)

        vis_det_key = draw_det_key_box(info_box, warped.copy(), texts=None)
        vis_rec = draw_det_key_box(info_box, warped.copy(), texts=tmp)
        # vis_res['detect_key'] = vis_det_key
        # vis_res['text_recognition'] = vis_rec

    print(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def draw_det_key_box(boxes, img, texts=None):
    global mapping
    font_path = "Vietnamese.ttf"
    font = ImageFont.truetype(font_path, 32)

    for label in boxes.keys():
        if label == 'portrait':
            continue
        box = np.array(boxes[label], dtype=np.int32)
        text = mapping[label]
        if texts:
            text += ': ' + texts[label]
        if box.ndim == 1:
            (xmin, ymin, xmax, ymax) = box[0], box[1], box[2], box[3]
        else:
            (xmin, ymin) = tuple(np.min(box[:, :2], axis=0, keepdims=False))
            (xmax, ymax) = tuple(np.max(box[:, 2:4], axis=0, keepdims=False))
        img = cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((xmin, ymin-20), text=text, font=font, fill= (255, 0, 0, 0))
        img = np.array(img_pil)
    return img


def main():
    st.title('Identity Card Information Extraction Demo')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        print('uploaded_file', uploaded_file)
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption='Uploaded Image.', width=500)
        st.write("")
        ph1 = st.empty()
        ph1.info("Extracting...")
        ph2 = st.empty()
        ph2.write('')

        vis_res = dict()
        info = extract(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), uploaded_file.name, vis_res=vis_res)

        if info is None:
            st.error('Extraction failed.')
        else:
            ph1.success('Extraction completed.')
            ph2.write(info)

            st.info('Cropping image...')
            time.sleep(0.5)
            if vis_res['cropped_image'] is not None:
                st.image(vis_res['cropped_image'], width=500)

            else:
                st.error('Cropping image failed.')
                st.info('Rectify image...')
                time.sleep(0.5)
                det_img = Image.fromarray(vis_res['text_detection'])
                rect_img = Image.fromarray(vis_res['rectified_image'])

                st.write('Detecting text...')
                st.image([det_img], width=400)
                st.write('Rectify...')
                st.image([rect_img], width=400)

            det_key = Image.fromarray(vis_res['detect_key'])
            rec_img = Image.fromarray(vis_res['text_recognition'])

            st.info('Detecting keys...')
            time.sleep(0.5)
            st.image(det_key, width=500)

            st.info('Recognize text...')
            time.sleep(0.5)
            st.image(rec_img, width=500)


if __name__ == '__main__':
    main()