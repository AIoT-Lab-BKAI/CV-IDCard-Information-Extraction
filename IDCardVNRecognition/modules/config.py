import os

CONFIG_ROOT = os.path.dirname(__file__)
OUTPUT_ROOT = ''


def full_path(sub_path, file=False):
    path = os.path.join(CONFIG_ROOT, sub_path)
    if not file and not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('full_path. Error makedirs',path)
    return path


def output_path(sub_path):
    path = os.path.join(OUTPUT_ROOT, sub_path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('output_path. Error makedirs',path)
    return path

gpu = None  # None or 0,1,2...

# text detector
det_model_dir = full_path('text_detector/PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer')
det_visualize = False
det_db_thresh = 0.3
det_db_box_thresh = 0.3

# rotation corrector
rot_drop_thresh = [.5, 2]
rot_visualize = False
rot_model_path = full_path('rotation_corrector/weights/mobilenetv3-Epoch-487-Loss-0.03-Acc-0.99.pth', file=True)

# text classifier (OCR)
cls_visualize = False
cls_ocr_thres = 0.65
cls_model_path = full_path('text_classifier/vietocr/vietocr/weights/vgg19_bn_seq2seq.pth', file=True)
cls_base_config_path = full_path('text_classifier/vietocr/app/base.yml', file=True)
cls_config_path = full_path('text_classifier/vietocr/app/vgg-seq2seq.yml', file=True)


# key information
kie_visualize = True
kie_model = full_path('key_info_extraction/PICK/saved/models/PICK_Default/test_0121_212713/model_best.pth', file=True)

