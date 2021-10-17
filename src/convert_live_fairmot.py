import os
import numpy as np
import json
import cv2
from bs4 import BeautifulSoup

ROOT_PATH = '../data'
ROOT_DATA_DIR = 'live_44_val'
DATA_PATH = 'images'
ANN_PATH = 'labels_with_ids'
SKIP_FRAME = 0
DATASET = 'live_44_val'
W, H = 1920, 1080


def get_hs_tlwh_fm(face_tlwh):
    t, l, w, h = face_tlwh[0], face_tlwh[1], face_tlwh[2], face_tlwh[3]
    new_cx = t + w / 2
    new_cy = l + h
    new_w, new_h = 2 * h, 2 * h
    # add border = h/4

    new_w += h / 2
    new_h += h / 2
    return new_cx / W, new_cy / H, new_w / W, new_h / H


def get_hs_tlwh(face_tlwh):
    t, l, w, h = face_tlwh[0], face_tlwh[1], face_tlwh[2], face_tlwh[3]
    new_t = t + w / 2 - h
    new_l = l
    new_w, new_h = 2 * h, 2 * h
    # add border = h/4
    new_t -= h / 4
    new_l -= h / 4
    new_w += h / 2
    new_h += h / 2
    return [new_t, new_l, new_w, new_h]


if __name__ == '__main__':
    root_data_path = os.path.join(ROOT_PATH, ROOT_DATA_DIR)
    data_path = os.path.join(root_data_path, DATA_PATH)

    ann_path = os.path.join(root_data_path, ANN_PATH)
    if not os.path.exists(ann_path):
        os.mkdir(ann_path)

    ds_cfg = os.path.join(root_data_path, DATASET + ".train")
    ds_cfg = open(ds_cfg, 'w')

    seqs = os.listdir(data_path)
    #
    for i, seq in enumerate(sorted(seqs)):
        print(seq)
        seq_path = os.path.join(data_path, seq)
        ann_file = os.path.join(seq_path, 'annotations.xml')
        seq_ann_path = os.path.join(ann_path, seq)
        if not os.path.exists(seq_ann_path):
            os.mkdir(seq_ann_path)
        seq_ann_path = os.path.join(seq_ann_path, "img1")
        if not os.path.exists(seq_ann_path):
            os.mkdir(seq_ann_path)

        out_json = {'images': [], 'annotations': [],
                    'categories': [{'id': 1, 'name': 'hs', 'supercategory': ''}]}

        gt_path = os.path.join(seq_path, 'gt_coco')
        if not os.path.exists(gt_path):
            os.mkdir(gt_path)

        with open(ann_file, 'r') as f:
            data = f.read()
        data = BeautifulSoup(data, "xml")
        images = data.find_all('image')
        ann_count = 1
        for image in images:
            img_id = int(image.get('id'))
            img_name = image.get('name')
            bboxes = image.find_all('box')
            if len(bboxes) > 0:
                out_json['images'].append(
                    {'id': img_id, 'width': W, 'height': H, 'file_name': img_name, 'license': 0,
                     'flickr_url': '', 'coco_url': '', 'date_captured': 0})
                ds_cfg.write(os.path.join(ROOT_DATA_DIR, DATA_PATH, seq, "img1", img_name) + "\n")

                fm_ann_out = os.path.join(seq_ann_path, img_name.replace('jpeg', 'txt'))
                fm_ann_out = open(fm_ann_out, 'w')

                for bbox in bboxes:
                    xtl, ytl, xbr, ybr = float(bbox.get('xtl')), float(bbox.get('ytl')), float(bbox.get('xbr')), float(
                        bbox.get('ybr'))
                    face_tlwh = [xtl, ytl, xbr - xtl, ybr - ytl]
                    hs_tlwh_fm = get_hs_tlwh_fm(face_tlwh)
                    hs_tlwh = get_hs_tlwh(face_tlwh)
                    out_json['annotations'].append(
                        {'id': ann_count, 'image_id': img_id, 'category_id': 1, 'segmentation': [], 'iscrowd': 0,
                         'bbox': hs_tlwh,
                         'area': hs_tlwh[2] * hs_tlwh[3], 'score': 1})
                    ann_count += 1
                    fm_ann = "0 -1 {:.2f} {:.2f} {:.2f} {:.2f}".format(hs_tlwh_fm[0], hs_tlwh_fm[1],
                                                                       hs_tlwh_fm[2], hs_tlwh_fm[3])
                    fm_ann_out.write(fm_ann + "\n")
                fm_ann_out.close()
        json.dump(out_json, open(os.path.join(gt_path, 'gt.json'), 'w'))
