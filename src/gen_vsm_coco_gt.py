import os
import numpy as np
import json
import cv2

# ann_file = '../data/sample/predict_2labels.json'
# gt_json = json.load(open(ann_file))
# print(gt_json)
ROOT_PATH = '../data/cvat47'
DATA_PATH = 'images'
SKIP_FRAME = 0


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

    out = {'images': [], 'annotations': [],
           'categories': [{'id': 1, 'name': 'hs', 'supercategory': ''}]}
    data_path = os.path.join(ROOT_PATH, DATA_PATH)
    seqs = os.listdir(data_path)
    for i, seq in enumerate(sorted(seqs)):
        seq_path = os.path.join(data_path, seq)
        out = {'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'hs', 'supercategory': ''}]}

        gt_path = os.path.join(seq_path, 'gt_coco')
        if not os.path.exists(gt_path):
            os.mkdir(gt_path)

        ann_file = os.path.join(seq_path, 'annotations', 'instances_default.json')
        gt_json = json.load(open(ann_file))

        curr_img_id = -1
        # start_idx = int(len(gt_json['images']) / 2)
        start_idx = 0
        images = gt_json['images']
        out['images'] = gt_json['images']

        for j, ann in enumerate(gt_json['annotations']):
            img_id = int(ann['image_id'])
            if img_id <= start_idx:
                continue
            is_skip = False
            if SKIP_FRAME != 0:
                is_skip = True
                if curr_img_id == -1:
                    is_skip = False
                    curr_img_id = img_id
                elif img_id - curr_img_id == SKIP_FRAME:
                    is_skip = True
                else:
                    is_skip = False
                    curr_img_id = img_id

            if not is_skip:
                face_tlwh = ann['bbox']
                hs_tlwh = get_hs_tlwh(face_tlwh)
                hs_area = hs_tlwh[2] * hs_tlwh[3]
                ann['bbox'] = hs_tlwh
                ann['area'] = hs_area
                ann['category_id'] = 1
                out['annotations'].append(ann)
            else:
                print('Skip', img_id)

        json.dump(out, open(os.path.join(gt_path, 'gt.json'), 'w'))
        # print(seq)
    print(','.join(seqs))
