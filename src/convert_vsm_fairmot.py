import os
import numpy as np
import json
import cv2

ROOT_PATH = '../data'
ROOT_DATA_DIR = 'cvat47'
DATA_PATH = 'images'
ANN_PATH = 'labels_with_ids'
SKIP_FRAME = 0
DATASET = 'cvat47'


def get_hs_tlwh(face_tlwh):
    t, l, w, h = face_tlwh[0], face_tlwh[1], face_tlwh[2], face_tlwh[3]
    new_cx = t + w / 2
    new_cy = l + h
    new_w, new_h = 2 * h, 2 * h
    # add border = h/4

    new_w += h / 2
    new_h += h / 2
    return new_cx / 1920, new_cy / 1080, new_w / 1920, new_h / 1080


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
        seq_path = os.path.join(data_path, seq)
        ann_file = os.path.join(seq_path, 'annotations/instances_default.json')
        gt_json = json.load(open(ann_file))
        seq_ann_path = os.path.join(ann_path, seq)
        if not os.path.exists(seq_ann_path):
            os.mkdir(seq_ann_path)
        seq_ann_path = os.path.join(seq_ann_path, "img1")
        if not os.path.exists(seq_ann_path):
            os.mkdir(seq_ann_path)

        fairmot_anns = {}
        for j, ann in enumerate(gt_json['annotations']):
            img_id = int(ann['image_id'])
            face_tlwh = ann['bbox']
            hs_cxcywh = get_hs_tlwh(face_tlwh)
            track_id = int(ann['attributes']['tracking_id']) + 1
            fairmot_ann = "0 {:d} {:.2f} {:.2f} {:.2f} {:.2f}".format(track_id, hs_cxcywh[0], hs_cxcywh[1],
                                                                      hs_cxcywh[2],
                                                                      hs_cxcywh[3])
            if img_id not in fairmot_anns:
                fairmot_anns[img_id] = [fairmot_ann]
            else:
                fairmot_anns[img_id].append(fairmot_ann)

        images = gt_json['images']
        curr_img_id = -1
        # start_idx = int(len(images) / 2)
        start_idx = 0
        for img in images:
            img_id = int(img['id'])
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
                if img_id in fairmot_anns:
                    ds_cfg.write(os.path.join(ROOT_DATA_DIR, DATA_PATH, seq, "img1", img['file_name']) + "\n")
                    ann_out = os.path.join(seq_ann_path, img['file_name'].replace('jpeg', 'txt'))
                    ann_out = open(ann_out, 'w')
                    for ann in fairmot_anns[img_id]:
                        ann_out.write(ann + "\n")
                    ann_out.close()
            else:
                print('Skip', img_id)
    # print(seq)
    print(','.join(seqs))
