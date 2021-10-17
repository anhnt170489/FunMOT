import os
import numpy as np
import json
import cv2

DATA_PATH = '../data/test_face_matching/images'
SKIP_FRAME = 0
if __name__ == '__main__':
    seqs = os.listdir(DATA_PATH)
    for i, seq in enumerate(sorted(seqs)):
        seq_path = '{}/{}/'.format(DATA_PATH, seq)
        gt_path = seq_path + 'gt/'
        if not os.path.exists(gt_path):
            os.mkdir(gt_path)

        ann_file = seq_path + 'annotations/instances_default.json'
        gt_json = json.load(open(ann_file))
        gt = {}
        gt_out = gt_path + '/gt.txt'
        fout = open(gt_out, 'w')
        curr_img_id = -1
        # start_idx = int(len(gt_json['images']) / 2)
        start_idx = 0
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
                tlwh = ann['bbox']
                track_id = int(ann['attributes']['tracking_id']) + 1
                fout.write(
                    '{:d},{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:d},{:d},{:d},{:.6f}\n'.format(
                        int(img_id), int(track_id), tlwh[0], tlwh[1], tlwh[2], tlwh[3], 1, -1, -1, -1))
            else:
                print('Skip', img_id)
        fout.close()
        # print(seq)
    print(','.join(seqs))
