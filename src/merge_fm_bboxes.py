import os
import numpy as np
import cv2

import _init_paths
from tracker import matching

ROOT = '../out/videos/AN12_0_5_pretrained'
frames = 'org'
fm_anns = 'results.txt'
body_bboxes = 'labels.txt'
err_dir = os.path.join(ROOT, 'err')
if not os.path.exists(err_dir):
    os.mkdir(err_dir)

# Blue color in BGR
FM_COLOR = (255, 0, 0)
BODY_COLOR = (0, 255, 0)

# Line thickness of 2 px
THICKNESS = 2

fm_anns = np.loadtxt(os.path.join(ROOT, fm_anns))
body_bboxes = np.loadtxt(os.path.join(ROOT, body_bboxes))
# print(fm_anns)
# print(bboxes)

fm_anns_map = {}
bboxes_map = {}


def draw_rectangle(img, bbox, color):
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, THICKNESS)


for ann in fm_anns:
    fid = int(ann[0])
    track_id = int(ann[1])
    bboxes = ann[2:6]
    bboxes[2:] = bboxes[:2] + bboxes[2:]
    ann = {'fid': fid, 'track_id': track_id, 'bboxes': bboxes}
    if fid not in fm_anns_map:
        fm_anns_map[fid] = [ann]
    else:
        fm_anns_map[fid].append(ann)

for ann in body_bboxes:
    fid = int(ann[1])
    bboxes = ann[2:6]
    ann = {'fid': fid, 'bboxes': bboxes}
    if fid not in bboxes_map:
        bboxes_map[fid] = [ann]
    else:
        bboxes_map[fid].append(ann)

# print(fm_anns_map)
# print(bboxes_map)
frames_idr = os.path.join(ROOT, frames)
frames = sorted(os.listdir(frames_idr))
for i, frame in enumerate(frames):
    if not frame.endswith('.jpg') and not frame.endswith('.png'):
        continue
    fid = i + 1
    img = cv2.imread(os.path.join(frames_idr, frame))
    if fid in fm_anns_map and fid in bboxes_map:
        fm_anns = fm_anns_map[fid]
        bboxes = bboxes_map[fid]
        fm_bboxes = [np.array(fm_ann['bboxes']) for fm_ann in fm_anns]
        body_bboxes = [np.array(ann['bboxes']) for ann in bboxes]
        _ious = matching.ious(fm_bboxes, body_bboxes)
        cost_matrix = 1 - _ious
        matches, u_fm, u_bboxes = matching.linear_assignment(cost_matrix, thresh=1)
        # print(matches, u_fm, u_bboxes)
        for idx in u_fm:
            bbox = fm_bboxes[idx]
            draw_rectangle(img, bbox, FM_COLOR)
        for idx in u_bboxes:
            bbox = body_bboxes[idx]
            draw_rectangle(img, bbox, BODY_COLOR)
        # for match in matches:
        #     online_idx, gt_idx = match
        #     online_head_tlwhs[online_idx] = gt_tlhws[gt_idx]
        cv2.imwrite(os.path.join(err_dir, frame), img)

    else:
        if fid in fm_anns_map:
            fm_bboxes = [fm_ann['bboxes'] for fm_ann in fm_anns]
            for bbox in fm_bboxes:
                draw_rectangle(img, bbox, FM_COLOR)
        if fid in bboxes_map:
            body_bboxes = [ann['bboxes'] for ann in bboxes]
            for bbox in body_bboxes:
                draw_rectangle(img, bbox, BODY_COLOR)
        cv2.imwrite(os.path.join(err_dir, frame), img)
