from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2

import numpy as np
import cv2
import _init_paths
from tracker import matching
from tracking_utils import visualization as vis

import os

"""
Detection error: (iou < 50% là coi như không match)
False negative: Các bbox có trong pred nhưng không có trong gt (Loss detection)
False positive: Các bbox có trong gt nhưng không có trong pred

Tracking error:
Các tracks bị nhảy ids (output 1 bbox đại diện cho mỗi lần nhảy id)
Các tracks bị link nhầm (output 1 bbox đại diện cho mỗi lần link nhầm)
Trường hợp bị loss tracks sẽ được count là Loss detection ở trên
"""

IOU_THRES = 0.5

def vis_err(image_path, frame_id, tlwhs, track_ids):
    image = cv2.imread(image_path)
    online_im, bboxes_im = vis.plot_tracking(image, tlwhs, track_ids, frame_id=frame_id)
    return online_im, bboxes_im

def read_label_files(label_path):
    "Read and save label to dict."
    label_files = sorted(os.listdir(label_path))
    map_labels = {}
    for label_file in label_files:
        image_name = label_file.replace('.txt', '')
        map_labels[image_name] = []
        file = open(os.path.join(label_path, label_file), "r")
        for x in file:
            label = x.replace('\n', '').split(',')
            label = [int(float(ele)) for ele in label]
            map_labels[image_name].append(label)

    return map_labels

def get_bboxes_and_track(gt_labels):
    track_ids = []
    bboxes = []
    for label in gt_labels:
        track, t, l, w, h = label
        track_ids.append(track)
        # Convert tlwh to tlbr
        tlbr = [t, l, t+w, l+h]
        bboxes.append(dict(tlbr=tlbr,
                           tlwh=[t, l, w, h]))

    return track_ids, bboxes

def main(data_path):
    label_gt_path = os.path.join(data_path, 'gt', 'labels')
    label_pred_path = os.path.join(data_path, 'predict', 'labels')
    map_gt_labels = read_label_files(label_gt_path)
    map_pred_labels = read_label_files(label_pred_path)


    # Predict error
    false_negative = {}
    false_positive = {}
    # Tracking error
    map_pred_gt_track = {}
    err_new_track = []
    err_wrong_link = []

    for i, image in enumerate(map_gt_labels.keys()):
        # if i == 100:
        #     break
        gt_labels = map_gt_labels[image]
        pred_labels = map_pred_labels[image]

        gt_track_ids, gt_bboxes = get_bboxes_and_track(gt_labels)
        pred_track_ids, pred_bboxes = get_bboxes_and_track(pred_labels)
        # Do some magic here, match bbox!
        gt_tlbrs = [np.array(gt_ann['tlbr']) for gt_ann in gt_bboxes]
        pred_tlbrs = [np.array(pred_ann['tlbr']) for pred_ann in pred_bboxes]
        cost_matrix = matching.iou_distance(gt_tlbrs, pred_tlbrs)
        matches, u_gt, u_pred = matching.linear_assignment(cost_matrix, thresh=IOU_THRES)

        # Update predict error
        for ele in u_gt:
            if image not in false_negative:
                false_negative[image] = [(gt_track_ids[ele], gt_bboxes[ele])]
            else:
                false_negative[image].append((gt_track_ids[ele], gt_bboxes[ele]))
        for ele in u_pred:
            if image not in false_positive:
                false_positive[image] = [(pred_track_ids[ele], pred_bboxes[ele])]
            else:
                false_positive[image].append((pred_track_ids[ele], pred_bboxes[ele]))
        # print('false_negative', false_negative)
        # print('false_positive', false_negative)
        # Update tracking error
        # Update map track btw gt and pred here
        for m_gt, m_pred in matches:
            gt_track = gt_track_ids[m_gt]
            pred_track = pred_track_ids[m_pred]
            if pred_track not in map_pred_gt_track:
                if i != 0:
                    err_new_track.append((image, pred_track))
                map_pred_gt_track[pred_track] = {gt_track}
            else:
                if i != 0:
                    if gt_track not in map_pred_gt_track[pred_track]:
                        err_wrong_link.append((image, pred_track))
                map_pred_gt_track[pred_track].add(gt_track)

        # print('map_pred_gt_track', map_pred_gt_track)
        # if i == 100:
        #     raise 1==2
    print("Number of images has false negative ", len(false_negative.keys()))
    print("Number of images has false positive ", len(false_positive.keys()))

    print("Number of images has err new track ", len(err_new_track))
    print("Number of images has err wrong link track ", len(err_wrong_link))
    print('err_new_track', err_new_track)
    print('err_wrong_link', err_wrong_link)
    # TODO convert tlwh bbox to tlbr bbox

    # TODO read the label file


if __name__ == "__main__":
    eval_data = "/home/namtd/workspace/projects/smartcity/src/multiple-tracking/dataset/eval/LiveTrack/images/outputs/default_val/set-3"
    main(eval_data)