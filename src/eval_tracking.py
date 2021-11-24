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
​
Tracking error:
Các tracks bị nhảy ids (output 1 bbox đại diện cho mỗi lần nhảy id)
Các tracks bị link nhầm (output 1 bbox đại diện cho mỗi lần link nhầm)
Trường hợp bị loss tracks sẽ được count là Loss detection ở trên
"""

IOU_THRES = 0.5

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_out_eval_dirs(eval_ds, output_path):
    pred_err = os.path.join(output_path, eval_ds, 'PRED_ERR')
    tracking_err = os.path.join(output_path, eval_ds, 'TRACKING_ERR')
    tracking_incorrect_linking = os.path.join(tracking_err, 'LINK')
    tracking_id_switches = os.path.join(tracking_err, 'IDSW')
    create_dir(pred_err)
    create_dir(tracking_incorrect_linking)
    create_dir(tracking_id_switches)

    return pred_err, tracking_incorrect_linking, tracking_id_switches

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None, text_color=(0, 0, 255)):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w / 140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i].split('_')[-1])
        id_text = obj_ids[i].split('_')[0]
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = vis.get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)

        if i == 1:
            text_color = (0, 255, 255)

        if id_text == 'FN':
            text_color = (0, 255, 255)
        elif id_text == 'FP':
            text_color = (0, 0, 255)

        cv2.putText(im, id_text, (intbox[0], intbox[3] + 10), cv2.FONT_HERSHEY_PLAIN, text_scale, text_color,
                    thickness=text_thickness)
    return im

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
        tlbr = [t, l, t + w, l + h]
        bboxes.append(dict(tlbr=tlbr,
                           tlwh=[t, l, w, h]))

    return track_ids, bboxes


def get_track_route(map_pred_gt_track, gt_track):
    hist = [str(id) for id in map_pred_gt_track[gt_track]]
    return str(gt_track) + '-' + '-'.join(hist)


def process_one_set(images_path, eval_path, output_path):
    eval_ds = eval_path.split("/")[-1]
    pred_err, tracking_incorrect_linking, tracking_id_switches = create_out_eval_dirs(eval_ds, output_path)

    # images_path = os.path.join(data_path, 'gt', 'images')
    label_gt_path = os.path.join(eval_path, 'gt', 'labels')
    label_pred_path = os.path.join(eval_path, 'predict', 'labels')
    map_gt_labels = read_label_files(label_gt_path)
    map_pred_labels = read_label_files(label_pred_path)

    # Predict error
    false_negative = {}
    false_positive = {}
    pred_errs = []
    # Tracking error
    map_pred_gt_track = {}
    err_new_track = []
    err_wrong_link = []
    idsw_cases = []
    inlink_cases = []
    existed_pred_tracks = []

    for i, image in enumerate(map_gt_labels.keys()):
        if i > 400:
            break
        gt_labels = map_gt_labels[image]
        pred_labels = map_pred_labels[image]

        gt_track_ids, gt_bboxes = get_bboxes_and_track(gt_labels)
        pred_track_ids, pred_bboxes = get_bboxes_and_track(pred_labels)
        # Do some magic here, match bbox!
        gt_tlbrs = [np.array(gt_ann['tlbr']) for gt_ann in gt_bboxes]
        pred_tlbrs = [np.array(pred_ann['tlbr']) for pred_ann in pred_bboxes]
        cost_matrix = matching.iou_distance(gt_tlbrs, pred_tlbrs)
        matches, u_gt, u_pred = matching.linear_assignment(cost_matrix, thresh=IOU_THRES)

        fn_tlwhs = []
        fp_tlwhs = []
        # Update predict error
        for ele in u_gt:
            if image not in false_negative:
                false_negative[image] = [(gt_track_ids[ele], gt_bboxes[ele])]
            else:
                false_negative[image].append((gt_track_ids[ele], gt_bboxes[ele]))
            fn_tlwhs.append(gt_bboxes[ele]['tlwh'])
        for ele in u_pred:
            if image not in false_positive:
                false_positive[image] = [(pred_track_ids[ele], pred_bboxes[ele])]
            else:
                false_positive[image].append((pred_track_ids[ele], pred_bboxes[ele]))
            fp_tlwhs.append(pred_bboxes[ele]['tlwh'])

        if len(fn_tlwhs) > 0 or len(fp_tlwhs) > 0:
            pred_errs.append((image, fn_tlwhs, fp_tlwhs))
        # print('false_negative', false_negative)
        # print('false_positive', false_negative)
        # Update tracking error
        # Update map track btw gt and pred here

        idsw_ids = []
        idsw_bboxes = []
        inlink_ids = []
        inlink_bboxes = []

        for m_gt, m_pred in matches:
            gt_track = gt_track_ids[m_gt]
            gt_tlwh = gt_bboxes[m_gt]['tlwh']
            pred_track = pred_track_ids[m_pred]
            pred_tlwh = pred_bboxes[m_pred]['tlwh']
            if gt_track not in map_pred_gt_track:
                if pred_track in existed_pred_tracks:  # incorrect linking
                    inlink_ids.append(str(gt_track) + "-new_" + str(gt_track))
                    inlink_bboxes.append(gt_tlwh)
                    inlink_ids.append(str(pred_track) + "_" + str(pred_track))
                    inlink_bboxes.append(pred_tlwh)
                map_pred_gt_track[gt_track] = [pred_track]
                existed_pred_tracks.append(pred_track)
            else:
                if pred_track not in existed_pred_tracks:  # ID switch
                    idsw_ids.append(get_track_route(map_pred_gt_track, gt_track) + "_" + str(gt_track))
                    idsw_bboxes.append(gt_tlwh)
                    idsw_ids.append(str(pred_track) + "_" + str(pred_track))
                    idsw_bboxes.append(pred_tlwh)
                    map_pred_gt_track[gt_track].append(pred_track)
                    existed_pred_tracks.append(pred_track)
                else:
                    last_associated_track = map_pred_gt_track[gt_track][-1]
                    # if pred_track != last_associated_track and pred_track not in map_pred_gt_track[gt_track]:  # incorrect linking
                    if pred_track != last_associated_track:  # incorrect linking
                        inlink_ids.append(get_track_route(map_pred_gt_track, gt_track) + "_" + str(gt_track))
                        inlink_bboxes.append(gt_tlwh)
                        inlink_ids.append(str(pred_track) + "_" + str(pred_track))
                        inlink_bboxes.append(pred_tlwh)
                        map_pred_gt_track[gt_track].append(pred_track)

        if len(idsw_ids) > 0:
            idsw_cases.append((image, idsw_ids, idsw_bboxes))
        if len(inlink_ids) > 0:
            inlink_cases.append((image, inlink_ids, inlink_bboxes))
    # print('map_pred_gt_track', map_pred_gt_track)
    # if i == 100:
    #     raise 1==2

    for image, fn_tlwhs, fp_tlwhs in pred_errs:
        fn_ids = ['FN_0' for _ in fn_tlwhs]
        fp_ids = ['FP_1' for _ in fp_tlwhs]
        ids = fn_ids
        ids.extend(fp_ids)
        tlwhs = fn_tlwhs
        tlwhs.extend(fp_tlwhs)
        img = cv2.imread(os.path.join(images_path, image + '.jpg'))
        err_frame = plot_tracking(img, tlwhs, ids, frame_id=int(image), fps=20)
        cv2.imwrite(os.path.join(pred_err, image + '.jpg'), err_frame)

    for image, idsw_ids, idsw_bboxes in idsw_cases:
        img = cv2.imread(os.path.join(images_path, image + '.jpg'))
        err_frame = plot_tracking(img, idsw_bboxes, idsw_ids, frame_id=int(image), fps=20)
        cv2.imwrite(os.path.join(tracking_id_switches, image + '.jpg'), err_frame)

    for image, idsw_ids, idsw_bboxes in inlink_cases:
        img = cv2.imread(os.path.join(images_path, image + '.jpg'))
        err_frame = plot_tracking(img, idsw_bboxes, idsw_ids, frame_id=int(image), fps=20)
        cv2.imwrite(os.path.join(tracking_incorrect_linking, image + '.jpg'), err_frame)

    total_fn = 0
    total_fp = 0
    for key, value in false_negative.items():
        total_fn += len(value)
    for key, value in false_positive.items():
        total_fp += len(value)
    print("Number of false negative bboxes", total_fn)
    print("Number of false positive bboxes", total_fp)

def main(images_path, eval_path, output_path):
    # eval_sets = ['set-3', 'set-7', 'set-8', 'set-12', 'set-13', 'set-16', 'set-18']
    eval_sets = ['set-0', 'set-1', 'set-2', 'set-4', 'set-5', 'set-6', 'set-9', 'set-10', 'set-11', 'set-14', 'set-15', 'set-17', 'set-19', 'set-20', 'set-21']
    for eval_set in eval_sets:
        images_set = os.path.join(images_path, eval_set, 'img1')
        eval_set = os.path.join(eval_path, eval_set)
        process_one_set(images_set, eval_set, output_path)

if __name__ == "__main__":
    images_path = "/home/namtd/workspace/projects/smartcity/src/multiple-tracking/dataset/eval/LiveTrack_train/images/train/"
    eval_path = "/home/namtd/workspace/projects/smartcity/src/multiple-tracking/dataset/eval/LiveTrack_train/images/outputs/default_val/"
    output_path = "/home/namtd/Desktop/lab/eval_track/eval_results/eval_train_silver1.2"
    main(images_path, eval_path, output_path)