from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import motmetrics as mm
import logging
import os
import os.path as osp
import json
import torch
from progress.bar import Bar

import datasets.dataset.jde as datasets
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.evaluation import Evaluator
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.utils import mkdir_if_missing
from tracker import matching
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tracking_utils.coco_eval_smartcity import COCOeval2

SKIP_FRAME = 0
DEBUG = True


def handle_hs_file(gt_hs_path):
    gt_hs = {}
    set_track = set()
    map_track = {}
    with open(gt_hs_path) as file:
        for line in file:
            line = line.rstrip().split(',')
            image_id, track_id, t, l, w, h, _, _, _ = line
            set_track.add(int(track_id))
    for i, track_id in enumerate(sorted(set_track)):
        map_track[str(track_id)] = str(i + 1)

    with open(gt_hs_path) as file:
        for line in file:
            line = line.rstrip().split(',')
            image_id, track_id, t, l, w, h, _, _, _ = line
            tlwh = [int(t), int(l), int(w), int(h)]
            if int(image_id) in gt_hs:
                gt_hs[int(image_id)].append((map_track[track_id], tlwh))
            else:
                gt_hs[int(image_id)] = [(map_track[track_id], tlwh)]

    return gt_hs

def save_labels(save_label_dir, tlwhs, ids, label_name):
    lines = []
    for tlwh, id_ in zip(tlwhs, ids):
        t, l, w, h = tlwh
        # [     1218.6     -21.185      105.42      104.18] 1
        line = [id_, t, l, w, h]
        line = [str(int(i)) for i in line]
        line = ','.join(line) + '\n'
        lines.append(line)
    # lines = ['Readme', 'How to write text files in Python']
    with open(os.path.join(save_label_dir, label_name), 'w') as f:
        for line in lines:
            f.write(line)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Validator:
    def __init__(self, opt, model=None, det_only=False, pred_only=False, fps=30):
        self.opt = opt
        f = open(opt.data_cfg)
        data_config = json.load(f)
        if det_only:
            val_paths = data_config['val_det']
        else:
            val_paths = data_config['val_ids']
        if 'root' in val_paths:
            data_root = val_paths['root']
        else:
            data_root = data_config['root']
        seqs_str = val_paths['seqs']
        val_dir = val_paths['val_dir']
        f.close()
        self.seqs = [seq.strip() for seq in seqs_str.split(',')]
        self.data_root = os.path.join(data_root, val_dir)
        self.val_ds = val_paths['val_ds']
        self.model = model
        self.det_only = det_only
        self.pred_only = pred_only
        self.fps = fps

    def get_hs_tlwh(self, face_tlwh):
        t, l, w, h = face_tlwh[0], face_tlwh[1], face_tlwh[2], face_tlwh[3]
        new_t = t + w / 2 - h
        new_l = l
        new_w, new_h = 2 * h, 2 * h
        # add border = h/4
        new_t -= h / 4
        new_l -= h / 4
        new_w += h / 2
        new_h += h / 2

        return new_t, new_l, new_w, new_h

    def write_results(self, filename, results, data_type):
        if data_type == 'mot':
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        elif data_type == 'kitti':
            save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
        else:
            raise ValueError(data_type)

        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids in results:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(
                        frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    f.write(line)

    def write_coco_preds(self, filename, images, results):
        # out = {'images': gt['images'], 'annotations': gt['annotations'],
        out = {'images': images, 'annotations': [],
               'categories': [{'id': 1, 'name': 'hs', 'supercategory': ''}]}
        det_count = 1
        for i, dets in enumerate(results):
            img_id = dets[0]
            dets = dets[1]
            for det in dets:
                ann = {'id': det_count, 'image_id': img_id,
                       'category_id': 1, 'segmentation': [], 'iscrowd': 0}
                det_count += 1
                # w, h = det[2] - det[0], det[3] - det[1]
                w, h = det[2], det[3]
                ann['bbox'] = [float(det[0]), float(
                    det[1]), float(w), float(h)]
                ann['area'] = float(w * h)
                # ann['score'] = float(det[4])
                ann['score'] = 1
                out['annotations'].append(ann)

        json.dump(out, open(filename, 'w'))

    def get_head_area_bbox(self, bbox_tlwh):
        tl_x, tl_y, w, h = bbox_tlwh[0], bbox_tlwh[1], bbox_tlwh[2], bbox_tlwh[3]
        c_x, c_y = tl_x + w / 2, tl_y + h / 2
        head_area_bbox_tlbr = [c_x - h / 4, tl_y, c_x + h / 4, c_y]
        head_area_bbox_tlwh = [c_x - h / 4, tl_y, h / 2, h / 2]
        return head_area_bbox_tlbr, head_area_bbox_tlwh

    def eval_seq(self, seq, dataloader, data_type, result_filename, gt=None, gt_hs=None, save_dir=None, show_image=True,
                 frame_rate=30):
        if save_dir:
            mkdir_if_missing(save_dir)
        tracker = JDETracker(self.opt, model=self.model, frame_rate=frame_rate)
        timer = Timer()
        results = []
        len_all = len(dataloader)

        if self.pred_only:
            start_frame = 0
            end_frame = len_all
            frame_id = 0
        else:
            if self.opt.val_half and not self.det_only:
                start_frame = int(len_all / 2)
                end_frame = len_all - 1
                frame_id = int(len_all / 2)
            elif self.opt.val_half:
                start_frame = 0
                end_frame = int(len_all / 2) + 1
                frame_id = 0
            else:
                start_frame = 0
                end_frame = len_all - 1
                frame_id = 0

        images = []
        if gt is not None:
            images, gt = gt
            img_name_2_ids = {}
            for image in images:
                img_name_2_ids[image['file_name']] = image['id']
        current_gt_track = set()
        current_pred_track = set()
        for i, (path, img, img0) in enumerate(dataloader):
            if i < start_frame:
                continue
            if i >= end_frame:
                break
            if gt is not None:
                if path.split('/')[-1] not in img_name_2_ids:
                    continue

            img_id = frame_id + 1
            img_name = path.split('/')[-1]
            if self.pred_only:
                images.append({'id': img_id, 'width': 0, 'height': 0, 'file_name': img_name, 'license': 0,
                               'flickr_url': '', 'coco_url': '', 'date_captured': 0})
            if gt is not None:
                img_name = path.split('/')[-1]
                img_id = img_name_2_ids[img_name]
                frame_id = img_id
            # This code is for debugging
            gt_hs_tlhws = []
            gt_ids = []
            if DEBUG:
                if gt_hs is not None:
                    tmp_gt = gt_hs[img_id]
                    for track_id, tlwh in tmp_gt:
                        gt_ids.append(track_id)
                        gt_hs_tlhws.append(tlwh)

            # run tracking
            timer.tic()
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            online_targets, dets = tracker.update(blob, img0)
            # print([t.track_id for t in online_targets])
            online_tlwhs = []
            online_ids = []
            online_head_areas = []
            # online_scores = []

            if not self.det_only:
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        # online_ids.append(0)
                        online_head_areas.append(self.get_head_area_bbox(tlwh))
                        # online_scores.append(t.score)
            else:
                for det in dets:
                    tlwh = det[0:4]
                    tlwh[2:] -= tlwh[:2]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(1)

            timer.toc()
            # save results
            if self.det_only or self.pred_only:
                results.append((img_id, dets))
            else:
                results.append((frame_id + 1, online_tlwhs, online_ids))
            # results.append((frame_id + 1, online_head_tlwhs, online_ids))
            # results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            if show_image or save_dir is not None:
                image_path = path
                # TODO get bbox of frames
                online_im, bboxes_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                              fps=1. / timer.average_time)
                # online_im = vis.plot_tracking(img0, online_head_tlwhs, online_ids, frame_id=frame_id,
                #                               fps=1. / timer.average_time)
                online_im_gt, bboxes_gt_im = vis.plot_tracking(img0, gt_hs_tlhws, gt_ids, frame_id=frame_id,
                                              fps=1. / timer.average_time)
            label_name = path.split('/')[-1].replace('jpg', 'txt')

            # TODO: save predict labels labels_with_ids
            if show_image:
                cv2.imshow('online_im', online_im)
            save_gt_image = os.path.join(save_dir, 'gt', 'images')
            save_gt_bboxes = os.path.join(save_dir, 'gt', 'bboxes')
            save_gt_set_bboxes = os.path.join(save_dir, 'gt', 'set_bboxes')
            save_gt_label_dir = os.path.join(save_dir, 'gt', 'labels')
            save_predicted_image = os.path.join(save_dir, 'predict', 'images')
            save_predicted_bboxes = os.path.join(save_dir, 'predict', 'bboxes')
            save_predicted_set_bboxes = os.path.join(save_dir, 'predict', 'set_bboxes')
            save_predicted_label_dir = os.path.join(save_dir, 'predict', 'labels')
            create_dir(save_gt_image)
            create_dir(save_gt_label_dir)
            create_dir(save_gt_bboxes)
            create_dir(save_gt_set_bboxes)
            create_dir(save_predicted_image)
            create_dir(save_predicted_label_dir)
            create_dir(save_predicted_bboxes)
            create_dir(save_predicted_set_bboxes)
            if save_dir is not None:
                # # Save gt
                # print(save_gt_image)
                # print(save_predicted_image)
                # raise 1==2
                cv2.imwrite(os.path.join(
                    save_gt_image, '{:05d}.jpg'.format(frame_id)), online_im_gt)
                for track_id, bbox in bboxes_gt_im:
                    create_dir(os.path.join(save_gt_bboxes, img_name))
                    cv2.imwrite(os.path.join(save_gt_bboxes, img_name , '{}.jpg'.format(track_id)), bbox)
                for track_id, bbox in bboxes_gt_im:
                    if track_id not in current_gt_track:
                        current_gt_track.add(track_id)
                        print(current_gt_track)
                        # create_dir(os.path.join(save_gt_set_bboxes, img_name))
                        cv2.imwrite(os.path.join(save_gt_set_bboxes, '{}.jpg'.format(track_id)), bbox)
                save_labels(save_gt_label_dir, gt_hs_tlhws, gt_ids, label_name)
                # Save predict
                cv2.imwrite(os.path.join(
                    save_predicted_image, '{:05d}.jpg'.format(frame_id)), online_im)
                for track_id, bbox in bboxes_im:
                    create_dir(os.path.join(save_predicted_bboxes, img_name))
                    cv2.imwrite(os.path.join(save_predicted_bboxes, img_name , '{}.jpg'.format(track_id)), bbox)
                for track_id, bbox in bboxes_im:
                    if track_id not in current_pred_track:
                        current_pred_track.add(track_id)
                        # create_dir(os.path.join(save_predicted_set_bboxes, img_name))
                        cv2.imwrite(os.path.join(save_predicted_set_bboxes, '{}.jpg'.format(track_id)), bbox)
                save_labels(save_predicted_label_dir, online_tlwhs, online_ids, label_name)

            if gt is None:
                frame_id += SKIP_FRAME + 1
        # save results
        if self.det_only or self.pred_only:
            self.write_coco_preds(result_filename.replace(
                '.txt', '.json'), images, results)
            # pass
        else:
            self.write_results(result_filename, results, data_type)
        # write_results_score(result_filename, results, data_type)
        return frame_id, timer.average_time, timer.calls

    def evaluate(self, exp_name='demo', epoch=0, save_images=False, save_videos=False, show_image=True, logger_main=None):
        logger.setLevel(logging.INFO)
        result_root = os.path.join(self.data_root, '..', 'results', exp_name)
        mkdir_if_missing(result_root)
        data_type = 'mot'

        # run tracking
        accs = []
        n_frame = 0
        timer_avgs, timer_calls = [], []

        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        bar = Bar('{}/{}'.format('VAL', self.val_ds), max=len(self.seqs))
        curr_seqs = []
        mAPs = []
        for i, seq in enumerate(self.seqs):
            output_dir = os.path.join(self.data_root, '..', 'outputs', exp_name,
                                      seq) if save_images or save_videos else None
            # logger.info('start seq: {}'.format(seq))
            dataloader = datasets.LoadImages(
                osp.join(self.data_root, seq, 'img1'), self.opt.img_size)
            result_filename = os.path.join(result_root, '{}.txt'.format(seq))
            frame_rate = self.fps
            if os.path.exists(os.path.join(self.data_root, seq, 'seqinfo.ini')):
                meta_info = open(os.path.join(
                    self.data_root, seq, 'seqinfo.ini')).read()
                frame_rate = int(meta_info[meta_info.find(
                    'frameRate') + 10:meta_info.find('\nseqLength')])

                # Reading gt
            gt = None
            gt_hs = None
            if DEBUG:
                gt_hs_path = os.path.join(
                    self.data_root, seq, 'gt_hs', 'gt.txt')
                gt_hs = handle_hs_file(gt_hs_path)

            if self.det_only and not self.pred_only:
                gt_path = osp.join(self.data_root, seq, 'gt_coco', 'gt.json')
                # print(gt_path)
                # gt_path = osp.join(self.data_root, seq, 'annotations', 'instances_default.json')
                # gt_path = osp.join(self.data_root, seq, 'gt_pseudo', 'gt.json')
                # This for debug

                gt_json = json.load(open(gt_path))
                gt = {}
                for ann in gt_json['annotations']:
                    img_id = ann['image_id']
                    tlwh = ann['bbox']
                    # track_id = ann['attributes']['tracking_id']
                    track_id = 0
                    if img_id in gt:
                        gt[img_id].append((track_id, tlwh))
                    else:
                        gt[img_id] = [(track_id, tlwh)]
                    gt = (gt_json['images'], gt)

            # eval
            nf, ta, tc = self.eval_seq(seq, dataloader, data_type, result_filename, gt=gt, gt_hs=gt_hs,
                                       save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
            n_frame += nf
            timer_avgs.append(ta)
            timer_calls.append(tc)

            # eval
            # logger.info('Evaluate seq: {}'.format(seq))
            if not self.pred_only:
                if self.det_only:
                    # pass
                    cocoGt = COCO(gt_path)
                    preds_path = result_filename.replace('.txt', '.json')
                    with open(preds_path, 'r') as js:
                        res_obj = json.load(js)["annotations"]
                    if len(res_obj) > 0:
                        cocoDt = cocoGt.loadRes(res_obj)
                        evaluation = COCOeval2(cocoGt, cocoDt, "bbox")
                        evaluation.evaluate()
                        evaluation.accumulate()
                        mAPs.append(evaluation.summarize())
                    # print(str(i + 1) + "/" + str(len(self.seqs)) + ":", str(sum(mAPs) / len(mAPs)))
                    logger_main.write('\n')
                    logger_main.write('val: [{0}/{1}]|mAP@.5: {mAP:}'.format(
                        i + 1, len(self.seqs), mAP=sum(mAPs) / (len(mAPs))))
                    logger_main.write('\t')
                    logger_main.write('val_each_set: [{0}/{1}]|mAP@.5: {mAP:}'.format(
                        i + 1, len(self.seqs), mAP=evaluation.summarize()))
                    Bar.suffix = 'val: [{0}/{1}]|mAP@.5: {mAP:}'.format(
                        i + 1, len(self.seqs), mAP=sum(mAPs) / (len(mAPs)))
                else:
                    evaluator = Evaluator(self.data_root, seq, data_type)
                    accs.append(evaluator.eval_file(result_filename))
                    curr_seqs.append(seq)
                    res = Evaluator.get_summary(accs, curr_seqs, metrics)
                    recall = res.recall['OVERALL']
                    precision = res.precision['OVERALL']
                    mota = res.mota['OVERALL']
                    logger_main.write('\n')
                    logger_main.write('val: [{0}/{1}]|R: {recall:} |P: {precision:} |M: {mota:} |Score: {score:}'.format(
                        i + 1, len(self.seqs), recall=recall, precision=precision, mota=mota,
                        score=recall * 0.4 + precision * 0.3 + mota * 0.3))
                    Bar.suffix = 'val: [{0}/{1}]|R: {recall:} |P: {precision:} |M: {mota:} |Score: {score:}'.format(
                        i + 1, len(self.seqs), recall=recall, precision=precision, mota=mota,
                        score=recall * 0.4 + precision * 0.3 + mota * 0.3)

            if save_videos:
                output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
                cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(
                    output_dir, output_video_path)
                os.system(cmd_str)
            bar.next()
        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
            all_time, 1.0 / avg_time))
        bar.finish()

        # get summary
        # metrics = mm.metrics.motchallenge_metrics
        # mh = mm.metrics.create()
        if not self.pred_only:
            if not self.det_only:
                summary = Evaluator.get_summary(accs, self.seqs, metrics)
                recall = summary.recall['OVERALL']
                precision = summary.precision['OVERALL']
                mota = summary.mota['OVERALL']
                score = 0.4 * recall + 0.2 * precision + 0.2 * mota
                strsummary = mm.io.render_summary(
                    summary,
                    formatters=mh.formatters,
                    namemap=mm.io.motchallenge_metric_names
                )
                logger_main.write('\n')
                logger_main.write(strsummary)
                print(strsummary)
                Evaluator.save_summary(summary, os.path.join(
                    result_root, 'summary_{}.xlsx'.format(exp_name)), epoch)
                return mota
            else:
                return sum(mAPs) / (len(mAPs))
