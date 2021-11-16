import os
import numpy as np
import cv2

import _init_paths
from tracker import matching
from tracking_utils import visualization as vis

# ROOT = '../out/videos/AN12_0_5_pretrained'
ROOT = '../data/sample/video_frinday1'
frames = 'org'
fm_anns = 'results.txt'
body_bboxes = 'results_pseudo.txt'
# err_dir = os.path.join(ROOT, 'err')
# if not os.path.exists(err_dir):
#     os.mkdir(err_dir)

merge_dir = os.path.join(ROOT, 'merge')
if not os.path.exists(merge_dir):
    os.mkdir(merge_dir)

merge_label_dir = os.path.join(ROOT, 'merge_label')
if not os.path.exists(merge_label_dir):
    os.mkdir(merge_label_dir)

gt_img_dir = os.path.join(ROOT, 'image')
if not os.path.exists(gt_img_dir):
    os.mkdir(gt_img_dir)

VIDEO = 'merge.mp4'

# Blue color in BGR
FM_COLOR = (255, 0, 0)
BODY_COLOR = (0, 255, 0)
IOU_THRES = 0.7

# Line thickness of 2 px
THICKNESS = 2

fm_anns = np.loadtxt(os.path.join(ROOT, fm_anns))
# fm_anns = []
body_bboxes = np.loadtxt(os.path.join(ROOT, body_bboxes))
# body_bboxes = []
# print(fm_anns)
# print(bboxes)

fm_anns_map = {}
body_bboxes_map = {}


def draw_rectangle(img, bbox, color):
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, THICKNESS)


def add_special_track(ids, tlwhs, pseudo_tracks, id):
    if id not in ids:
        for track in pseudo_tracks:
            if track['track_id'] == id:
                ids.append(id)
                tlwhs.append(track['tlwh'])


def tlwh_2_center(tlwh, width_image, height_image):
    x1, y1, w, h = tlwh
    x2, y2 = x1 + w, y1 + h
    x2, y2 = min(width_image, x2), min(height_image, y2)
    w, h = x2 - x1, y2 - y1
    center_x = x1 + w / 2
    center_y = y1 + h / 2
    return [center_x / width_image, center_y / height_image,
            w / width_image, h / height_image]


def reformat_output(bbox):
    track_id, c_x1, c_y1, w, h = bbox
    new_box = ('0', str(track_id), str(c_x1), str(c_y1), str(w), str(h))
    return new_box


def init_track(track_id, tlwh):
    t, l, w, h = tlwh
    b, r = t + w, l + h
    return {'track_id': track_id, 'tlbr': (t, l, b, r), 'tlwh': tlwh}


def update_pseudo_track(frame_id, pseudo_track, ann):
    if frame_id == 124 and pseudo_track['track_id'] == 14:
        print()
    if pseudo_track['track_id'] == 14 and ann['tlwh'][1] > 203 \
            or pseudo_track['track_id'] == 9 and ann['tlwh'][1] > 178:
        pass
    else:
        pseudo_track['tlbr'] = ann['tlbr']
        pseudo_track['tlwh'] = ann['tlwh']


def update_results(results, pseudo_tracks, frame, img, ids, tlwhs):
    frame_id = int(frame.replace('.jpg', ''))
    # add_special_track(ids, tlwhs, pseudo_tracks, 7)
    # add_special_track(ids, tlwhs, pseudo_tracks, 11)
    # add_special_track(ids, tlwhs, pseudo_tracks, 4)
    # add_special_track(ids, tlwhs, pseudo_tracks, 5)
    # add_special_track(ids, tlwhs, pseudo_tracks, 10)
    # add_special_track(ids, tlwhs, pseudo_tracks, 9)
    # add_special_track(ids, tlwhs, pseudo_tracks, 8)
    # if 118 < frame_id < 134:
    #     add_special_track(ids, tlwhs, pseudo_tracks, 13)
    # elif 134 < frame_id < 384:
    #     add_special_track(ids, tlwhs, pseudo_tracks, 14)
    # if frame_id in [392, 393]:
    #     ids.append(15)
    #     tlwhs.append((1057.76770155, 64.71291435, 111.63906807, 111.63906807))

    results.append((frame, img, ids, tlwhs))


track_count = 0
for ann in fm_anns:
    fid = int(ann[0])
    track_id = int(ann[1])
    tlbr = ann[2:6].copy()
    tlbr[2:] = tlbr[:2] + tlbr[2:]
    tlwh = ann[2:6].copy()
    tlwh[3] = tlwh[2]
    ann = {'fid': fid, 'track_id': track_id, 'tlbr': tlbr, 'tlwh': tlwh}
    if fid not in fm_anns_map:
        fm_anns_map[fid] = [ann]
    else:
        fm_anns_map[fid].append(ann)

for ann in body_bboxes:
    fid = int(ann[1])
    tlwh = ann[2:6].copy()
    tlwh[2:] = tlwh[2:] - tlwh[:2]
    tlwh[3] = tlwh[2]
    ann = {'fid': fid, 'tlbr': ann[2:6], 'tlwh': tlwh}
    if fid not in body_bboxes_map:
        body_bboxes_map[fid] = [ann]
    else:
        body_bboxes_map[fid].append(ann)

# print(fm_anns_map)
# print(bboxes_map)
frames_idr = os.path.join(ROOT, frames)
frames = sorted(os.listdir(frames_idr))

pseudo_tracks = []
# pseudo_tracks.append(init_track(1, ([220.699646, 336.35784912, 152.12185669, 152.12185669])))
# pseudo_tracks.append(init_track(2, ([137.35307312, 378.98498535, 98.69044495, 98.69044495])))
# track_count = 2
results = []
for i, frame in enumerate(frames):
    # print('fid', i)
    # print(len(pseudo_tracks))
    # if i == 119:
    #     pseudo_tracks.append(init_track(13, (196.21838379, 203.77883911, 165.66113281, 165.66113281)))
    #     track_count = 13
    # if i > 420:
    #     break
    if not frame.endswith('.jpg') and not frame.endswith('.png'):
        continue
    tlwhs = []
    ids = []
    fid = i + 1
    img = cv2.imread(os.path.join(frames_idr, frame))
    if fid in fm_anns_map and fid in body_bboxes_map:
        fm_anns = fm_anns_map[fid]
        body_bboxes = body_bboxes_map[fid]
        # Fuse FM bboxes & body bboxes
        fm_tlbrs = [np.array(fm_ann['tlbr']) for fm_ann in fm_anns]
        body_tlbrs = [np.array(bbox['tlbr']) for bbox in body_bboxes]
        cost_matrix = matching.iou_distance(fm_tlbrs, body_tlbrs)
        matches, _, u_body = matching.linear_assignment(cost_matrix, thresh=IOU_THRES)
        for fm_idx, body_idx in matches:
            # Replace FM bbox by body bbox, keep FM track_id
            fm_tlbrs[fm_idx] = body_tlbrs[body_idx]
            fm_anns[fm_idx]['tlbr'] = body_bboxes[body_idx]['tlbr']
            fm_anns[fm_idx]['tlwh'] = body_bboxes[body_idx]['tlwh']

        if len(pseudo_tracks) == 0:
            for fm_ann in fm_anns:
                # Add FM track
                track_count += 1
                ids.append(track_count)
                tlwhs.append(fm_ann['tlwh'])
                pseudo_tracks.append({'track_id': track_count, 'tlbr': fm_ann['tlbr'], 'tlwh': fm_ann['tlwh']})
                # tlbr = fm_tlbrs[idx]
                # draw_rectangle(img, tlbr, FM_COLOR)
            for idx in u_body:
                body_bbox = body_bboxes[idx]
                # Create new pseudo Track
                track_count += 1
                pseudo_tracks.append({'track_id': track_count, 'tlbr': body_bbox['tlbr'], 'tlwh': body_bbox['tlwh']})
                # Add pseudo track
                ids.append(track_count)
                tlwhs.append(body_bbox['tlwh'])
        else:
            # Fuse FM tracks & pseudo tracks
            pseudo_track_tlbrs = [np.array(pseudo_track['tlbr']) for pseudo_track in pseudo_tracks]
            cost_matrix = matching.iou_distance(pseudo_track_tlbrs, fm_tlbrs)
            matches, u_pseudo, u_fm = matching.linear_assignment(cost_matrix, thresh=IOU_THRES)
            for pseudo_idx, fm_idx in matches:
                # Replace pseudo bbox by FM bbox
                pseudo_track = pseudo_tracks[pseudo_idx]
                update_pseudo_track(i, pseudo_track, fm_anns[fm_idx])
                # Add to output
                ids.append(pseudo_track['track_id'])
                tlwhs.append(pseudo_track['tlwh'])

            if len(u_body) > 0:
                if len(u_pseudo) > 0:
                    # Fuse unmatched body bboxes & pseudo tracks
                    u_pseudo_track_tlbrs = [np.array(pseudo_tracks[pseudo_idx]['tlbr']) for pseudo_idx in u_pseudo]
                    u_body_tlbrs = [np.array(body_bboxes[body_idx]['tlbr']) for body_idx in u_body]
                    cost_matrix = matching.iou_distance(u_pseudo_track_tlbrs, u_body_tlbrs)
                    matches, _, new_body_bboxes = matching.linear_assignment(cost_matrix, thresh=IOU_THRES)
                    for pseudo_idx, body_idx in matches:
                        # Update pseudo track bbox by body bbox
                        pseudo_track = pseudo_tracks[u_pseudo[pseudo_idx]]
                        body_bbox = body_bboxes[u_body[body_idx]]
                        update_pseudo_track(i, pseudo_track, body_bbox)
                        # Add to output
                        ids.append(pseudo_track['track_id'])
                        tlwhs.append(pseudo_track['tlwh'])
                    new_body_bboxes = [u_body[idx] for idx in new_body_bboxes]
                    for idx in new_body_bboxes:
                        body_bbox = body_bboxes[idx]
                        # Create new pseudo Track
                        track_count += 1
                        pseudo_tracks.append(
                            {'track_id': track_count, 'tlbr': body_bbox['tlbr'], 'tlwh': body_bbox['tlwh']})
                        # Add pseudo track
                        ids.append(track_count)
                        tlwhs.append(body_bbox['tlwh'])

            for fm_idx in u_fm:
                fm_ann = fm_anns[fm_idx]
                # Add FM track
                track_count += 1
                ids.append(track_count)
                tlwhs.append(fm_ann['tlwh'])
                pseudo_tracks.append({'track_id': track_count, 'tlbr': fm_ann['tlbr'], 'tlwh': fm_ann['tlwh']})

        # merged_img = vis.plot_tracking(img, tlwhs, ids, frame_id=i, fps=20)
        # cv2.imwrite(os.path.join(merge_dir, frame), merged_img)
        update_results(results, pseudo_tracks, frame, img, ids, tlwhs)

    else:
        if fid in fm_anns_map:
            fm_anns = fm_anns_map[fid]
            # fm_bboxes = [fm_ann['bboxes'] for fm_ann in fm_anns]
            # for bbox in fm_bboxes:
            #     draw_rectangle(img, bbox, FM_COLOR)
            if len(pseudo_tracks) > 0:
                # Fuse FM tracks & pseudo tracks
                fm_tlbrs = [np.array(fm_ann['tlbr']) for fm_ann in fm_anns]
                pseudo_track_tlbrs = [np.array(pseudo_track['tlbr']) for pseudo_track in pseudo_tracks]
                cost_matrix = matching.iou_distance(pseudo_track_tlbrs, fm_tlbrs)
                matches, u_pseudo, u_fm = matching.linear_assignment(cost_matrix, thresh=IOU_THRES)
                for pseudo_idx, fm_idx in matches:
                    # Update Pseudo tracks
                    pseudo_track = pseudo_tracks[pseudo_idx]
                    pseudo_track['tlbr'] = fm_anns[fm_idx]['tlbr']
                    pseudo_track['tlwh'] = fm_anns[fm_idx]['tlwh']
                    # Add to output
                    ids.append(pseudo_track['track_id'])
                    tlwhs.append(pseudo_track['tlwh'])
                # Add unmatched FM tracks
                for u_fm_idx in u_fm:
                    u_fm_ann = fm_anns[u_fm_idx]
                    # Add to output
                    track_count += 1
                    ids.append(track_count)
                    tlwhs.append(u_fm_ann['tlwh'])
                    # Add to Pseudo Tracks
                    pseudo_tracks.append(
                        {'track_id': track_count, 'tlbr': u_fm_ann['tlbr'], 'tlwh': u_fm_ann['tlwh']})
            else:
                # Add FM tracks
                for fm_ann in fm_anns:
                    track_count += 1
                    ids.append(track_count)
                    tlwhs.append(fm_ann['tlwh'])
                    pseudo_tracks.append(
                        {'track_id': track_count, 'tlbr': fm_ann['tlbr'], 'tlwh': fm_ann['tlwh']})
        if fid in body_bboxes_map:
            body_bboxes = body_bboxes_map[fid]
            # body_bboxes = [ann['bboxes'] for ann in bboxes]
            # for bbox in body_bboxes:
            #     draw_rectangle(img, bbox, BODY_COLOR)
            # Create new pseudo Track
            new_body_bboxes = []
            if len(pseudo_tracks) > 0:
                # Fuse unmatched body bboxes & pseudo tracks
                pseudo_track_tlbrs = [np.array(pseudo_track['tlbr']) for pseudo_track in pseudo_tracks]
                body_tlbrs = [np.array(body_bbox['tlbr']) for body_bbox in body_bboxes]
                cost_matrix = matching.iou_distance(pseudo_track_tlbrs, body_tlbrs)
                matches, _, new_body_bboxes = matching.linear_assignment(cost_matrix, thresh=IOU_THRES)

                # print('pseudo_track_tlbrs', pseudo_track_tlbrs)
                # print('body_tlbrs', body_tlbrs)
                # print('matches', matches)
                for pseudo_idx, body_idx in matches:
                    # Update pseudo track bbox by body bbox
                    pseudo_track = pseudo_tracks[pseudo_idx]
                    body_bbox = body_bboxes[body_idx]
                    update_pseudo_track(i, pseudo_track, body_bbox)
                    # Add to output
                    ids.append(pseudo_track['track_id'])
                    tlwhs.append(pseudo_track['tlwh'])
            else:
                new_body_bboxes = [i for i in range(len(body_bboxes))]

            for idx in new_body_bboxes:
                body_bbox = body_bboxes[idx]
                # Create new pseudo Track
                track_count += 1
                pseudo_tracks.append({'track_id': track_count, 'tlbr': body_bbox['tlbr'], 'tlwh': body_bbox['tlwh']})
                # Add pseudo track
                ids.append(track_count)
                tlwhs.append(body_bbox['tlwh'])

        # cv2.imwrite(os.path.join(err_dir, frame), img)

        # merged_img = vis.plot_tracking(img, tlwhs, ids, frame_id=i, fps=20)
        # cv2.imwrite(os.path.join(merge_dir, frame), merged_img)
        update_results(results, pseudo_tracks, frame, img, ids, tlwhs)

mapping_ids = {1: 1, 2: 2, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 17: 15, 16: 16}
must_have_ids = [1, 2]

frame_count = 0
ds_cfg = os.path.join(ROOT, "images.train")
ds_cfg = open(ds_cfg, 'w')
for frame, img, ids, tlwhs in results:
    frame_id = int(frame.replace('.jpg', ''))

    merged_img = vis.plot_tracking(img, tlwhs, ids, frame_id=frame_id, fps=20)
    frame_path = os.path.join(merge_dir, format(frame_count, '05d') + ".jpg")
    cv2.imwrite(frame_path, merged_img)
    frame_count += 1

#     norm_ids = []
#     filtered_tlwhs = []
#     for i, id in enumerate(ids):
#         if frame_id < 69:
#             if id in mapping_ids:
#                 norm_ids.append(mapping_ids[id])
#                 filtered_tlwhs.append(tlwhs[i])
#             elif id == 3:
#                 norm_ids.append(3)
#                 filtered_tlwhs.append(tlwhs[i])
#         else:
#             if id == 14:
#                 norm_ids.append(8)
#                 filtered_tlwhs.append(tlwhs[i])
#             elif id == 8:
#                 norm_ids.append(13)
#                 filtered_tlwhs.append(tlwhs[i])
#             elif id == 15:
#                 norm_ids.append(6)
#                 filtered_tlwhs.append(tlwhs[i])
#             elif id == 6:
#                 norm_ids.append(17)
#                 filtered_tlwhs.append(tlwhs[i])
#             elif id in mapping_ids:
#                 norm_ids.append(mapping_ids[id])
#                 filtered_tlwhs.append(tlwhs[i])
#
#     is_valid = True
#     # for id in must_have_ids:
#     #     if id not in norm_ids:
#     #         is_valid = False
#     #         break
#
#     if is_valid:
#
#         merged_img = vis.plot_tracking(img, filtered_tlwhs, norm_ids, frame_id=frame_id, fps=20)
#         frame_path = os.path.join(merge_dir, format(frame_count, '05d') + ".jpg")
#         cv2.imwrite(frame_path, merged_img)
#         frame_count += 1
#         if len(norm_ids) > 0:
#             # save org img
#             cv2.imwrite(os.path.join(gt_img_dir, frame), img)
#             ds_cfg.write(os.path.join('LiveTrack/train/set-22', frame) + "\n")
#
#             # write labels
#             with open(os.path.join(merge_label_dir, frame.replace('.jpg', '.txt')), "w") as fobj:
#                 for i, id in enumerate(norm_ids):
#                     lbls = [id]
#                     lbls.extend(tlwh_2_center(tuple(filtered_tlwhs[i]), img.shape[1], img.shape[0]))
#                     fobj.write(' '.join(reformat_output(tuple(lbls))) + "\n")
#             fobj.close()
# ds_cfg.close()

print("Saving video")
cmd_str = 'ffmpeg -f image2 -i {}/%5d.jpg -b 5000k -c:v mpeg4 {}'.format(merge_dir, os.path.join(ROOT, VIDEO))
os.system(cmd_str)
