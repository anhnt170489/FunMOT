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

VIDEO = 'merge.mp4'

# Blue color in BGR
FM_COLOR = (255, 0, 0)
BODY_COLOR = (0, 255, 0)
IOU_THRES = 0.7

# Line thickness of 2 px
THICKNESS = 2

fm_anns = np.loadtxt(os.path.join(ROOT, fm_anns))
body_bboxes = np.loadtxt(os.path.join(ROOT, body_bboxes))
# print(fm_anns)
# print(bboxes)

fm_anns_map = {}
body_bboxes_map = {}


def draw_rectangle(img, bbox, color):
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, THICKNESS)


track_count = 0
for ann in fm_anns:
    fid = int(ann[0])
    track_id = int(ann[1])
    if track_id > track_count:
        track_count = track_id
    tlbr = ann[2:6].copy()
    tlbr[2:] = tlbr[:2] + tlbr[2:]
    ann = {'fid': fid, 'track_id': track_id, 'tlbr': tlbr, 'tlwh': ann[2:6]}
    if fid not in fm_anns_map:
        fm_anns_map[fid] = [ann]
    else:
        fm_anns_map[fid].append(ann)

for ann in body_bboxes:
    fid = int(ann[1])
    tlwh = ann[2:6].copy()
    tlwh[2:] = tlwh[2:] - tlwh[:2]
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
for i, frame in enumerate(frames):
    # print('fid', i)
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
                ids.append(fm_ann['track_id'])
                tlwhs.append(fm_ann['tlwh'])
                pseudo_tracks.append({'track_id': fm_ann['track_id'], 'tlbr': fm_ann['tlbr'], 'tlwh': fm_ann['tlwh']})
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
                pseudo_track['tlbr'] = fm_anns[fm_idx]['tlbr']
                pseudo_track['tlwh'] = fm_anns[fm_idx]['tlwh']
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
                        pseudo_track['tlbr'] = body_bbox['tlbr']
                        pseudo_track['tlwh'] = body_bbox['tlwh']
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
                ids.append(fm_ann['track_id'])
                tlwhs.append(fm_ann['tlwh'])
                pseudo_tracks.append({'track_id': fm_ann['track_id'], 'tlbr': fm_ann['tlbr'], 'tlwh': fm_ann['tlwh']})

        merged_img = vis.plot_tracking(img, tlwhs, ids, frame_id=i, fps=20)
        cv2.imwrite(os.path.join(merge_dir, frame), merged_img)

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
                    ids.append(fm_ann['track_id'])
                    tlwhs.append(fm_ann['tlwh'])
                    # Add to Pseudo Tracks
                    pseudo_tracks.append(
                        {'track_id': fm_ann['track_id'], 'tlbr': fm_ann['tlbr'], 'tlwh': fm_ann['tlwh']})
            else:
                # Add FM tracks
                for fm_ann in fm_anns:
                    ids.append(fm_ann['track_id'])
                    tlwhs.append(fm_ann['tlwh'])
                    pseudo_tracks.append(
                        {'track_id': fm_ann['track_id'], 'tlbr': fm_ann['tlbr'], 'tlwh': fm_ann['tlwh']})
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
                    pseudo_track['tlbr'] = body_bbox['tlbr']
                    pseudo_track['tlwh'] = body_bbox['tlwh']
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

        merged_img = vis.plot_tracking(img, tlwhs, ids, frame_id=i, fps=20)
        cv2.imwrite(os.path.join(merge_dir, frame), merged_img)

print("Saving video")
cmd_str = 'ffmpeg -f image2 -i {}/%5d.jpg -b 5000k -c:v mpeg4 {}'.format(merge_dir, os.path.join(ROOT, VIDEO))
os.system(cmd_str)
