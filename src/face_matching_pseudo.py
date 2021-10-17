import numpy as np
from tracker import matching

# tracks containing head & shoulder bboxes
tracks = []  # output of tracking
# face detection wrt hs bboxes
face_dets = [[]]  # output face bboxes of face detections

# ambiguous tracks, face_dets
amb_tracks, ambs_face_dets = [], []  # empty list


def get_head_area_bbox(bbox_tlwh):
    tl_x, tl_y, w, h = bbox_tlwh[0], bbox_tlwh[1], bbox_tlwh[2], bbox_tlwh[3]
    c_x, c_y = tl_x + w / 2, tl_y + h / 2
    head_area_bbox_tlbr = [c_x - h / 4, tl_y, c_x + h / 4, c_y]
    head_area_bbox_tlwh = [c_x - h / 4, tl_y, h / 2, h / 2]
    return head_area_bbox_tlbr, head_area_bbox_tlwh


for i, face_det in enumerate(face_dets):
    if len(face_det) == 1:
        tracks[i].face_bbox = face_det[0]
    else:
        amb_tracks.append((i, tracks[i]))
        for j, sub_face_det in enumerate(face_det):
            ambs_face_dets.append((i, j, sub_face_det))

# generate head area box from tlwh bbox
track_head_bboxes = [np.array(get_head_area_bbox(amb_track[1].bbox_tlwh)) for amb_track in amb_tracks]
face_dets = [np.array(ambs_face_det[2]) for ambs_face_det in ambs_face_dets]

# calculate iou as cost matrix for hungarian matching
_ious = matching.ious(track_head_bboxes, face_dets)
cost_matrix = 1 - _ious

# do hungarian matching
matches, _, _ = matching.linear_assignment(cost_matrix, thresh=1)

# set track face_bbox with matched face det
for match in matches:
    amb_track_idx, amb_face_det_idx = match
    track_idx = amb_tracks[track_idx][0]
    face_det_idx = ambs_face_dets[amb_face_det_idx][0]
    sub_face_det_idx = ambs_face_dets[amb_face_det_idx][1]
    tracks[track_idx].face_bbox = face_dets[face_det_idx][sub_face_det_idx]
