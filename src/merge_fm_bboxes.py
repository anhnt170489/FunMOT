import os
import argparse
from cv2 import createTonemap, data
import numpy as np
import cv2
from shutil import copyfile

import _init_paths
from tracker import matching

VIDEO_PATH = '/home/namtd/workspace/projects/smart-city/src/G1-phase2/FunMOT/out/'
VIDEO_NAME = "IP_Camera1_27.24_27.24_20211024215751_20211024215821_3013972"
parser = argparse.ArgumentParser(
    description="Matching pseudo labels and predicted labels."
)
parser.add_argument("--data", "-d", default="out/frame/org", type=str, help="org.")
parser.add_argument("--debug", "-de", default="out/debug/", type=str, help="org.")
parser.add_argument("--fm", "-f", default="out/results.txt", type=str, help="fm.")
parser.add_argument("--pseudo", "-p", default="out/results_pseudo.txt", type=str, help="pseudo.")
parser.add_argument("--save" , "-s", default="/dataset/IP_Camera1_27.24_27.24_20211024215751_20211024215821_3013972", type=str, help='save dataset')
args = parser.parse_args()


# Blue color in BGR
FM_COLOR = (255, 0, 0)
BODY_COLOR = (0, 0, 255)
# Line thickness of 2 px
THICKNESS = 2

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_rectangle(img, bbox, color):
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])), color, THICKNESS)

def map_fairmot_annos(fm_anns):
    fm_anns_map = {}
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

    return fm_anns_map

def map_crd_annos(body_bboxes):
    bboxes_map = {}
    for ann in body_bboxes:
        fid = int(ann[1])
        bboxes = ann[2:6]
        ann = {'fid': fid, 'bboxes': bboxes}
        if fid not in bboxes_map:
            bboxes_map[fid] = [ann]
        else:
            bboxes_map[fid].append(ann)
    return bboxes_map

def reformat_output(bboxes):
    new_bboxes = []
    for track_id, bbox in bboxes:
        x1, y1, x2, y2 = bbox
        new_box = ('0', str(track_id),str(x1),str(y1),str(x2),str(y2))
        new_bboxes.append(new_box)
    return new_bboxes

def save_labels(center_bboxes, image_name, save_folder):
    label_file = image_name.replace('.jpg', '.txt')
    # label_file = image_name.replace('.jpeg', '.txt')
    print(label_file)
    save_file = os.path.join(save_folder, label_file)
    new_bboxes = reformat_output(center_bboxes)
    with open(save_file, "w") as fobj:
        for box in new_bboxes:
            box = ' '.join(box)
            fobj.write(box + "\n")

def tlbr_2_center(tlbr, width_image, height_image):
    x1, y1, x2, y2 = tlbr
    width = x2 - x1
    height = y2 - y1
    center_x = x2 - width / 2
    center_y = y2 - height / 2
    return (center_x / width_image, center_y / height_image,
            width / width_image, height / height_image)

def save_to_dataset(image_dir, save_dir, images, total_bboxes):
    print("Number of images ", len(images))
    print("Number of bboxes ", len(total_bboxes))
    save_image_dir = os.path.join(save_dir, 'images')
    save_label_dir = os.path.join(save_dir, 'labels')
    create_dir(save_image_dir)
    create_dir(save_label_dir)
    print(save_image_dir)
    print(save_label_dir)
    for image, bboxes in zip(images, total_bboxes):
        image_name = image.split('/')[-1]
        print(image_name)
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        width_image = img.shape[1]
        height_image = img.shape[0]
        # TODO Copy image to save_dir (images)
        # print(image)
        copyfile(image, os.path.join(save_image_dir, image_name))
        # TODO Save to labels folder
        print(len(bboxes))
        print(bboxes)
        center_bboxes = []
        for track_id, bbox in bboxes:
            print(bbox)
            bbox[3] = bbox[1] + (bbox[2] - bbox[0])
            center_bboxes.append((track_id, tlbr_2_center(bbox, width_image, height_image)))
        save_labels(center_bboxes, image_name, save_label_dir)

def main(args):
    # Handle with debugor
    debug_dir = args.debug
    create_dir(debug_dir)
    # Save dataset
    save_dir = args.save
    create_dir(save_dir)

    # Read and load fairmot and crd bboxes in map
    fm_anns = np.loadtxt(args.fm)
    body_bboxes = np.loadtxt(args.pseudo)

    fm_anns_map = map_fairmot_annos(fm_anns)
    bboxes_map = map_crd_annos(body_bboxes)
    print("Number of faimot annos ", len(fm_anns_map.keys()))
    print("Number of crd annos ", len(bboxes_map.keys()))

    # Read images
    frames_dir = args.data
    frames = sorted(os.listdir(frames_dir))
    saved_frames = []
    saved_bboxes = []
    for i, frame in enumerate(frames):
        # Check type of images
        if not frame.endswith('.jpg') and not frame.endswith('.png'):
            continue
        # Check if frame id dont have bbvox for both model
        fid = i + 1
        if fid not in fm_anns_map or fid not in bboxes_map:
            continue
        # Read image - for debug
        saved_frames.append(os.path.join(frames_dir, frame))
        img = cv2.imread(os.path.join(frames_dir, frame))
        fm_anns = fm_anns_map[fid]
        bboxes = bboxes_map[fid]
        fm_bboxes = [np.array(fm_ann['bboxes']) for fm_ann in fm_anns]
        body_bboxes = [np.array(ann['bboxes']) for ann in bboxes]
        track_ids = [np.array(fm_ann['track_id']) for fm_ann in fm_anns]
        _ious = matching.ious(fm_bboxes, body_bboxes)
        cost_matrix = 1 - _ious
        matches, u_fm, u_bboxes = matching.linear_assignment(
            cost_matrix, thresh=1)
        matched_bboxes = [(track_ids[fm_id], body_bboxes[body_id]) for fm_id, body_id in matches]
        saved_bboxes.append(matched_bboxes)
        for fm_id, body_id in matches:
            fm_bbox = fm_bboxes[fm_id]
            draw_rectangle(img, fm_bbox, FM_COLOR)
            body_bbox = body_bboxes[body_id]
            draw_rectangle(img, body_bbox, BODY_COLOR)
        cv2.imwrite(os.path.join(debug_dir, frame), img)
    save_to_dataset(frames_dir, save_dir, saved_frames, saved_bboxes)
if __name__ == '__main__':
    main(args)