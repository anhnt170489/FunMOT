# Gen coco_gt and hs_gt program
import json
import os
import argparse

import cv2


"""
1 - tạo gt_coco/gt.json --> gt theo coco format
2 - tạo gt_hs/gt.txt --> gt theo MOT format
Images = LiveTrack/images/val
Anns = LiveTrack/labels_with_ids/val
seqs = list_dir(Images)
for seq in Anns:
   anns = sorted(list_dir(seq))
   images = []
   annotations = []
   hs = []
   for i, ann in enum(anns)
       // Add image -> json
       image = {}
       id = i+1 // frame_id
       img = cv2.img_read(Images/seq/ann.jpg)
       width = img.shape[1]
       height = img.shape[0]
       filename = ann.jpg
       images.append(image)

       // Add annotation
       // get bbox from ann (relative bbox, ctx, cty, w,h)
       // coco_bbox = tlwh
       // hs_bbox = tlwn
       coco_ann = write_coco_ann(coco_bbox)
       hs_ann = write_hs_ann(hs_bbox)
       annotations.append(coco_ann)
       hs.append(hs_ann)

out_json = {"images": images, "annotations": annotations, .....}
--> dump out_json --> gt_coco/gt.son
--- write_txt (csv) hs --> gt_hs/gt.txt
"""

parser = argparse.ArgumentParser()
parser.add_argument("--images", default="/home/namtd/workspace/projects/smart-city/src/G1-phase3/pseudo-label/FunMOT/data/LT/images/train",
                    type=str, help="path to dataset")
parser.add_argument("--annos", default="/home/namtd/workspace/projects/smart-city/src/G1-phase3/pseudo-label/FunMOT/data/LT/labels_with_ids/train",
                    type=str, help="path to dataset")
args = parser.parse_args()


def create_dir(path):
    if not os.path.exists(path):
        return os.makedirs(path)


def center_2_tlwh(center, width, height):
    real_center_x = center[0] * width
    real_center_y = center[1] * height
    real_width = center[2] * width
    real_height = center[3] * height

    x1 = real_center_x - real_width / 2
    y1 = real_center_y - real_height / 2

    return [x1, y1, real_width, real_height]


def read_anno_file(path, file_name, image_id, width, height):
    dict_annos = []
    file = open(os.path.join(path, file_name), "r")
    for line in file:
        stt, track_id, center_x, center_y, w, h = str(line)[:-1].split(" ")
        center_x = float(center_x)
        center_y = float(center_y)
        w = float(w)
        h = float(h)
        tlwh = center_2_tlwh((center_x, center_y, w, h), width, height)
        # TODO:  Do some convert here
        dict_annos.append(dict(
            image_id=image_id,
            category_id=1,
            segmentation=[],
            iscrowd=0,
            bbox=tlwh,
            area=tlwh[2] * tlwh[3],
            score=1
        ))

    return dict_annos


def read_anno_hs_file(path, file_name, image_id, width, height):
    list_annos = []
    file = open(os.path.join(path, file_name), "r")
    for line in file:
        stt, track_id, center_x, center_y, w, h = str(line)[:-1].split(" ")
        center_x = float(center_x)
        center_y = float(center_y)
        w = float(w)
        h = float(h)
        tlwh = center_2_tlwh((center_x, center_y, w, h), width, height)
        # TODO:  Do some convert here
        list_annos.append([
            str(image_id),
            str(track_id),
            str(tlwh[0]),
            str(tlwh[1]),
            str(tlwh[2]),
            str(tlwh[3]),
            '1',
            '1',
            '1'
        ])
        return list_annos


def write_gt_coco(data_path, annos_path):
    json_coco = {}
    json_coco['categories'] = [
        {
            "id": 1,
            "name": "hs",
            "supercategory": ""
        }
    ]
    images = []
    annotations = []
    annos_files = sorted(os.listdir(annos_path))
    image_files = sorted(os.listdir(data_path))
    # assert len(annos_files) == len(image_files)
    for i, (image_name, anno) in enumerate(zip(image_files, annos_files)):
        image_id = i + 1
        image = cv2.imread(os.path.join(data_path, image_name))
        width = image.shape[1]
        height = image.shape[0]
        images.append(dict(
            id=image_id,
            width=width,
            height=height,
            file_name=image_name,
            license=0,
            flickr_url="",
            coco_url="",
            date_captured=0
        ))
        dict_annos = read_anno_file(annos_path, anno,
                                    image_id, width, height)
        annotations.extend(dict_annos)
    for i, anno in enumerate(annotations):
        id = i + 1
        anno['id'] = id
    json_coco['images'] = images
    json_coco['annotations'] = annotations

    return json_coco


def write_gt_hs(data_path, annos_path):
    annotations = []
    annos_files = sorted(os.listdir(annos_path))
    image_files = sorted(os.listdir(data_path))
    # assert len(annos_files) == len(image_files)
    for i, (image_name, anno) in enumerate(zip(image_files, annos_files)):
        image_id = i + 1
        image = cv2.imread(os.path.join(data_path, image_name))
        width = image.shape[1]
        height = image.shape[0]
        list_annos = read_anno_hs_file(annos_path, anno,
                                       image_id, width, height)
        annotations.extend(list_annos)
    # for i, anno in enumerate(annotations):
    #     id = i + 1
    #     anno['id'] = id
    return annotations


def save_hs_to_file(save_hs_path, hs_file):
    print(save_hs_path)
    print(len(hs_file))
    with open(os.path.join(save_hs_path, 'gt.txt'), 'w') as f:
        for label in hs_file:
            label = " ".join(label) + '\n'
            f.write(label)


def main(args):
    # Gen gt for new dataset.
    image_dir = args.images
    annos_dir = args.annos
    data_set = [set_name for set_name in os.listdir(image_dir)
                if 'train' not in set_name]
    print(data_set)
    # TODO: Read dataset
    for data in data_set:

        data_path = os.path.join(image_dir, data)
        annos_path = os.path.join(annos_dir, data)
        print("image dir", data_path)
        print("annos dir", annos_path)
        # TODO: Write Head and Shoulder groundtruth.
        hs_file = write_gt_hs(data_path, annos_path)
        save_hs_path = os.path.join(image_dir, data, 'gt_hs')
        create_dir(save_hs_path)
        save_hs_to_file(save_hs_path, hs_file)

        # TODO: Write COCO groundtruth.
        # Read annos and write to coco format
        json_coco = write_gt_coco(data_path, annos_path)
        # Save to file
        save_coco_path = os.path.join(image_dir, data, 'gt_coco')
        create_dir(save_coco_path)
        with open(os.path.join(save_coco_path, 'gt.json'), 'w') as f:
            json.dump(json_coco, f)
        print("Done generate set", data)
    return None


if __name__ == "__main__":
    main(args)
