import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
import traceback


def convert_full_to_head(bbox, h, w):
    cy = bbox[:, 3]
    ch = bbox[:, 5]
    cw = bbox[:, 4]
    
    new_cy = cy - ch/2 + (cw/2*w/h)
    new_h = cw*w/h
    
    bbox[:, 3] = new_cy
    bbox[:, 5] = new_h
    
    return bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert Full body annotation to head-shoulder annotation")
    parser.add_argument('--full', type=str, required=True, help="Folder contain body annotation")
    parser.add_argument('--image', type=str, help="Folder contain images")
    parser.add_argument('--image_ext', type=str, default="jpg", help="Image extension")
    parser.add_argument('--height', type=int, help="Image height if constant")
    parser.add_argument('--width', type=int, help="Image width if constant")
    parser.add_argument('--out', type=str, required=True, help="Output annotation for cropped head")
    args = parser.parse_args()

    
    assert args.image is not None or (args.height is not None and args.width is not None)
    
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    try:
        for root, dirs, files in tqdm(os.walk(args.full)):
            for dirname in dirs:
                relpath = os.path.relpath(root, args.full)
                outfolder = os.path.join(args.out, relpath, dirname)
                if not os.path.isdir(outfolder):
                    os.makedirs(outfolder)

            for filename in files:
                gt = np.loadtxt(os.path.join(root, filename)).reshape((-1, 6))
                relpath = os.path.relpath(root, args.full)

                if args.height is None or args.width is None:
                    img_path = os.path.splitext(filename)[0] + "." + args.image_ext
                    img_path = os.path.join(args.image, relpath, img_path)
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]
                else:
                    h = args.height
                    w = args.width
                
                head = convert_full_to_head(gt, h, w)

                np.savetxt(os.path.join(args.out, relpath, filename), head, fmt='%d %d %.6f %.6f %.6f %.6f')
    except:
        # print(root, dirname, filename)
        # print(img_path)
        traceback.print_exc()
