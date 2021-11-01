from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import argparse
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger.setLevel(logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', '-i', default='/home/namtd/workspace/projects/smart-city/src/G1-phase2/asset/Pseudo-label-FairMOT/videos/beatiful-videos', type=str)
parser.add_argument('--output_path', '-o', default='raw-out/', type=str)
args = parser.parse_args()

def demo(opt, video_name):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(
        result_root, 'frame')
    eval_seq(opt, video_name, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus != [-1])

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'results.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'),
                                                                                  output_video_path)
        os.system(cmd_str)

def batch_inference(args):
    videos = os.listdir(args.input_path)
    for video_name in videos:
        print("===================VinAI===================")
        print("Infering video ", video_name)
        new_args = ['mot',
            '--conf_thres=0.4',
            '--input-video=' + os.path.join(args.input_path, video_name),
            '--output-root=' + os.path.join(args.output_path, video_name),
            '--load_model=models/FM_pretrained/fairmot_dla34.pth']
        opt = opts().init(new_args)
        demo(opt, video_name)
        print("Finished Infereing video ", video_name)
        print("===================GuardPro===================")


if __name__ == '__main__':
    batch_inference(args)