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

logger.setLevel(logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='videos/repeat-videos/total-videos/h264vids/IP_Camera1_27.24_27.24_20211024215751_20211024215821_3013972.mp4', type=str)
parser.add_argument('--output', '-o', default='out/', type=str)
args = parser.parse_args()

def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(
        result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus != [-1])

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'results.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'),
                                                                                  output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input_video = '--input-video=' + args.input
    output_root = '--output-root=' + args.output
    args = ['mot',
            # '--arch=resfpndcn_34',
            # '--val_mot17=True',
            # '--val_mot15=True',
            '--conf_thres=0.4',
            input_video,
            output_root,
            '--load_model=models/FM_pretrained/fairmot_dla34.pth']
    opt = opts().init(args)
    demo(opt)
