from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq

logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
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
    args = ['mot',
            # '--arch=resfpndcn_34',
            # '--arch=resfpndcn_18',
            '--conf_thres=0.4',
            '--input-video=../videos/AN12_0_5.mp4',
            '--output-root=../out/videos/AN12_0_5_pretrained',
            # '--val_mot17=True',y
            # '--val_mot15=True',
            # '--load_model=../models/mix_half_live_ft_resfpndcn_18_576_320/model_best_2nd.pth']
            '--load_model=../models/FM_pretrained/fairmot_dla34.pth']
    # '--load_model=../models/crowdhuman_head_resnet34fpn/model_last.pth']
    opt = opts().init(args)
    demo(opt)
