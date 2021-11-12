from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.validator import Validator


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up head of model ...')
    opt = opts().update_res_and_set_heads(opt)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model ...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)

    # Get dataloader
    print('Loading model...')
    start_epoch = 0
    model = load_model(model, opt.load_model)
    model.to('cuda')
    print("Loaded model.")

    # Create validator for each val dataset
    validator_det = Validator(opt, model=model, det_only=True)
    validator_ids = Validator(opt, model=model, det_only=False)

    print('Starting evaluating...')
    ids_mota = validator_ids.evaluate(
        exp_name=opt.exp_id + '_val',
        epoch=start_epoch,
        show_image=True,
        save_images=True,
        save_videos=False,
        logger_main=logger
    )
    det_mAP = validator_det.evaluate(
        exp_name=opt.exp_id + '_val',
        epoch=start_epoch,
        show_image=True,
        save_images=True,
        save_videos=False,
        logger_main=logger
    )

    print("Finished evaluate.")
    # score = det_mAP + ids_mota
    logger.write('\n')
    logger.write('Final result: {} | mAP: {} | MOTA: {}'.format(
        start_epoch, det_mAP, ids_mota))


if __name__ == '__main__':
    args = ['mot',
            '--arch=resfpndcn_18',
            '--conf_thres=0.4',
            '--input_h=256',
            '--input_w=480',
            '--data_cfg=/home/namtd/workspace/projects/smart-city/src/G1-phase3/pseudo-label/FunMOT/src/lib/cfg/LT.json',
            '--load_model=/home/namtd/workspace/projects/smart-city/src/G1-phase3/pseudo-label/FunMOT/models/silver_1/model_best.pth']
    opt = opts().init(args)
    main(opt)
