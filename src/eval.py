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
    # det_mAP = validator_det.evaluate(
    #     exp_name=opt.exp_id + '_val',
    #     epoch=start_epoch,
    #     show_image=False,
    #     save_images=True,
    #     save_videos=False,
    #     logger_main=logger
    # )
    ids_mota = validator_ids.evaluate(
        exp_name=opt.exp_id + '_val',
        epoch=start_epoch,
        show_image=False,
        save_images=True,
        save_videos=False,
        logger_main=logger
    )

    print("Finished evaluate.")
    # score = det_mAP + ids_mota
    logger.write('\n')
    # logger.write('Final result: {} | mAP: {} | MOTA: {}'.format(
    #     start_epoch, det_mAP, ids_mota))


if __name__ == '__main__':
    args = ['mot',
            '--arch=resfpndcn_18',
            '--conf_thres=0.4',
            '--img_size=(576,320)',
            # '--img_size=(480,256)',
            # '--img_size=(384,224)',
            '--data_cfg=/home/namtd/workspace/projects/smartcity/src/multiple-tracking/FunMOT/src/lib/cfg/LiveTrack.json',
            '--load_model=/home/namtd/workspace/projects/smartcity/src/multiple-tracking/FunMOT/models/deploy_model/silver_1.2/model_42.pth',
            '--log_model_dir=/home/namtd/workspace/projects/smartcity/src/multiple-tracking/FunMOT/exp/lab/model_1310_384']
    opt = opts().init(args)
    main(opt)
