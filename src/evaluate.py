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

    print('Setting up data...')
    # Dataset = get_dataset(opt.dataset, opt.task)
    # f = open(opt.data_cfg)
    # data_config = json.load(f)
    # trainset_paths = data_config['train']
    # dataset_root = data_config['root']
    # f.close()
    # transforms = T.Compose([T.ToTensor()])
    # img_size = (opt.input_w, opt.input_h)
    # dataset = Dataset(opt, dataset_root, trainset_paths,
    #                   img_size, augment=True, transforms=transforms)
    # opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    opt = opts().update_dataset_info_and_set_heads(opt)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)

    # Get dataloader
    print('Starting training...')
    start_epoch = 0
    # if opt.load_model != '':
    model = load_model(
        model, opt.load_model)
    model.cuda()
    # best_score = -1

    validator_det = Validator(opt, model=model, det_only=True)
    validator_ids = Validator(opt, model=model, det_only=False)

    print('Starting evaluating...')
    det_mAP = validator_det.evaluate(
        exp_name=opt.exp_id + '_val',
        epoch=start_epoch,
        show_image=False,
        save_images=False,
        save_videos=False
    )

    ids_mota = validator_ids.evaluate(
        exp_name=opt.exp_id + '_val',
        epoch=start_epoch,
        show_image=False,
        save_images=False,
        save_videos=False
    )

    score = det_mAP + ids_mota
    logger.write('\n')
    logger.write('epoch: {} | mAP: {} | MOTA: {}'.format(
        start_epoch, det_mAP, ids_mota))


if __name__ == '__main__':
    # torch.cuda.set_device(0)
    args = ['mot',
            '--arch=resfpndcn_18',
            '--conf_thres=0.99',
            '--data_cfg=/home/namtd/workspace/projects/smart-city/src/G1-phase3/pseudo-label/FunMOT/src/lib/cfg/vsm.json',
            # '--reid_dim=64',
            # '--val_half',
            '--load_model=/home/namtd/workspace/projects/smart-city/src/G1-phase3/pseudo-label/FunMOT/models/FM_pretrained/model_best.pth']
    opt = opts().init(args)
    main(opt)
