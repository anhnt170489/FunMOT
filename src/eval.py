from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import os.path as osp

import _init_paths

import cv2
import datasets.dataset.jde as datasets
import motmetrics as mm
import numpy as np
import torch
from opts import opts
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.evaluation import Evaluator
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.utils import mkdir_if_missing
from trains.validator import Validator
from models.model import create_model, load_model

import json
from tracker import matching

if __name__ == '__main__':
    torch.cuda.set_device(0)
    opt = opts().init()
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)

    # validator_det = Validator(opt, model=model, det_only=True)
    validator_ids = Validator(opt, model=model, det_only=opt.det_only, pred_only=opt.pred_only)

    # det_mAP = validator_det.evaluate(
    #     exp_name=opt.exp_id,
    #     epoch=0,
    #     show_image=False,
    #     save_images=True,
    #     save_videos=False
    # )

    ids_mota = validator_ids.evaluate(
        exp_name=opt.exp_id,
        epoch=0,
        show_image=False,
        save_images=True,
        save_videos=False
    )
