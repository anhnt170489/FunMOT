from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from trains.validator import Validator
from optimizer.radam import RAdam
import numpy as np

import numpy as np

np.seterr(all='ignore')


warnings.filterwarnings('ignore')


def main(opt):

    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    # For bug
    # torch.cuda.is_available()
    # print(torch.cuda.current_device())
    # print(torch.cuda.device(0))
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))
    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    print(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    img_size = (opt.input_w, opt.input_h)
    dataset = Dataset(opt, dataset_root, trainset_paths,
                      img_size, augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    opt = opts().update_res_and_set_heads(opt)
    print(opt)

    logger = Logger(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    print("Finshed create model + load backbone")

    if opt.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.optimizer == 'RADAM':
        print("Using RADAM")
        optimizer = RAdam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        # num_workers=opt.num_workers,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step, opt.optimizer)

    if opt.val_intervals > 0:
        validator_det = Validator(opt, model=model, det_only=True)
        # validator_ids = Validator(opt, model=model, det_only=False)

        # validator_det.evaluate(
        # exp_name=opt.exp_id + '_val',
        # epoch=0,
        # show_image=False,
        # save_images=False,
        # save_videos=False,
        # logger_main=logger
        # )

        # validator_ids.evaluate(
        #     exp_name=opt.exp_id + '_val',
        #     epoch=0,
        #     show_image=False,
        #     save_images=False,
        #     save_videos=False,
        #     logger_main=logger
        # )
    best_score = -1

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            print('Starting evaluating...')
            det_mAP = validator_det.evaluate(
                exp_name=opt.exp_id + '_val',
                epoch=epoch,
                show_image=False,
                save_images=False,
                save_videos=False,
                logger_main=logger
            )

            # ids_mota = validator_ids.evaluate(
            #     exp_name=opt.exp_id + '_val',
            #     epoch=epoch,
            #     show_image=False,
            #     save_images=False,
            #     save_videos=False,
            #     logger_main=logger
            # )
            score = det_mAP  # + ids_mota
            ids_mota = 0
            logger.write('\n')
            if score > best_score:
                best_score = score
                save_model(os.path.join(
                    opt.save_dir, 'model_best.pth'), epoch, model, optimizer)
                print('New best model at epoch {}'.format(epoch))
                logger.write('epoch: {} | mAP: {} | MOTA: {} | BEST'.format(
                    epoch, det_mAP, ids_mota))
            else:
                logger.write('epoch: {} | mAP: {} | MOTA: {}'.format(
                    epoch, det_mAP, ids_mota))
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step and opt.optimizer == 'ADAM':
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # if epoch % 5 == 0 or epoch >= 25:
        #     save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
        #                epoch, model, optimizer)

    epoch = opt.num_epochs
    if opt.val_intervals > 0 and epoch % opt.val_intervals != 0:
        print('Starting evaluating...')
        det_mAP = validator_det.evaluate(
            exp_name=opt.exp_id + '_val',
            epoch=epoch,
            show_image=False,
            save_images=False,
            save_videos=False,
            logger_main=logger
        )

        # ids_mota = validator_ids.evaluate(
        #     exp_name=opt.exp_id + '_val',
        #     epoch=epoch,
        #     show_image=False,
        #     save_images=False,
        #     save_videos=False,
        #     logger_main=logger
        # )
        score = det_mAP  # + ids_mota
        ids_mota = 0
        if score > best_score:
            save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                       epoch, model, optimizer)
            print('New best model at epoch {}'.format(epoch))
            logger.write('epoch: {} | mAP: {} | MOTA: {} | BEST'.format(
                epoch, det_mAP, ids_mota))
        else:
            logger.write('epoch: {} | mAP: {} | MOTA: {}'.format(
                epoch, det_mAP, ids_mota))
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                   epoch, model, optimizer)
        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                   epoch, model, optimizer)
    logger.close()


if __name__ == '__main__':
    # torch.cuda.set_device(0)
    opt = opts().parse()
    main(opt)
