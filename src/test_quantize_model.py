from __future__ import absolute_import, division, print_function

import logging
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer
from tqdm import tqdm

import _init_paths
import lib.datasets.dataset.jde as datasets
from lib.models.decode import mot_decode
from lib.models.model import create_model, load_model
from lib.models.utils import _tranpose_and_gather_feat
from lib.opts import opts
from lib.tracking_utils.log import logger
from lib.tracking_utils.utils import mkdir_if_missing
from lib.utils.post_process import ctdet_post_process
from track import eval_seq

logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(opt, quant=False, xmodel_path=None):
    def post_process(dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(),
            [meta["c"]],
            [meta["s"]],
            meta["out_height"],
            meta["out_width"],
            opt.num_classes,
        )
        for j in range(1, opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(detections):
        max_per_image = opt.K
        results = {}
        for j in range(1, opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0
            ).astype(np.float32)

        scores = np.hstack([results[j][:, 4] for j in range(1, opt.num_classes + 1)])
        if len(scores) > max_per_image:
            kth = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, opt.num_classes + 1):
                keep_inds = results[j][:, 4] >= thresh
                results[j] = results[j][keep_inds]
        return results

    image_folder = "/home/ubuntu/workspace/trungdt21/data/FM_quantization_check"
    dataloader = datasets.LoadImages(image_folder, opt.img_size)

    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to("cuda:0")
    model.eval()

    if quant:
        x = torch.randn([1, 3, 320, 576])
        quantizer = torch_quantizer(
            "test", model, (x), output_dir=xmodel_path, device=device
        )
        model = quantizer.quant_model
        model.eval()

    for i, (path, img, img0) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            z = img.copy()
            print(z.shape)
            width = img0.shape[1]
            height = img0.shape[0]
            inp_height = img.shape[1]
            inp_width = img.shape[2]
            c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
            s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
            meta = {
                "c": c,
                "s": s,
                "out_height": inp_height // opt.down_ratio,
                "out_width": inp_width // opt.down_ratio,
            }

            if torch.cuda.is_available():
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
            else:
                blob = torch.from_numpy(img).unsqueeze(0)
            output = model(blob)
            hm = output[0]
            wh = output[1]
            reg = output[2]
            id_feature = output[3]
            pair = [[12, 15], [23, 19], [7, 40], [21, 21], [12, 12]]
            # pair2 = [
            #     [64, 252],
            #     [64, 251],
            #     [63, 252],
            #     [33, 250],
            #     [63, 251],
            #     [34, 249],
            #     [33, 249],
            # ]
            # print(output[0].shape)
            # print(output[1].shape)
            # print(output[2].shape)
            # print(output[3].shape)
            # print(output[4].shape)
            # id_feature = F.normalize(id_feature, dim=1)
            # print(id_feature.shape)
            for (i, j) in pair:
                print(f"Checking pair [{i},{j}] of channel 0")
                print("hm", hm[0][0][i][j])
                print("wh", wh[0][0][i][j])
                print("reg", reg[0][0][i][j])
                print("id", id_feature[0][0][i][j])
            # id_feature = id_feature.permute(0, 2, 3, 1).contiguous()  # switch id dim
            # for (i, j) in pair2:
            #     print(f"Checking pair [{i},{j}] of channel 0")
            # print("hm", hm[0][0][i][j])
            # print("wh", wh[0][0][i][j])
            # print("wh", wh[0][1][i][j])
            # print("wh", wh[0][2][i][j])
            # print("wh", wh[0][3][i][j])
            # print("reg", reg[0][0][i][j])
            # print("id", id_feature[0][i][j][:])
            # print("hm_pool", hm_pool[0][0][i][j])

            # print(hm[0][0][0][:100])
            # print(wh[0][0][0][:100])
            # print(reg[0][0][0][:100])
            # print(id_feature[0][0][0][:100])
            # print(hm_pool[0][0][0][:100])
            # dets, inds = mot_decode(hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K)
            # print(torch.max(dets[:, 4]))
            # id_feature = _tranpose_and_gather_feat(id_feature, inds)
            # id_feature = id_feature.squeeze(0)
            # id_feature = id_feature.cpu().numpy()

        # dets = post_process(dets, meta)
        # dets = merge_outputs([dets])[1]
        # remain_inds = dets[:, 4] > opt.conf_thres
        # dets = dets[remain_inds]
        # id_feature = id_feature[remain_inds]
        # print(dets.shape)
        # print(dets)
        # for i in range(0, dets.shape[0]):
        #     bbox = dets[i][0:4].astype(int)
        #     # print(bbox)
        #     cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # cv2.imwrite("./test.png", img0)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = opts().init()

    # calibration or evaluation
    # eval(opt)
    eval(
        opt,
        True,
        "/home/ubuntu/workspace/trungdt21/FunMOT/src/quant_out/0811_silver_1plus_final/quantize_result",
    )
