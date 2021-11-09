from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
from time import gmtime, strftime

import datasets.dataset.jde as datasets
import torch
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process
from models.model import create_model, load_model
from tracking_utils.timer import Timer
from tracking_utils.utils import xyxy2xywh
from tracking_utils import visualization as vis
from pytorch_nndct.apis import torch_quantizer


def write_results_score(filename, results):
    save_format = "{frame},{x1},{y1},{w},{h},{s}\n"
    with open(filename, "w") as f:
        for frame_id, tlwhs, scores in results:
            for tlwh, score in zip(tlwhs, scores):
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id, x1=x1, y1=y1, w=w, h=h, s=score
                )
                f.write(line)
    print("save results to {}".format(filename))


def post_process(opt, dets, meta):
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


def merge_outputs(opt, detections):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0
        ).astype(np.float32)

    scores = np.hstack([results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > 128:
        kth = len(scores) - 128
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = results[j][:, 4] >= thresh
            results[j] = results[j][keep_inds]
    return results


def eval_seq(
    opt,
    dataloader,
    data_type,
    result_filename,
    save_dir=None,
    show_image=True,
    frame_rate=30,
):
    print(save_dir)
    if opt.gpus[0] >= 0:
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")
    print("Creating model...")
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    # model = torch.nn.DataParallel(model)
    model = model.to(opt.device)

    if opt.quant:
        print("Im here")
        x = torch.randn([1, 3, 320, 576])
        quantizer = torch_quantizer(
            "test", model, (x), output_dir=opt.xmodel, device=opt.device
        )
        model = quantizer.quant_model
    model.eval()

    timer = Timer()
    results = []
    frame_id = 0
    for path, img, img0 in dataloader:
        # if frame_id % 20 == 0:
        #     logger.info(
        #         "Processing frame {} ({:.2f} fps)".format(
        #             frame_id, 1.0 / max(1e-5, timer.average_time)
        #         )
        #     )
        # run detecting
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = blob.shape[2]
        inp_width = blob.shape[3]
        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {
            "c": c,
            "s": s,
            "out_height": inp_height // opt.down_ratio,
            "out_width": inp_width // opt.down_ratio,
        }
        with torch.no_grad():
            k = model(blob)
            # print(k)
            hm, wh, reg, id_feature = k
            hm = hm.sigmoid_()
            # hm = output["hm"].sigmoid_()
            # wh = output["wh"]
            # reg = output["reg"] if opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K)

        dets = post_process(opt, dets, meta)
        dets = merge_outputs(opt, [dets])[1]

        dets = dets[dets[:, 4] > 0.4]
        # dets[:, :4] = xyxy2xywh(dets[:, :4])

        tlbrs = []
        scores = []
        for *tlbr, conf in dets:
            tlbrs.append(tlbr)
            scores.append(conf)
        timer.toc()
        # save results
        results.append((frame_id + 1, tlbrs, scores))
        k = vis.plot_detections(img0, tlbrs, scores)

        frame_id += 1
        cv2.imwrite(os.path.join(save_dir, "{:05d}.jpg".format(frame_id)), k)
    # save results
    write_results_score(result_filename, results)
    # write_results_score_hie(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(
    opt,
    data_root="/data/MOT16/train",
    exp_name="demo",
    save_images=True,
    save_videos=False,
    show_image=True,
):
    # logger.setLevel(logging.INFO)
    # mkdir_if_missing(result_root)
    data_type = "mot"

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    now = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    out_folder = f"./{exp_name}/out_{now}"
    os.makedirs(out_folder)

    output_dir = out_folder if save_images or save_videos else None
    dataloader = datasets.LoadVideo(data_root, opt.img_size)
    result_filename = os.path.join(out_folder, "out.txt")

    nf, ta, tc = eval_seq(
        opt,
        dataloader,
        data_type,
        result_filename,
        save_dir=output_dir,
        show_image=show_image,
    )
    n_frame += nf
    timer_avgs.append(ta)
    timer_calls.append(tc)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    # logger.info(
    #     "Time elapsed: {:.2f} seconds, FPS: {:.2f}".format(all_time, 1.0 / avg_time)
    # )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = opts().init()
    data_root = opt.input_video

    main(
        opt,
        data_root=data_root,
        exp_name="fairmot_mot17",
        show_image=False,
        save_images=True,
        save_videos=False,
    )

