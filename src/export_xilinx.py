from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pytorch_nndct.apis import torch_quantizer

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
import torch
from track import eval_seq
from lib.models.model import create_model, load_model
from tqdm import tqdm
import random

logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(12345)


def evaluate(model, val_loader):
    print("evaluating...")
    model.eval()

    with torch.no_grad():
        for i, (path, img) in tqdm(enumerate(val_loader)):
            # if i > 5:
            #     break
            # if torch.cuda.is_available():
            #     blob = torch.from_numpy(img).cuda().unsqueeze(0)
            # else:
            #     blob = torch.from_numpy(img).unsqueeze(0)
            blob = img.cuda()
            output = model(blob)

    return 0.0


def load_data(subset_len, batch_size, lines, img_size):
    dataset = datasets.LoadImagesCalib(lines, img_size)
    if subset_len:
        assert subset_len <= len(dataset)
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    return data_loader


def quantization(opt, title="optimize", model_name="", file_path=""):

    # quant_mode = args.quant_mode
    quant_mode = opt.quant_mode
    deploy = opt.deploy
    batch_size = opt.batch_size
    subset_len = opt.subset_len
    finetune = opt.fast_finetune

    if quant_mode != "test" and deploy:
        deploy = False
        print(
            r"Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!"
        )

    if deploy and (batch_size != 1 or subset_len != 1):
        print(
            r"Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!"
        )
        batch_size = 1
        subset_len = 1

    logger.info("Starting tracking...")

    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(device)

    input = torch.randn([batch_size, 3, 320, 576])
    if quant_mode == "float":
        quant_model = model
    else:
        ## new api
        ####################################################################################
        quantizer = torch_quantizer(quant_mode, model, (input), device=device)
        quant_model = quantizer.quant_model
        #####################################################################################

    # print(quant_model.state_dict())

    print("=" * 80)
    print(f"Quant mode is {quant_mode}, deploy: {deploy}")

    ext = opt.calib_datapath.split(".")[1]
    if ext in ["txt", "train"]:
        with open(opt.calib_datapath, "r") as f:
            lines = f.readlines()
        data_base_path = "/home/ubuntu/workspace/trungdt21/data_calib_fairmot"
        print(f"Calib using {opt.list_images}; img_size: {opt.img_size}")
        lines = list(map(lambda x: os.path.join(data_base_path, x.rstrip()), lines))
        ######################### Using load_data
        val_loader = load_data(
            subset_len=subset_len,
            batch_size=batch_size,
            lines=lines,
            img_size=opt.img_size,
        )

        ######################### Using LoadImagesCalib

        # val_dataset = datasets.LoadImagesCalib(lines, opt.img_size)
        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset,
        #     batch_size=batch_size,
        #     num_workers=1,
        #     pin_memory=True,
        #     drop_last=True,
        # )
    elif ext == "mp4":
        ############################# Using Load Video
        val_loader = datasets.LoadVideo(opt.calib_datapath, (576, 320))

    if finetune == True:
        if quant_mode == "calib":
            quantizer.fast_finetune(evaluate, (quant_model, val_loader))
        elif quant_mode == "test":
            quantizer.load_ft_param()

    evaluate(quant_model, val_loader)

    # handle quantization result
    if quant_mode == "calib":
        quantizer.export_quant_config()
        # quantizer.processor.quantizer.export_param()

    if deploy:
        print("DEPLOYING")
        # quant_model.eval()
        quantizer.export_xmodel(deploy_check=True)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = opts().init()

    title = "clgt"

    # calibration or evaluation
    quantization(title=title, model_name=title)

    print("-------- End of {} test ".format(title))
