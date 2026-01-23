import argparse
import time
from importlib import import_module
from thop import profile, clever_format

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import get_val_loader
# from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.TIFDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import group_weight, init_weight
from utils.lr_policy import WarmUpPolyLR
from utils.metric import compute_score, hist_info
from utils.pyt_utils import all_reduce_tensor, ensure_dir, link_file, load_model, parse_devices
from utils.val_mm import evaluate, evaluate_msf
# from utils.visualize import print_iou, show_img

# from eval import evaluate_mid

# SEED=1
# # np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic=False
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", help="used gpu number")
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
# parser.add_argument('-d', '--devices', default='0,1', type=str)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else 'cpu', 
                    type=str, help="device to use for training / testing")
parser.add_argument("--save_path", default=None)
logger = get_logger()

# os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = getattr(import_module(args.config), "C")
    config.pad = False  # Do not pad when inference
    if "x_modal" not in config:
        config["x_modal"] = "d"

    if config.dataset_name != "SUNRGBD":
        val_batch_size = int(config.batch_size)
    elif not args.pad_SUNRGBD:
        val_batch_size = int(args.gpus)
    else:
        val_batch_size = 8 * int(args.gpus)

    val_loader, val_sampler = get_val_loader(
                                            engine, 
                                            RGBXDataset, 
                                            config, 
                                            val_batch_size=val_batch_size
                                        )
    # print(len(val_loader))
    logger.info(f"val dataset len:{len(val_loader) * int(args.gpus)}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + "/tb"
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    logger.info("args parsed:")
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
        logger.info("using syncbn")
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")

    model = segmodel(cfg=config, 
                     criterion=criterion,
                     norm_layer=BatchNorm2d,
                     syncbn=engine.distributed,
                    )
    weight = torch.load(args.continue_fpath, map_location=torch.device("cpu"))
    if "model" in weight:
        weight = weight["model"]
    elif "state_dict" in weight:
        weight = weight["state_dict"]

    logger.info(f"load model from {args.continue_fpath}")
    print(model.load_state_dict(weight, strict=False))


    ################################# FLOPs and Params ###################################################

    # input=(torch.randn(1, 3, 512, 512).to("cuda"), torch.randn(1, 1, 512, 512).to("cuda"))

    # model.to("cuda").eval()
    # flops, params = profile(model, inputs=input)

    # flops, params = clever_format([flops, params])
    # print('# Model FLOPs: {}'.format(flops))
    # print('# Model Params: {}'.format(params))


    if engine.distributed:
        logger.info(".............distributed training.............")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=True,
            )
    else:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(args.device)

    if args.compile:
        model = torch.compile(model, backend="inductor", mode=args.compile_mode)

    engine.register_state(dataloader=val_loader, model=model)

    logger.info("begin testing:")

    torch.cuda.empty_cache()
    if engine.distributed:
        print("multi GPU test")
        with torch.no_grad():
            model.eval()
            if args.mst:

                # 测试时增强
                all_metrics = evaluate_msf(
                    model,
                    val_loader,
                    config,
                    args.device,
                    [0.5, 0.75, 1.0, 1.25, 1.5],
                    True,
                    engine,
                )

            else:
                all_metrics = evaluate(
                    model,
                    val_loader,
                    config,
                    args.device,
                    engine,
                    save_dir=None,
                )

            if engine.local_rank == 0:
                metric = all_metrics[0]
                
                score, class_iou = metric.get_scores()
                for k, v in score.items():
                    print('{}: {}'.format(k, round(v * 100, 2)) + '\n')
                
                # 第二种 metric，淘汰
                # for other_metric in all_metrics[1:]:
                #     metric.update_hist(other_metric.hist)
                # ious, miou = metric.compute_iou()
                # acc, macc = metric.compute_pixel_acc()
                # f1, mf1 = metric.compute_f1()
                # print(acc, "---------")
                # print(macc, "---------")
                # print(mf1, "---------")
                # print(miou, "---------")

    else:
        with torch.no_grad():
            model.eval()
            if args.mst:
                metric = evaluate_msf(
                    model,
                    val_loader,
                    config,
                    args.device,
                    [0.5, 0.75, 1.0, 1.25, 1.5],
                    True,
                    engine,
                )

            else:
                metric = evaluate(
                    model,
                    val_loader,
                    config,
                    args.device,
                    engine,
                    save_dir=args.save_path,
                )

            score, class_iou = metric.get_scores()
            for k, v in score.items():
                print('{}: {}'.format(k, round(v * 100, 2)) + '\n')
            
            # 第二种 metric，淘汰
            # for other_metric in all_metrics[1:]:
            #     metric.update_hist(other_metric.hist)
            # ious, miou = metric.compute_iou()
            # acc, macc = metric.compute_pixel_acc()
            # f1, mf1 = metric.compute_f1()
            # print(acc, "---------")
            # print(macc, "---------")
            # print(mf1, "---------")
            # print(miou, "---------")
