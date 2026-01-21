import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

remoteip = os.popen("pwd").read()
C.root_dir = "/home/icclab/Documents/lqw/DatasetMMF"
# C.root_dir = "/home/icclab/Documents/lqw/Multimodal_Segmentation/RSSegExpts/assets/test_Vai"
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = "Vaihingen"
C.dataset_path = osp.join(C.root_dir, "Vaihingen")
C.rgb_root_folder = osp.join(C.dataset_path, "images256")
C.rgb_format = ".tif"
C.gt_root_folder = osp.join(C.dataset_path, "masks256")
C.gt_format = ".tif"
C.gt_transform = True
C.x_root_folder = osp.join(C.dataset_path, "DSM256")
C.x_format = ".tif"
C.x_is_single_channel = True
C.train_source = osp.join(C.dataset_path, "train.txt")
# C.eval_source = osp.join(C.dataset_path, "val.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 1470
# C.num_eval_imgs = 611
C.num_eval_imgs = 2416
C.num_classes = 6
C.class_names = ["Imp.Surf.", "Tree", "Low Veg.", "Building",  "Car", "Clutter"]

"""Image Config"""
C.background = 5
C.image_height = 256
C.image_width = 256
C.norm_mean = np.array([0.4731, 0.3206, 0.3182])
C.norm_std = np.array([0.1970, 0.1306, 0.1276])

""" Settings for network, this would be different for each kind of model"""
C.backbone = "DFormerv2_L"  # Remember change the path below.
C.pretrained_model = "checkpoints/DFormerv2_Large_pretrained.pth"
C.decoder = "ham"
C.decoder_embed_dim = 1024
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 100
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 2
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10
C.channels = [96, 192, 288, 576]

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.3
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]  # [0.75, 1, 1.25] #
C.eval_flip = True  # False #
C.eval_crop_size = [256, 256]  # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 50
C.checkpoint_step = 25

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath("results/" + C.dataset_name + "_" + C.backbone)
C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(
    osp.join(C.log_dir, "results")
)  #'/mnt/sda/repos/2023_RGBX/pretrained/'#osp.abspath(osp.join(C.log_dir, "checkpoint"))
if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir, exist_ok=True)
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"
