import os
import cv2
import torch
import numpy as np
import rasterio
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import torch.utils.data as data
from torchvision.transforms import v2


def rgb_to_2D_label(label):
    """
    Suply our label masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    ImSurf = [255, 255, 255] # Impervious 
    Clutter = [0, 0, 255] # Building  
    Car = [0, 255, 255] # Vegetation 
    Tree = [0, 255, 0] # Tree
    LowVeg = [255, 255, 0] # Car
    Building = [255, 0, 0] # Clutter 

    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg[np.all(label==ImSurf,axis=-1)] = 0
    label_seg[np.all(label==Building,axis=-1)] = 1
    label_seg[np.all(label==LowVeg,axis=-1)] = 2
    label_seg[np.all(label==Tree,axis=-1)] = 3
    label_seg[np.all(label==Car,axis=-1)] = 4
    label_seg[np.all(label==Clutter,axis=-1)] = 5

    # label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg


def get_path(
    dataset_name,
    _rgb_path,
    _rgb_format,
    _x_path,
    _x_format,
    _gt_path,
    _gt_format,
    x_modal,
    item_name,
):
    if dataset_name == "StanFord2D3D":
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("/rgb/", "/depth/").replace("_rgb", "_newdepth")
            + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "")
            .replace(".png", "")
            .replace("/rgb/", "/semantic/")
            .replace("_rgb", "_newsemantic")
            + _gt_format,
        )
    elif dataset_name == "new_stanford":
        area = item_name.split(" ")[0]
        name = item_name.split(" ")[1]
        # print(area,name)
        rgb_path = os.path.join(_rgb_path, area + "/image/" + name + _rgb_format)
        d_path = os.path.join(_x_path, area + "/hha/" + name + _x_format)
        gt_path = os.path.join(_gt_path, area + "/label/" + name + _gt_format)
    elif dataset_name == "KITTI-360":
        rgb_path = os.path.join(_rgb_path, item_name.split(" ")[0])
        d_path = os.path.join(
            _x_path,
            item_name.split(" ")[0]
            .replace("data_2d_raw", "data_3d_rangeview")
            .replace("image_00/data_rect", "velodyne_points/data"),
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.split(" ")[1].replace("data_2d_semantics", "data_2d_semantics_trainID"),
            # .replace("/train", ""),
        )
    elif dataset_name == "Scannet":
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("color", "convert_depth") + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("color", "convert_label") + _gt_format,
        )
    elif dataset_name == "MFNet":
        rgb_path = os.path.join(_rgb_path, item_name + _rgb_format)
        d_path = os.path.join(_x_path, item_name + _x_format)
        gt_path = os.path.join(_gt_path, item_name + _gt_format)
    elif dataset_name == "EventScape":
        item_name = item_name.split(".png")[0]
        img_name = item_name.split("/")[-1]
        img_id = img_name.replace("_image", "").split("_")[-1]
        rgb_path = os.path.join(_rgb_path, item_name + _rgb_format)
        d_path = os.path.join(
            _x_path,
            item_name.replace("rgb", "depth").replace("data", "frames").replace(img_name, img_id) + _x_format,
        )
        e_path = os.path.join(
            _x_path,
            item_name.replace("rgb", "events").replace("data", "frames").replace(img_name, img_id) + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace("rgb", "semantic").replace("image", "gt_labelIds") + _gt_format,
        )

    elif dataset_name == "Vaihingen" or dataset_name == "Potsdam":
        # print("Vaihingen", item_name)

        # item_name = item_name.split("/")[1].split(".jpg")[0]
        rgb_path = os.path.join(_rgb_path, item_name + _rgb_format)
        d_path = os.path.join(_x_path, item_name + _x_format)
        gt_path = os.path.join(_gt_path, item_name + _gt_format)

    else:
        item_name = item_name.split("/")[1].split(".jpg")[0]
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "") + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "").replace(".png", "") + _gt_format,
        )
    path_result = {"rgb_path": rgb_path, "d_path": d_path, "gt_path": gt_path}
    for modal in x_modal:
        path_result[modal + "_path"] = eval(modal + "_path")
    return path_result


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting["rgb_root"]
        self._rgb_format = setting["rgb_format"]
        self._gt_path = setting["gt_root"]
        self._gt_format = setting["gt_format"]
        self._transform_gt = setting["transform_gt"]
        self._x_path = setting["x_root"]
        self._x_format = setting["x_format"]
        self._x_single_channel = setting["x_single_channel"]
        self._train_source = setting["train_source"]
        self._val_source = setting["val_source"]
        self._test_source = setting["test_source"]
        self.class_names = setting["class_names"]
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.dataset_name = setting["dataset_name"]
        self.x_modal = setting.get("x_modal", ["d"])   # x_modal 有值，就返回 x_modal，否则返回 ['d']
        self.backbone = setting["backbone"]

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        path_dict = get_path(
            self.dataset_name,
            self._rgb_path,
            self._rgb_format,
            self._x_path,
            self._x_format,
            self._gt_path,
            self._gt_format,
            self.x_modal,      # RGB 的 x modal
            item_name,
        )
        # if self.dataset_name == "SUNRGBD" and self.backbone.startswith("DFormerv2"):
        #     rgb_mode = "RGB"  # some checkpoints are run by BGR and some are on RGB, need to select
        # else:
        #     rgb_mode = "BGR"
        # rasterio.open(filepath)
        rgb = rasterio.open(path_dict["rgb_path"]).read().transpose(1, 2, 0)
        x = rasterio.open(path_dict["d_path"]).read()
        x = np.tile(x, (3, 1, 1)).transpose(1, 2, 0)   #通过复制通道数，变成3通道
        gt = rasterio.open(path_dict["gt_path"]).read().transpose(1, 2, 0)
        gt = rgb_to_2D_label(gt)[:, :, 0]
        # print("rgb", rgb.shape)
        # print("gt", gt.shape, gt.min(), gt.max(), '---')

        # gt = self._open_image(path_dict["gt_path"], cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        # if self._transform_gt:
        #     gt = self._gt_transform(gt)

        # x = {}
        # for modal in self.x_modal:
        #     if modal == "d":
        #         x[modal] = self._open_image(path_dict[modal + "_path"], cv2.IMREAD_GRAYSCALE)
        #         x[modal] = cv2.merge([x[modal], x[modal], x[modal]])
        #     else:
        #         x[modal] = self._open_image(path_dict[modal + "_path"], "RGB")
        # if len(self.x_modal) == 1:
        #     x = x[self.x_modal[0]]

        # if self.dataset_name == "Scannet":
        #     rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
        #     x = cv2.resize(x, (640, 480), interpolation=cv2.INTER_LINEAR)
        #     gt = cv2.resize(gt, (640, 480), interpolation=cv2.INTER_NEAREST)
        # elif self.dataset_name == "StanFord2D3D":
        #     rgb = cv2.resize(rgb, dsize=(480, 480), interpolation=cv2.INTER_LINEAR)
        #     x = cv2.resize(x, dsize=(480, 480), interpolation=cv2.INTER_LINEAR)
        #     gt = cv2.resize(gt, dsize=(480, 480), interpolation=cv2.INTER_NEAREST)

        # print("rgb", rgb.shape, "x", x.shape, "gt", gt.shape)
        # image (256, 256, 3) x (256, 256, 3) gt (256, 256)


        # if self._x_single_channel:
        #     x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
        #     x = cv2.merge([x, x, x])
        # else:
        #     x = self._open_image(x_path, cv2.COLOR_BGR2RGB)

        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        # for modal in x:
        x = torch.from_numpy(np.ascontiguousarray(x)).float()

        # 合成数据
        x = v2.RandomPerspective(distortion_scale=0.6, p=1)(x)
        # print("x_augmented done")

        # if self._split_name == "train":
        #     rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        #     gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        #     x = torch.from_numpy(np.ascontiguousarray(x)).float()
        # else:
        #     rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        #     gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        #     x = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(path_dict["rgb_path"]), n=len(self._file_names))

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ["train", "val", 'test']
        source = self._train_source
        if split_name == "val":
            source = self._val_source
        elif split_name == "test":
            source = self._test_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[: length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        if mode == "RGB":
            img = np.array(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), dtype=dtype)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == "BGR":
            img = np.array(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), dtype=dtype)
        else:
            img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img
    # @staticmethod
    # def _open_rs(filepath):
    #     img = rasterio.open(filepath)
    #     return img
    
    @staticmethod
    def _gt_transform(gt):
        return gt - 1

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
