
In this repository, I apply the DFormer in my study area (Remote sensing). I found this code is so powerful in remote sensing image semantic segmentaion.

How to change the dataset, 1\) compare the local_configs/template/DFormer_Large.py and ISPRS_Potsdam_L you will know how to configure. And  2\) compare utils/dataloader/RGBXDataset and utils/dataloader/TIFDataset you will know how to use your own datasets.


The difference of this version compared to orgianl version is estimation metric. The concret details refer to 2.File description Metric.



## 1.Get Start
Install: Refer to [DFormer](https://github.com/quanweiliu/DFormer)

## 2.File description
Train： bash train.sh

Test: bash eval.sh

Visulization: bash infer.sh

Datasets:
- ISPRS_Vaihingen_S
- ISPRS_Vaihingen_B
- ISPRS_Vaihingen_L

Metric：
- utils
    - metrics_new：我有两套计算 IoU F1 等指标的代码，都是正确的，但是对于背景像素的计算逻辑却有所不同，导致最终结果有些微的差别。选择一种即可，保持对所有的数据集应用相同的指标进行评价即可。

    - 在 Vaihingen 数据集，背景像素 5. 第一种计算方法上得到 ACC [88.78, 93.39, 76.63, 89.39, 73.08, 0.0] 6 个类别平均 mAcc 70.021, mIou: 61.53, F1 70.4。这是错误的，最后一个是 0，不应该引入到最终的精度评价中。
    - 第二种计算方法得到 [88.69， 93.1， 76.87， 89.33， 74.25， 34.31]， 前 5 个类别平均 mAcc 87.22, mIou: 73.47, F1: 84.36. 这才正确嘛！

    - 在 NYUv2 数据集上, 背景像素255. 第一种计算算法：mAcc 71.03, mIou: 57.2, F1: 70.92.
    - 第二种计算算法：mAcc 79.2, mIou: 56.69, F1: 70.45.
    - 在没有背景像素的情况下， mIou 和 F1 值查不多，仅有微小的差别。

    - 综上，都采用第二种更加通用的评价指标。

RGBXDataset: load RGBX datasets
TIFDataset: load TIF datasets


## 3.RUN

### test RGBD datasets:
1. 从 [DFormer](https://github.com/quanweiliu/DFormer) 下载权重文件，并放到
    - checkpoints

2. 配置 infer.sh 文件
    - 支持单机单卡或者单机多卡，但是默认为单机多卡，只需要确定 GPU 数量即可
    - 以 DFormerv2_B 为例子，选择 --config=local_configs.NYUDepthv2.DFormerv2_B
    - 配置权重 --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/checkpoints/DFormerv2_Base_NYU.pth

3. 在 utils/infer.py 
    - 打开 from utils.dataloader.RGBXDataset import RGBXDataset
    - 打开 torch.load(args.continue_fpath)["state_dict"]

4. 运行文件
    - cd /home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer
    - conda activate dformer
    - bash infer.sh

5. 得到结果
    - 结果只有打印，没有保存


### train RS datasets:
1. 配置 train.sh 文件
    - 支持单机单卡或者单机多卡，但是默认为单机多卡，只需要确定 GPU 数量即可
    - 选择 --config=local_configs.ISPRS_Vaihingen_S
    - 配置 ISPRS_Vaihingen_S, 
        - 选择 train 和 val 数据集和 train 和 val 样本数量
        - **background 255**, 只有我这个需要改，对于RGB-X数据集，他的背景就是 255 不需要改。

2. 在 utils/train.py
    - 打开 from utils.dataloader.TIFDataset import RGBXDataset



### Test RS datasets:
1. 配置 infer.sh 文件
    - 支持单机单卡或者单机多卡，但是默认为单机多卡，只需要确定 GPU 数量即可
    - 以 ISPRS_Vaihingen_S 为例子，选择 --config=local_configs.ISPRS_Vaihingen_S
    - 配置 ISPRS_Vaihingen_S, 
        - 选择 test 数据集和 test 样本数量
        - **background 5**
    - 配置权重 --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/results/Vaihingen_DFormerv2_S_20251023-140912/epoch-97_miou_61.81.pth

2. 在 utils/infer.py 
    - 打开 from utils.dataloader.TIFDataset import RGBXDataset
    - 打开 torch.load(args.continue_fpath)["model"]

3. 运行文件
    - cd /home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer
    - conda activate dformer
    - bash infer.sh

4. 得到结果
    - 结果只有打印，没有保存



## 4.Thanks

Come from: [DFormer](https://github.com/quanweiliu/DFormer)
