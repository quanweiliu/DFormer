
In this repository, I apply the DFormer in my study area (Remote sensing). I found this code is so powerful in remote sensing image semantic segmentaion.

How to change the dataset, 1\) compare the local_configs/template/DFormer_Large.py and ISPRS_Potsdam_L you will know how to configure. And  2\) compare utils/dataloader/RGBXDataset and utils/dataloader/TIFDataset you will know how to use your own datasets.


The main difference of this version compared to orgianl version is estimation metric. I also introduce two more remote sensing datasets and clean up the code. The concret details refer to 2.File description Metric.


<!-- 
目前的版本已经稳定，下一步要做的是将 mask2former 加到这个模块的头上。
好像不是很难加，输出的格式是符合 maskformer2 的输入要求的。需要改的有
数据的输入格式
模型
criterion -->

## 1.Get Start
Install: Refer to [DFormer](https://github.com/quanweiliu/DFormer)

## 2.File description
Train： bash train.sh
Test: bash test.sh
infer: bash infer.sh   repetition abolished

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

### Test RGBD datasets:
1. 从 [DFormer](https://github.com/quanweiliu/DFormer) 下载权重文件，并放到
    - checkpoints

2. 配置 test.sh 文件
    - 支持单机单卡或者单机多卡，但是默认为单机多卡，只需要确定 GPU 数量即可
    - 以 DFormerv2_B 为例子，选择 --config=local_configs.NYUDepthv2.DFormerv2_B
    - 配置权重 --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/checkpoints/DFormerv2_Base_NYU.pth

3. 在 utils/test.py 
    - 打开 from utils.dataloader.RGBXDataset import RGBXDataset 应为数据类型不同

4. 在 utils/val_mm.py 
    - 打开 metrics = Metrics(n_classes, config.background, device) 因为不同数据集的背景类别不同

5. 运行文件
    - cd /home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer
    - conda activate dformer
    - bash infer.sh

6. 得到结果
    - 结果只有打印，没有保存
    - 想要保存可视化的结果打开 evaluate 函数的 save_dir





### Train RS datasets:
1. 配置 train.sh 文件
    - 支持单机单卡或者单机多卡，但是默认为单机多卡，只需要确定 GPU 数量即可
    - 选择 --config=local_configs.ISPRS_Vaihingen_S
    - 配置 ISPRS_Vaihingen_S, 
        - 选择 train 和 val 数据集和 train 和 val 样本数量
        - background 255, 这是填充的背景类别。

2. 在 utils/train.py
    - 打开 from utils.dataloader.TIFDataset import RGBXDataset

3. 在 utils/val_mm.py 
    - 打开 metrics = Metrics(n_classes, config.ignore_label, device) 因为不同数据集的背景类别不同

### Test RS datasets:
1. 配置 infer.sh 文件
    - 支持单机单卡或者单机多卡，但是默认为单机多卡，只需要确定 GPU 数量即可
    - 以 ISPRS_Vaihingen_S 为例子，选择 --config=local_configs.ISPRS_Vaihingen_S

2. 配置数据文件
    - 配置 ISPRS_Vaihingen_S, 
        - 选择 test 数据集和 test 样本数量
        - **ignore_label 5**， 这是图像自带的背景类别。我们的策略是在训练的时候算上这个类别，但是在测试的时候不要这个类别。这是遥感社区在这个数据集上的通用做法。
    - 配置权重 --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/results/Vaihingen_DFormerv2_S_20251023-140912/epoch-97_miou_61.81.pth

2. 在 utils/test.py 
    - 打开 from utils.dataloader.TIFDataset import RGBXDataset

3. 在 utils/val_mm.py 
    - 打开 metrics = Metrics(n_classes, config.ignore_label, device) 因为不同数据集的背景类别不同

4. 运行文件
    - cd /home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer
    - conda activate dformer
    - bash test.sh

5. 得到结果
    - 结果只有打印，没有保存
    - 想要保存可视化的结果打开 evaluate 函数的 save_dir


## 4.Thanks

Come from: [DFormer](https://github.com/quanweiliu/DFormer)
