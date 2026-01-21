Come from: [DFormer](https://github.com/quanweiliu/DFormer)


想要改這個跑通這個代碼需要改動的地方：


change directory
conda activate dformer


訓練
- bash train.sh

測試和出圖
- bash eval.sh

數據集：
- ISPRS_Vaihingen_S
- ISPRS_Vaihingen_B
- ISPRS_Vaihingen_L
    - 數據源: infer import
    - 背景像素

需要改評價指標
- utils
    - metrics_new









