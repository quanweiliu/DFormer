# CUDA_VISIBLE_DEVICES=0,1
# config -> which model config
# continue_fpath -> the trained pth path
GPUS=2
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29958}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/infer.py \
    --config=local_configs.ISPRS_Potsdam_B \
    --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/results/Potsdam_DFormerv2_B_20251024-001638/epoch-30_miou_74.69.pth \
    --save_path "" \
    --gpus=$GPUS

# choose the dataset and DFormer for evaluating

# NYUv2 DFormers
# --config=local_configs.NYUDepthv2.DFormerv2_B
# --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/checkpoints/DFormerv2_Base_NYU.pth

# --config=local_configs.NYUDepthv2.DFormerv2_L
# --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/checkpoints/DFormerv2_Large_NYU.pth

# --config=local_configs.NYUDepthv2.DFormer_Large/Base/Small/Tiny
# --continue_fpath=checkpoints/NYUv2_DFormer_Large/Base/Small/Tiny.pth

# SUNRGBD DFormers
# --config=local_configs.SUNRGBD.DFormer_Large/Base/Small/Tiny
# --continue_fpath=checkpoints/SUNRGBD_DFormer_Large/Base/Small/Tiny.pth

# ISPRS Vaihingen DFormer
# --config=local_configs.ISPRS_Vaihingen_S
# --config=local_configs.ISPRS_Vaihingen_B
# --config=local_configs.ISPRS_Vaihingen_L
# --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/results/Vaihingen_DFormerv2_S_20251023-140912/epoch-97_miou_61.81.pth
# --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/results/Vaihingen_DFormerv2_B_20251023-145429/epoch-81_miou_62.35.pth
# --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/results/Vaihingen_DFormerv2_L_20251022-235958/epoch-92_miou_63.53.pth


# ISPRS Potsdam DFormer
# --config=local_configs.ISPRS_Potsdam_S
# --config=local_configs.ISPRS_Potsdam_B
# --config=local_configs.ISPRS_Potsdam_L
# --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/results/Potsdam_DFormerv2_S_20251023-180934/epoch-52_miou_73.59.pth
# --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/results/Potsdam_DFormerv2_S_20251023-180934/epoch-63_miou_74.11.pth
# --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/results/Potsdam_DFormerv2_B_20251024-001638/epoch-30_miou_74.69.pth



