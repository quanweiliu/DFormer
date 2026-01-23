# CUDA_VISIBLE_DEVICES=0,1
# config -> which model config
# continue_fpath -> the trained pth path
GPUS=2
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0,1"
export TORCHDYNAMO_VERBOSE=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/test.py \
    --config=local_configs.ISPRS_Vaihingen_S \
    --continue_fpath=/home/icclab/Documents/lqw/Multimodal_Segmentation/DFormer/results/Vaihingen_DFormerv2_S_20251023-140912/epoch-97_miou_61.81.pth \
    --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --no-mst \
    --compile_mode="reduce-overhead" \
    --amp \
    # --pad_SUNRGBD \

# choose the dataset and DFormer for evaluating

# NYUv2 DFormers
# --config=local_configs.NYUDepthv2.DFormer_Large/Base/Small/Tiny
# --continue_fpath=checkpoints/NYUv2_DFormer_Large/Base/Small/Tiny.pth

# SUNRGBD DFormers
# --config=local_configs.SUNRGBD.DFormer_Large/Base/Small/Tiny
# --continue_fpath=checkpoints/SUNRGBD_DFormer_Large/Base/Small/Tiny.pth
