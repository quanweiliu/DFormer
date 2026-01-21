# 默认单机可运行，同时允许多机时被外部覆盖
# 这是多台计算机同时运行的设置，我只有一台机器，多个GPU，所以都是默认值。

GPUS=2
# 告诉 torchrun 这次分布式任务一共有多少台机器（节点）参与。
# 也就是分布式训练/推理里的 机器数量，不是 GPU 数量。
NNODES=1

# bash 里的参数展开语法，如果环境变量 NODE_RANK 已经存在，就用它原来的值；
# 如果没有设置，就把 NODE_RANK 设为默认值 0。
NODE_RANK=${NODE_RANK:-0}   # 表示 第几台机器，默认为 0
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 计算机上可见的 GPU 列表
export CUDA_VISIBLE_DEVICES="0,1"
# 开启这个，PyTorch 在使用 torch.compile() / Dynamo 编译时会打印更多调试信息。
export TORCHDYNAMO_VERBOSE=1    

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=local_configs.ISPRS_Vaihingen_S \
    --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    --pad_SUNRGBD \
    --no-use_seed

# config for DFormers on NYUDepthv2
# local_configs.NYUDepthv2.DFormer_Large
# local_configs.NYUDepthv2.DFormer_Base
# local_configs.NYUDepthv2.DFormer_Small
# local_configs.NYUDepthv2.DFormer_Tiny
# local_configs.NYUDepthv2.DFormerv2_S
# local_configs.NYUDepthv2.DFormerv2_B
# local_configs.NYUDepthv2.DFormerv2_L

# local_configs.ISPRS_Vaihingen_S
# local_configs.ISPRS_Vaihingen_B
# local_configs.ISPRS_Vaihingen_L

# local_configs.ISPRS_Potsdam_S
# local_configs.ISPRS_Potsdam_B
# local_configs.ISPRS_Potsdam_L






















# config for DFormers on SUNRGBD
# local_configs.SUNRGBD.DFormer_Large
# local_configs.SUNRGBD.DFormer_Base
# local_configs.SUNRGBD.DFormer_Small
# local_configs.SUNRGBD.DFormer_Tiny
# local_configs.SUNRGBD.DFormer_v2_S
# local_configs.SUNRGBD.DFormer_v2_B
# local_configs.SUNRGBD.DFormer_v2_L