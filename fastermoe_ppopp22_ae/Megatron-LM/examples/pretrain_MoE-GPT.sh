#! /bin/bash

BS=64
GPUS_PER_NODE=4
MP_SIZE=1
if [ -z $1 ]
then
	TASK_NAME="unamed-$(date +%m%d-%H%M)"
else
	TASK_NAME=$1
fi

master=`scontrol show hostname $SLURM_NODELIST| head -n 1`
time=$(date "+%m-%d-%H:%M")
export MASTER_ADDR=$master
MASTER_PORT=1234


if [ ! -z $SLURM_PROCID ]
then
    export NODE_RANK=$SLURM_PROCID
    export NNODES=$SLURM_NPROCS
else
    export NNODES=1
    export NODE_RANK=0
fi
export OMP_NUM_THREADS=4

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export FMOE_FUSE_GRAN=$WORLD_SIZE

DATA_PATH=./openwebtext/my_gpt2_text_document
VOCAB_PATH=./openwebtext/gpt2-vocab.json
MERGE_PATH=./openwebtext/gpt2-merges.txt
CHECKPOINT_PATH=path_to_ckpt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


# BAL_STG="hir"
BAL_STG="naive"
# BAL_STG="naiveloss"
# BAL_STG="noisy"
#BAL_STG="switch"
#BAL_STG="blockplus"
# BAL_STG="gshard"
echo Training on $(hostname) with strategy $BAL_STG

python3 -m torch.distributed.run ${DISTRIBUTED_ARGS[@]}  pretrain_gpt.py \
	   --fmoefy \
       --balance-strategy $BAL_STG \
       --num-experts 1 \
       --num-layers $2 \
	   --hidden-size $3 \
       --hidden-hidden-size 1408 \
	   --num-attention-heads 16 \
	   --micro-batch-size $(expr $BS / $GPUS_PER_NODE / $NNODES) \
	   --global-batch-size $BS \
	   --seq-length 256 \
	   --max-position-embeddings 1024 \
	   --train-iters 100 \
	   --lr-decay-iters 32000 \
	   --save $CHECKPOINT_PATH \
	   --load $CHECKPOINT_PATH \
	   --data-path $DATA_PATH \
	   --vocab-file $VOCAB_PATH \
       --merge-file $MERGE_PATH \
	   --data-impl mmap \
	   --split 949,50,1 \
       --distributed-backend nccl \
       --lr $5 \
       --lr-decay-style cosine \
       --min-lr $6 \
       --weight-decay 0.1 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 1000000 \
       --eval-interval 1000 \
	   --tensorboard-dir logs/gpt/$TASK_NAME \
       --eval-iters 1000 \
       --top-k $4 \
       --super-param $strr \
       --balance-loss-weight 1e-3 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 
    #   --seq-loss \
    #   --fp16 
