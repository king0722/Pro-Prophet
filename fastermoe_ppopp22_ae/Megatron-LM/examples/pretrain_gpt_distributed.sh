#! /bin/bash

# Runs the "345M" parameter model

BS=64
GPUS_PER_NODE=$NPN
MP_SIZE=1
if [ -z $1 ]
then
	TASK_NAME="unamed-$(date +%m%d-%H%M)"
else
	TASK_NAME=$1
fi

if [ -z $SLURM_JOB_ID ]
then
    export MASTER_ADDR=localhost
else
    export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
fi

# Change for multinode config
if [ -z $MASTER_ADDR ]
then
    MASTER_ADDR=localhost
    MASTER_PORT=8980
fi
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

DATA_PATH=$DATA_PREFIX/my-bert_text_sentence
VOCAB_PATH=$DATA_PREFIX/gpt2-vocab.json
MERGE_PATH=$DATA_PREFIX/gpt2-merges.txt
CHECKPOINT_PATH=ckpt/gpt-$(date +%y%m%d-%H%M%S)


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

echo Training on $(hostname) with strategy $BAL_STG

python3 -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --fmoefy \
       --num-experts 1 \
       --balance-strategy $BAL_STG \
	   --num-layers 12 \
	   --hidden-size 2048 \
	   --num-attention-heads 16 \
	   --micro-batch-size $(expr $BS / $GPUS_PER_NODE / $NNODES) \
	   --global-batch-size $BS \
	   --seq-length 256 \
	   --max-position-embeddings 1024 \
	   --train-iters 500 \
	   --lr-decay-iters 320000 \
	   --save $CHECKPOINT_PATH \
	   --load $CHECKPOINT_PATH \
	   --data-path $DATA_PATH \
	   --vocab-file $VOCAB_PATH \
       --merge-file $MERGE_PATH \
	   --data-impl mmap \
	   --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 1000000 \
       --eval-interval 1000 \
	   --tensorboard-dir logs/gpt/$TASK_NAME \
       --eval-iters 10
	  
