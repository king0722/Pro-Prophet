#!/bin/bash

# Runs the "345M" parameter model

source ~/scripts/torch.env
BS=64
GPUS_PER_NODE=8
MP_SIZE=1
if [ -z $1 ]
then
	export TASK_NAME="unamed-$(date +%m%d-%H%M)"
else
	export TASK_NAME=$1
fi

# Change for multinode config
if [ -z $MASTER_ADDR ]
then
    MASTER_ADDR=localhost
fi
MASTER_PORT=8980
if [ -z $OMPI_COMM_WORLD_SIZE ]
then
    NNODES=1
    NODE_RANK=0
else
    NNODES=$OMPI_COMM_WORLD_SIZE
    NODE_RANK=$OMPI_COMM_WORLD_RANK
fi
export OMP_NUM_THREADS=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=data/my-bert_text_sentence
DATA_PATH=data/wiki-small_text_sentence
DATA_PATH=/home/laekov/dataset/nlp/my-bert_text_sentence
VOCAB_PATH=/home/laekov/dataset/gpt2-vocab.json
MERGE_PATH=/home/laekov/dataset/gpt2-merges.txt
CHECKPOINT_PATH=ckpt/gpt-$(date +%y%m%d-%H%M%S)

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --fmoefy \
       --balance-strategy gshard \
       --num-experts 1 \
	   --num-layers 12 \
	   --hidden-size 2048 \
	   --num-attention-heads 16 \
	   --micro-batch-size $(expr $BS / $GPUS_PER_NODE / $NNODES) \
	   --global-batch-size $BS \
	   --seq-length 256 \
	   --max-position-embeddings 1024 \
	   --train-iters 100000 \
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
       --log-interval 100 \
       --save-interval 1000000 \
       --eval-interval 1000 \
	   --tensorboard-dir logs/gpt/$TASK_NAME \
       --eval-iters 10
	  
