#!/bin/bash

#BSUB -P BIE108
#BSUB -q batch-hm
#BSUB -W 12:00
#BSUB -nnodes 1
#BSUB -J 20210326_cafa3_preinit
#BSUB -o /ccs/home/jlaw/fairseq-job-output/%J.out
#BSUB -e /ccs/home/jlaw/fairseq-job-output/%J.err
#BSUB -alloc_flags NVME
#BSUB -B

nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)

#module load ibm-wml-ce/1.7.1.a0-0
#conda activate /ccs/proj/bie108/jlaw/envs/fairseq_1.7.1
# had to use this series of module loads:
module purge
module load gcc
module load spectrum-mpi
module load open-ce #
conda activate fairseq-open-ce

# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
TOTAL_UPDATES=50000      # Total number of training steps
WARMUP_UPDATES=100       # Warmup the learning rate over this many updates
PEAK_LR=1e-05            # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=1024   # Max sequence length
MAX_POSITIONS=1024       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8          # Number of sequences per batch (batch size)
UPDATE_FREQ=2            # Increase the batch size 2x

NUM_CLASSES=28474        # Number of GO terms in CAFA3

SAVE_DIR=$MEMBERWORK/bie108/fairseq-uniparc/$LSB_JOBNAME
#DATA_DIR=/gpfs/alpine/bie108/proj-shared/swissprot_go_annotation/fairseq_cafa3
DATA_DIR=/ccs/proj/bie108/jlaw/swissprot_go_annotation/fairseq_cafa3
ROBERTA_PATH=$MEMBERWORK/bie108/fairseq-uniparc/roberta_base_checkpoint/checkpoint_best.pt

jsrun -n ${nnodes} -a 1 -c 42 -r 1 cp -r ${DATA_DIR} /mnt/bb/${USER}/data

jsrun -n ${nnodes} -g 6 -c 42 -r1 -a1 -b none \
    fairseq-train --distributed-port 23456 \
    --fp16 /mnt/bb/${USER}/data \
    --user-dir $HOME/projects/deepgreen/fairseq-uniparc-fork/go_annotation/ \
    --restore-file $ROBERTA_PATH \
    --classification-head-name='go_prediction' \
    --task sentence_labeling --criterion go_prediction --regression-target --num-classes $NUM_CLASSES --init-token 0 \
    --arch roberta_base --max-positions $TOKENS_PER_SAMPLE --shorten-method='random_crop' \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --validate-interval-updates 500 \
    --save-interval-updates 2000 --keep-interval-updates 1 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ --save-dir $SAVE_DIR \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 10 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --tensorboard-logdir=$MEMBERWORK/bie108/fairseq-tensorboard/$LSB_JOBNAME \
    --find-unused-parameters
