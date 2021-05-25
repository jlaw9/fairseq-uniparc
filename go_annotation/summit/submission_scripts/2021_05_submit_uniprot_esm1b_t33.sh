#!/bin/bash

#BSUB -P BIE108
#BSUB -q batch-hm
#BSUB -W 12:00
#BSUB -nnodes 1
#BSUB -J 2021_05_fairseq_uniprot_esm1b_t33
#BSUB -o /gpfs/alpine/bie108/proj-shared/jlaw/fairseq-job-output/2021_05_fairseq_cafa3_esm1b_t33/%J.out
#BSUB -e /gpfs/alpine/bie108/proj-shared/jlaw/fairseq-job-output/2021_05_fairseq_cafa3_esm1b_t33/%J.err
#BSUB -alloc_flags NVME
#BSUB -B

nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)

#module load ibm-wml-ce/1.7.1.a0-0
#conda activate /ccs/proj/bie108/jlaw/envs/fairseq_1.7.1
# had to use this series of module loads:
module purge
module load gcc
module load spectrum-mpi
#module load open-ce #
#conda activate fairseq-open-ce
module load open-ce/1.1.3-py37-0
module load cuda/10.2.89
conda activate /autofs/nccs-svm1_proj/bie108/jlaw/envs/fairseq-open-ce-1.1.3-py37-0

# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
TOTAL_UPDATES=50000      # Total number of training steps
WARMUP_UPDATES=100       # Warmup the learning rate over this many updates
PEAK_LR=1e-05            # Peak learning rate, adjust as needed
# UPDATE: had to shorten from 1024 to fit into 32GB GPU memory
TOKENS_PER_SAMPLE=768    # Max sequence length
#MAX_POSITIONS=1024       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=1          # Number of sequences per batch (batch size)
UPDATE_FREQ=16           # Increase the batch size 2x

SAVE_DIR=$MEMBERWORK/bie108/fairseq-uniparc/$LSB_JOBNAME
#DATA_DIR=/gpfs/alpine/bie108/proj-shared/swissprot_go_annotation/fairseq_cafa3
# This directory contains the annotations & ontology directly from cafa3 
DATA_DIR=/ccs/proj/bie108/jlaw/swissprot_go_annotation/2021_05_fairseq_uniprot_expc_ann10

BASE_DIR="$(readlink -e ../../../)"
OBO_FILE="$BASE_DIR/inputs/2021-05/2021-05-25-go.obo.gz"
RESTRICT_TERMS_FILE="$BASE_DIR/inputs/2021-05/terms-10ann.txt.gz"
# Number of GO terms
NUM_CLASSES="$(gzip -cd $RESTRICT_TERMS_FILE | wc -l)"

jsrun -n ${nnodes} -a 1 -c 42 -r 1 cp -r ${DATA_DIR} /mnt/bb/${USER}/data

jsrun -n ${nnodes} -g 6 -c 42 -r 1 -a 1 -b none \
    fairseq-train --distributed-port 23456 \
    /mnt/bb/${USER}/data \
    --fp16 --memory-efficient-fp16 \
    --user-dir ${BASE_DIR}/go_annotation/ \
    --task="sentence_labeling" \
    --criterion="go_prediction" \
    --obo-file="${OBO_FILE}" --restrict-terms-file="${RESTRICT_TERMS_FILE}" \
    --regression-target \
    --classification-head-name="go_prediction" \
    --num-classes $NUM_CLASSES --init-token 0 \
    --batch-size $MAX_SENTENCES \
    --total-num-update $TOTAL_UPDATES \
    --shorten-method='random_crop' \
    --reset-optimizer --reset-dataloader --reset-meters \
    --arch esm1b_t33 --max-positions=$TOKENS_PER_SAMPLE  \
    --optimizer=adam --adam-betas="(0.9,0.98)" --adam-eps=1e-6 --clip-norm=0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --validate-interval-updates 500 \
    --save-interval-updates 2000 --keep-interval-updates 1 \
    --update-freq $UPDATE_FREQ --save-dir $SAVE_DIR \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 10 \
    --tensorboard-logdir=$MEMBERWORK/bie108/fairseq-tensorboard/$LSB_JOBNAME \
    --ddp-backend=legacy_ddp \
    --find-unused-parameters

