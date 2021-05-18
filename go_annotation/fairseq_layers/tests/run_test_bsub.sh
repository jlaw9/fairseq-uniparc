#!/bin/bash

#BSUB -P BIE108
##BSUB -q batch-hm
#BSUB -q debug
#BSUB -W 0:20
#BSUB -nnodes 1
#BSUB -J esm-test
#BSUB -o /ccs/home/jlaw/fairseq-job-output/debug/%J.out
#BSUB -e /ccs/home/jlaw/fairseq-job-output/debug/%J.err
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

#PATH=`cat /ccs/home/jlaw/projects/deepgreen/fairseq-uniparc-fork/go_annotation/fairseq_layers/tests/working-path.txt`
#export PATH
# not sure when this got set, but I need these on the python path for it to work
#PYTHONPATH="/sw/summit/xalt/1.2.1/site:/sw/summit/xalt/1.2.1/libexec:$PYTHONPATH"
#export PYTHONPATH
#echo $PATH
#echo $PYTHONPATH
#conda list

# make sure the same versions are being used
python -c "import torch; print(f'torch: {torch.__version__}');"
python -c "import fairseq; print(f'fairseq: {fairseq.__version__}');"
python -c "import numpy; print(f'numpy: {numpy.__version__}');"

#module list
#conda list > bsub-conda-list.txt

# export NCCL_DEBUG=INFO
#export OMP_NUM_THREADS=4

BASE_DIR="$(readlink -e ../../../)"
OBO_FILE="$BASE_DIR/inputs/2020-12-swissprot/go-basic.obo.gz"
RESTRICT_TERMS_FILE="$BASE_DIR/inputs/2020-12-swissprot/terms.csv.gz"
# Number of GO terms
NUM_CLASSES="$(gzip -cd $RESTRICT_TERMS_FILE | wc -l)"

echo "ONE CUDA DEVICE"

CUDA_VISIBLE_DEVICES=0 \
  fairseq-train ../../eagle/criterion_development/fairseq_swissprot_debug/ \
  --task sentence_labeling \
  --user-dir ../../../go_annotation/ \
  --criterion go_prediction \
  --obo-file="${OBO_FILE}" --restrict-terms-file="${RESTRICT_TERMS_FILE}" \
  --regression-target \
  --classification-head-name='go_prediction' \
  --num-classes $NUM_CLASSES --init-token 0 \
  --batch-size 1 \
  --update-freq 16 \
  --total-num-update 10 \
  --shorten-method='random_crop' \
  --reset-optimizer --reset-dataloader --reset-meters \
  --weight-decay 0.1 \
  --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
  --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr 1e-05 \
  --arch esm1b_t33 \
  --max-positions 1024 \
  --save-interval 1 \
  --log-format simple --log-interval 1 \
  --ddp-backend=legacy_ddp \
  --fp16 --memory-efficient-fp16 \
  --find-unused-parameters

  #--arch esm1_t6 \
