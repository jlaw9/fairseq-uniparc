# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

TOTAL_UPDATES=50000      # Total number of training steps
WARMUP_UPDATES=100       # Warmup the learning rate over this many updates
PEAK_LR=1e-05            # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=1024   # Max sequence length
MAX_POSITIONS=1024       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=2          # Number of sequences per batch (batch size)
UPDATE_FREQ=2            # Increase the batch size 2x

NUM_CLASSES=28474        # Number of GO terms in CAFA3

USER_DIR="/home/jlaw/projects/2020-01-deepgreen/fairseq-uniparc-fork/go_annotation_esm"
SAVE_DIR="/scratch/jlaw/deepgreen/fairseq-uniparc/2021-03-29-test/"
#DATA_DIR="/home/jlaw/projects/2020-01-deepgreen/fairseq-uniparc-fork/go_annotation/eagle/criterion_development/fairseq_swissprot_debug"
DATA_DIR="/projects/deepgreen/jlaw/swissprot_go_annotation/fairseq_cafa3"
#ROBERTA_PATH=""
ESM_PATH="/scratch/jlaw/torch/hub/checkpoints/esm1_t6_43M_UR50S.pt"

cmd="""fairseq-train $DATA_DIR \
    --user-dir=$USER_DIR \
    --restore-file=$ESM_PATH \
    --classification-head-name='go_prediction' \
    --task=sentence_labeling --criterion go_prediction --regression-target --num-classes=$NUM_CLASSES --init-token=0 \
    --arch=fairseq_esm --shorten-method=random_crop \
    --optimizer=adam --adam-betas='(0.9,0.98)' --adam-eps=1e-6 --clip-norm=0.0 \
    --lr-scheduler=polynomial_decay --lr=$PEAK_LR --warmup-updates=$WARMUP_UPDATES --total-num-update=$TOTAL_UPDATES \
    --weight-decay=0.01 \
    --validate-interval-updates=500 \
    --save-interval-updates=2000 --keep-interval-updates=1 \
    --batch-size=$MAX_SENTENCES --update-freq=$UPDATE_FREQ --save-dir=$SAVE_DIR \
    --max-update=$TOTAL_UPDATES --log-format=simple --log-interval=10 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --max-positions=$TOKENS_PER_SAMPLE \
    --dropout=0.1 --attention-dropout=0.1 \
    --find-unused-parameters
"""
    #--tensorboard-logdir=$MEMBERWORK/bie108/fairseq-tensorboard/$LSB_JOBNAME \

echo $cmd
$cmd
