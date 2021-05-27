#module purge
#module load gcc
#module load spectrum-mpi
#module load open-ce #
#conda activate fairseq-open-ce
#module load open-ce/1.1.3-py37-0
#module load cuda/10.2.89
#conda activate /autofs/nccs-svm1_proj/bie108/jlaw/envs/fairseq-open-ce-1.1.3-py37-0

# make sure the same versions are being used
#python -c "import torch; print(f'torch: {torch.__version__}');"
#python -c "import fairseq; print(f'fairseq: {fairseq.__version__}');"
#python -c "import numpy; print(f'numpy: {numpy.__version__}');"


BASE_DIR="$(readlink -e ../../../)"
OBO_FILE="$BASE_DIR/inputs/2020-12-swissprot/go-basic.obo.gz"
RESTRICT_TERMS_FILE="$BASE_DIR/inputs/2020-12-swissprot/terms.csv.gz"
# Number of GO terms
NUM_CLASSES="$(gzip -cd $RESTRICT_TERMS_FILE | wc -l)"

fairseq-train ../../eagle/criterion_development/fairseq_swissprot_debug/ \
  --task sentence_labeling \
  --user-dir ../../../go_annotation/ \
  --criterion go_prediction \
  --obo-file="${OBO_FILE}" --restrict-terms-file="${RESTRICT_TERMS_FILE}" \
  --regression-target \
  --classification-head-name='go_prediction' \
  --num-classes $NUM_CLASSES --init-token 0 \
  --batch-size 1 \
  --update-freq 4 \
  --total-num-update 10 \
  --shorten-method='random_crop' \
  --reset-optimizer --reset-dataloader --reset-meters \
  --weight-decay 0.1 \
  --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
  --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr 1e-05 \
  --arch esm1_t6 \
  --max-positions 512 \
  --save-interval 1 \
  --log-format simple --log-interval 1 \
  --fp16 --memory-efficient-fp16 \
  --ddp-backend=legacy_ddp

  #--batch-size 2 \
  #--arch esm1b_t33 \
  #--arch esm1_t6 \
