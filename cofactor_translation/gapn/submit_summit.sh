#!/bin/bash

#BSUB -P BIE108
#BSUB -q batch-hm
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -J 2021_10_14_gapn
#BSUB -o /gpfs/alpine/scratch/jlaw/bie108/gapn/2021-10-gapn
#BSUB -e /gpfs/alpine/scratch/jlaw/bie108/gapn/2021-10-gapn
#BSUB -alloc_flags NVME
#BSUB -B

# had to use this series of module loads:
module purge
module load gcc
module load spectrum-mpi
#module load open-ce #
#conda activate fairseq-open-ce
module load open-ce/1.1.3-py37-0
module load cuda/10.2.89
conda activate /autofs/nccs-svm1_proj/bie108/jlaw/envs/fairseq-open-ce-1.1.3-py37-0
echo `which python`

export OMP_NUM_THREADS=4

echo "`date` Running gapn.py"

# For more details about jsrun, see: 
# https://docs.olcf.ornl.gov/systems/summit_user_guide.html#id17
jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 42 --gpu_per_rs 6 -r 1 -b none \
    python -u gapn.py

echo "`date` Finished "
