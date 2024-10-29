#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS, 64 requests whole node
#SBATCH --partition=ghx4    # <- one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcjw-dtai-gh    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=deepseek-a512e16-freq-int3symm-iter20
### GPU options ###
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -e results/%x-%j.err
#SBATCH -o results/%x-%j.out

module load cuda
source /work/hdd/bcjw/yyuan6/miniconda3-gh/bin/activate
conda activate env310
srun python3 ../deepseek-performance.py --iteration 20 --dense_rank 512 --exp_rank 16 \
    --model_path /work/hdd/bcjw/yyuan6/hqq_lorc/deepseek-moe/deepseek-3bit-a512e16-freq-int3symm \
    --error_path /work/hdd/bcjw/yyuan6/hqq_lorc/deepseek-moe/a512e16-int3symm

srun python3 eval_script_a1024efreqadapt-32-int3symm.py --dense_rank 512 --exp_rank 16 \
    --model_path /work/hdd/bcjw/yyuan6/hqq_lorc/deepseek-moe/deepseek-3bit-a512e16-freq-int3symm-iter20 \
    --Error_path /work/hdd/bcjw/yyuan6/hqq_lorc/deepseek-moe/a512e16-int3symm-iter20 \
    --LoRC_dtype int3_symm
