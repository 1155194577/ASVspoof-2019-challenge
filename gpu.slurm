#!/bin/bash
#SBATCH -J asvspoof
#SBATCH -o slurm-%j.out
#SBATCH -c 6
#SBATCH --mem=8G
#SBATCH -t 0:30:00
#SBATCH -p gpu_2h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#
# CUHK CSE GPGPU cluster -- submit from the linux1 login node, preferably
# through ./submit.sh which syncs the code and features/ for you.
#
#   sbatch gpu.slurm train.py    --epochs 10
#   sbatch gpu.slurm evaluate.py --split validation
#   ./submit.sh --follow --epochs 10
#
# Everything runs inside the PyTorch container, exactly like the other projects
# on this cluster: no pip, no venv, no home-quota problems. The node only needs
# numpy + torch, and features/ holds the pre-extracted vectors, so the dataset
# never has to travel.
#
# The image lives in the node's /dev/shm, so a cold node re-pulls it (~5 min,
# once). To pin a node whose cache you know about, add e.g.
#     #SBATCH -w gpu56
# (projgpu33 is no longer part of gpu_2h).
# --cpus-per-task must stay <= 6 per GPU, otherwise Slurm demands
# --constraint=highcpucount.
#
# Overrides: IMAGE (path to .sif), ENGINE (singularity/apptainer), PROJ.

set -euo pipefail

PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJ" || { echo "cannot cd to PROJ=$PROJ" >&2; exit 1; }

TARGET="${1:-train.py}"
if [ $# -gt 0 ]; then shift; fi

IMAGE="${IMAGE:-/dev/shm/pytorch-2.5.1.sif}"
URI="docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"

echo "job ${SLURM_JOB_ID:-?} on $(hostname) partition=${SLURM_JOB_PARTITION:-?} $(date -Is)"
echo "project : $PROJ"
echo "command : $TARGET $*"

if [ ! -d features ]; then
    echo "warning: no features/ in $PROJ -- did ./submit.sh sync it?" >&2
fi

ENGINE="${ENGINE:-$(command -v singularity || command -v apptainer || true)}"
if [ -z "$ENGINE" ]; then
    echo "no singularity/apptainer on this node; set ENGINE=/path/to/singularity" >&2
    exit 1
fi

if [ ! -f "$IMAGE" ]; then
    echo "image missing on $(hostname), pulling $URI (~3 GB, a few minutes)"
    export SINGULARITY_CACHEDIR=/dev/shm/sgc SINGULARITY_TMPDIR=/dev/shm/stmp
    export APPTAINER_CACHEDIR=/dev/shm/sgc APPTAINER_TMPDIR=/dev/shm/stmp
    mkdir -p /dev/shm/sgc /dev/shm/stmp
    "$ENGINE" pull --force "$IMAGE" "$URI"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

"$ENGINE" exec --nv "$IMAGE" python -c \
    "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

exec "$ENGINE" exec --nv "$IMAGE" python -u "$TARGET" "$@"
