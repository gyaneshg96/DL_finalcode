#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --partition=n1s16-t4-2
#SBATCH --account=dl11
#SBATCH --time=24:00:00
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --exclusive
#SBATCH --requeue

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

cd $HOME/code_11

singularity exec --nv \
--bind /scratch \
--overlay $SCRATCH/conda.ext3:rw \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python moco-master/main_moco.py -a resnet50  --lr 0.06 --dist-url 'tcp://localhost:10004' --batch-size 512 --multiprocessing-distributed --world-size 1 --rank 0 -data /dataset --mlp --moco-t 0.2 --aug-plus --cos  --workers 4 --epochs 300
"
