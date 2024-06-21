#!/bin/bash
#$ -l tmem=15.9G
#$ -l h_vmem=15.9G
#$ -l gpu=true
#$ -l gpu_type=(rtx3090|rtx4090|a6000|a40|a100|a100_80)
#$ -l hostname='!hogthrob*'
#$ -l h_rt=00:55:30
#$ -S /bin/bash
#$ -N poet
#$ -t 1
#$ -o /SAN/orengolab/nsp13/protein-model-steering/qsub_logs/
#$ -wd /SAN/orengolab/nsp13/protein-model-steering
#$ -j y
hostname
date
mkdir -p /SAN/orengolab/nsp13/protein-model-steering/qsub_logs/
source ~/.bashrc
conda deactivate
source /share/apps/source_files/python/python-3.9.5.source
cd /SAN/orengolab/nsp13/protein-model-steering
export POETDIR=/share/apps/genomics/PoET/
export PYTHONPATH=$PYTHONPATH:${POETDIR}
export PYTHONPATH=$PYTHONPATH:${POETDIR}/poet
export PYTHONPATH=$PYTHONPATH:/SAN/orengolab/nsp13/PETase
nvidia-smi
python3 example_inference.py \
  --ckpt_path ${POETDIR}/data/poet.ckpt
date