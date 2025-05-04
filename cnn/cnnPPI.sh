#!/bin/bash
echo 'works'
source ~/anaconda3/etc/profile.d/conda.sh
#rm logger.txt
#rm /data/logs/log.txt
conda activate pytorch_p38
#source  activate /home/ubuntu/anaconda3/envs/aws_neuron_pytorch_p36
python cnnPPI_simplev1.py
