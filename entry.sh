#!/usr/bin/env bash

# echo "Initialize conda"
# conda init bash
# source $HOME/.bashrc

echo "Go Home"
cd $HOME/PLMBench

# echo "Activate PLM environment"
# conda activate plm

SCRATCH="/scratch/chair_kalinina/s8rojoer"

echo "Run training"
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/sample_10k.txt --output-dir $SCRATCH/profile --model-name prof

