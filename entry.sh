#!/usr/bin/env bash

# echo "Initialize conda"
# conda init bash
# source $HOME/.bashrc

echo "Go Home"
cd $HOME/PLMBench

# echo "Activate PLM environment"
# conda activate plm

SCRATCH="/scratch/chair_kalinina/s8rojoer/PLM"

echo "Run training"
# conda run -n plm --no-capture-output python embed.py --model-path $SCRATCH/train_11_3M/esm_11_3M/esm2_t6_final_model/ --data-path $SCRATCH/meltome_atlas.csv --output-path $SCRATCH/embeddings/meltome/esm_11_3M/
conda run -n plm --no-capture-output python train_claude.py --data-file $SCRATCH/train_files/train_11_3M.txt --output-dir $SCRATCH/models --model-name esm_t30_11_3M

