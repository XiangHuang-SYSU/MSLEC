#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=4381 MSRestoreX/train.py -opt options/train_VAREC_S1_lcdp.yml --launcher pytorch 

