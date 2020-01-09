#!/bin/bash

# Usage: `scripts/run.sh MODEL DATA UNIQUE_ID`
# where
# MODEL = cyclegan | cycleflow | flow2flow
# DATA = facades | cityscapes | maps | edges2shoes | edges2handbags | night2day
#      | horse2zebra | apple2orange | summer2winter_yosemite
#      | {monet,vangogh,ukiyoe,cezanne}2photo | iphone2dslr_flower
# First line in DATA are paired datasets, second and third lines are unpaired
# All datasets must be downloaded with `scripts/download.sh DATA` first

if [[ $1 == cyclegan ]]; then

    python train.py \
        --name $2_CycleGAN_$3 \
        --model CycleGAN \
        --crop_shape 128,128 \
        --resize_shape 144,144 \
        --data_dir data/$2 \
        --gpu_ids 0,1,2,3 \
        --batch_size 32 \
        --iters_per_print 32 \
        --iters_per_visual 640 \
        --norm_type instance \
        --num_epochs=200 \
        --use_dropout False \
        --use_mixer True

elif [[ $1 == cycleflow ]]; then

    python train.py \
        --name $2_CycleFlow_$3 \
        --model CycleFlow \
        --crop_shape 128,128 \
        --resize_shape 144,144 \
        --data_dir data/$2 \
        --gpu_ids 0,1,2,3 \
        --batch_size 16 \
        --iters_per_print 16 \
        --iters_per_visual 320 \
        --norm_type instance \
        --num_blocks 4 \
        --num_channels_g 64 \
        --num_epochs 200 \
        --num_scales 2 \
        --use_dropout False \
        --use_mixer True

elif [[ $1 == flow2flow ]]; then

    python train.py \
        --name $2_Flow2Flow_$3 \
        --model Flow2Flow \
        --crop_shape 128,128 \
        --resize_shape 144,144 \
        --data_dir data/$2 \
        --gpu_ids 0,1,2,3 \
        --batch_size 16 \
        --iters_per_print 16 \
        --iters_per_visual 320 \
        --norm_type instance \
        --num_blocks 4 \
        --num_channels_g 32 \
        --num_epochs 200 \
        --num_scales 2 \
        --use_dropout False \
        --use_mixer True \
        --lambda_mle 1e-4

else

    echo "Invalid model name:" $1
    echo "Options: cyclegan, cycleflow, flow2flow"

fi
