#!/bin/bash
# IMPORTANT: Due to size restriction (<100MB), we did not attach the checkpoints corresponding to
# blackbox attacks (ann_checkpoint2.pt, snnbp_checkpoint2.pt) here. 
#CIFAR100 checkpoints are also not attached
#**********************************************
# Whitebox==> type="bb", Blackbox==> type="wb"
# source = "ann" or "snnconv" or "snnbp"
# target = "ann" or "snnconv" or "snnbp"
#**********************************************
type="wb"
source="ann"
target="ann"
filename="evaluate_attacks.py"
#FGSM attack
#*************
CUDA_VISIBLE_DEVICES=0 python $filename --arch 'VGG5' --dataset 'CIFAR10' \
--attack 'fgsm' --type $type --source $source --target $target --epsilon 8 --batch_size 4

#PGD attack
#************
#CUDA_VISIBLE_DEVICES=0 python $filename --arch 'VGG5' --dataset 'CIFAR10' \
#--attack 'pgd' --type $type --source $source --target $target --epsilon 8 \
#--eps_iter 2 --pgd_steps 7 


