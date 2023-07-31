#!bin/bash
trial=1
EXP_GROUP=$(date +%Y%m%d_%H%M%S)
acqf=EEIPU

EXP_NAME=$(date +%Y%m%d_%H%M%S)_$RANDOM
WANDB_MODE=online python cost-aware-bo/codeformer_hparams.py \
    --trial $trial \
    --exp-group $EXP_GROUP \
    --acqf $acqf \
    --exp-name $EXP_NAME