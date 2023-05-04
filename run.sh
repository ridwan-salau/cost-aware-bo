# !/bin/bash

exp_group=$(date '+%Y%m%d-%H%M')
ACQF_ARRAY=("EEIPU")

for acqf in ${ACQF_ARRAY[@]}; do
    for trial in {1..1}; do
        /home/ridwan/miniconda3/envs/tuun/bin/python main.py --trial-num $trial --exp-group $exp_group --acqf $acqf &
    done;
done;

wait