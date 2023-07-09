# /bin/bash

set -e

SCRIPT=$(realpath "$0")
TUUN_DIR=$(dirname "$SCRIPT")

TUUN_CONDA_ENV=/home/ridwan/miniconda3/envs/tuun/bin/

OUTPUT_DIR=/home/ridwan/workdir/cost_aware_bo/outputs
COST_FILE=elapsed_time.json

echo Clearing the cache directory
rm -rf /home/ridwan/workdir/cost_aware_bo/cache_dir/*

LOG_DIR=$(date +%Y%m%d_%H%M%S)
trial=0
EXP_GROUP=$(date +%Y%m%d_%H%M%S)

STAGE_COSTS=("$OUTPUT_DIR/stage_1_$COST_FILE" \
            "$OUTPUT_DIR/stage_2_$COST_FILE" \
            "$OUTPUT_DIR/stage_3_$COST_FILE")
OBJ_OUT=$OUTPUT_DIR/score.json

echo Clearing output files... ${STAGE_COSTS[*]} $OBJ_OUT
rm -f ${STAGE_COSTS[*]} $OBJ_OUT

echo Running optimizer
# sleep 2; # Otherwise, 
ACQF_ARRAY=("EEIPU" "EI")
EXP_GROUP="20230516_212011"

for trial in {11..20}; do
    for acqf in ${ACQF_ARRAY[@]}; do
        rm -f ${STAGE_COSTS[*]} $OBJ_OUT;
        rm -rf /home/ridwan/workdir/cost_aware_bo/mycachestore/*
        python $TUUN_DIR/param_gen.py \
            --trial $trial \
            --stage_costs_outputs ${STAGE_COSTS[*]} \
            --obj-output  $OBJ_OUT \
            --exp-group $EXP_GROUP \
            --acqf $acqf \
            --base-dir $TUUN_DIR;
    done;
done;
