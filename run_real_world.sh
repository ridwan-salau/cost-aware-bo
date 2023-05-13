# /bin/bash

set -e

SCRIPT=$(realpath "$0")
TUUN_DIR=$(dirname "$SCRIPT")

TUUN_CONDA_ENV=/home/ridwan/miniconda3/envs/tuun/bin/
CONFIG_FILE=/home/ridwan/workdir/pirlib/examples/stacking/argo-stacking-pipeline.yml

OUTPUT_DIR=/home/ridwan/workdir/pirlib/examples/stacking/outputs
COST_FILE=elapsed_time.json

echo Clearing the cache directory
rm -rf /home/ridwan/workdir/pirlib/examples/stacking/cache_dir/*

LOG_DIR=$(date +%Y%m%d_%H%M%S)
trial=0

STAGE_COSTS=("$OUTPUT_DIR/build_basemodels1/return/$COST_FILE" \
            "$OUTPUT_DIR/build_basemodels2/return/$COST_FILE" \
            "$OUTPUT_DIR/build_metamodel/return/$COST_FILE")
OBJ_OUT=$OUTPUT_DIR/return/score.json

echo Clearing output files... ${STAGE_COSTS[*]} $OBJ_OUT
rm -f ${STAGE_COSTS[*]} $OBJ_OUT

echo Running optimizer
# sleep 2; # Otherwise, 

# ssh gpu-14 "$TUUN_CONDA_ENV"
python $TUUN_DIR/param_gen_v2.py \
    --trial 1 \
    --stage_costs_outputs ${STAGE_COSTS[*]} \
    --obj-output  $OBJ_OUT \
    --config-file $CONFIG_FILE \
    --base-dir $TUUN_DIR;

#     if [ $i -ne -1 ];
#     then
#         argo submit -n gpu-14 --wait $CONFIG_FILE;
#     fi
#     sleep 5;
#  done 