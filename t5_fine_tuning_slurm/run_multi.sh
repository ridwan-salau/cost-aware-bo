#! /bin/bash

set -e


BASE_DIR=/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Ridwan.Salahuddeen@mbzuai.ac.ae/cost_aware_bo/
BASE_DIR=./
mkdir -p $BASE_DIR
acqf=EEIPU
trial=1
DATE_NOW=`date +"%Y-%m-%d-%H%M"`
run_trial() {
    log_file=$BASE_DIR/log/$acqf/$exp_name"_trial_"$trial.log
    data_dir=$BASE_DIR/inputs

    cache_root=$BASE_DIR/.cachestore/${acqf}/${RANDOM}_trial_${trial} 
    # gpu_id=$((target_dev%max_concurrent_executions))
    # CUDA_VISIBLE_DEVICES=$gpu_id taskset --cpu-list $((60*gpu_id))-$((60*(gpu_id+1))) \
    python optimize_multi.py \
        --date-now $DATE_NOW --exp-name $exp_name --trial $trial --cache-root \
        $cache_root --acqf $acqf --data-dir $data_dir 2>&1 | tee ${log_file} 
    rm -rf $cache_root 

}

ACQF_ARRAY=(EEIPU CArBO LaMBO MS_BO EIPS EI MS_CArBO)
exp_name=t5-pipe-multi-new

mkdir -p $BASE_DIR/log/{EEIPU,EI,CArBO,EIPS,MS_CArBO,MS_BO,LaMBO}

run_trial
# for acqf in ${ACQF_ARRAY[@]}; do
#     for trial in {1..5}; do
#         # Execute your function in the background
#         run_trial &
#         ((target_dev+=1))

#         # Track the background processes
#         background_processes+=($!)

#         # If the number of background processes reaches the maximum, wait for them to finish
#         if (( ${#background_processes[@]} == max_concurrent_executions )); then
#             wait "${background_processes[@]}"
#             background_processes=()
#         fi
#     done
# done
