#! /bin/bash
set -e

run_trial() {
    # Your function code goes here
    # For example: sleep 5; echo "Function completed"
    log_file=log/$acqf/$exp_name"_trial_"$trial.log
    data_dir="${2:-./inputs}" 

    cache_root=.cachestore/${acqf}/${RANDOM}_trial_${trial} 
    gpu_id=$((target_dev%max_concurrent_executions))
    CUDA_VISIBLE_DEVICES=$gpu_id taskset --cpu-list $((60*gpu_id))-$((60*(gpu_id+1))) \
    python optimize.py \
        --exp-name $exp_name --trial $trial --cache-root \
        $cache_root --acqf $acqf --data-dir $data_dir 2>&1 | tee ${log_file} 
    rm -rf $cache_root 

}

ACQF_ARRAY=(EEIPU CArBO LaMBO MS_BO MS_CArBO EIPS EI) #LaMBO
exp_name=segment-exp-new
data_dir=mbz-hpc-aws-master/AROARU6TOWKRU3FNVE2PB:Ridwan.Salahuddeen@mbzuai.ac.ae/segmentation/inputs

mkdir -p log/{EEIPU,EI,CArBO,EIPS,MS_CArBO,MS_BO,LaMBO}
max_concurrent_executions=$(nvidia-smi --list-gpus | wc -l)
target_dev=0
for acqf in ${ACQF_ARRAY[@]}; do
    for trial in {1..5}; do
        # Execute your function in the background
        run_trial &
        ((target_dev+=1)) 

        # Track the background processes
        background_processes+=($!)

        # If the number of background processes reaches the maximum, wait for them to finish
        if (( ${#background_processes[@]} == max_concurrent_executions )); then
            wait "${background_processes[@]}"
            background_processes=()
        fi
    done
done

# Wait for any remaining background processes to finish
wait "${background_processes[@]}"
