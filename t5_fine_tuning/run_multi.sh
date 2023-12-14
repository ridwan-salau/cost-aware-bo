#! /bin/bash
set -e

ACQF_ARRAY=(EEIPU EI CArBO EIPS MS_CArBO)
exp_name=t5-pipeline-multi-tuun

mkdir -p log/{EEIPU,EI,CArBO,EIPS,MS_CArBO}
num_gpus=$(nvidia-smi --list-gpus | wc -l)
target_dev=0
for acqf in ${ACQF_ARRAY[@]}; do
    for trial in {1..5}; do
        log_file=log/$acqf/$exp_name"_trial_"$trial && \
        data_dir="${2:-./inputs}" && \
        ((target_dev+=1))
        cache_root=.cachestore/${acqf}/${RANDOM}_trial_${trial} && \
        CUDA_VISIBLE_DEVICES=$target_dev python optimize_multi.py \
            --exp-name $exp_name --trial $trial --cache-root \
            $cache_root --acqf $acqf --data-dir $data_dir 2>&1 | tee -a ${log_file}.log && rm -rf $cache_root &

        if [ $(($target_dev%$num_gpus)) -eq 0 ]; then
            wait # Wait for the inner loop to complete before continuing
        fi
    done

done

# wait