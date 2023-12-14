#! /bin/bash
set -e

mkdir
ACQF_ARRAY=(EEIPU EI CArBO EIPS MS_CArBO)
exp_name=t5-pipeline-multi-tuun

mkdir -p log/{EEIPU,EI,CArBO,EIPS,MS_CArBO}
num_gpus=$(nvidia-smi --list-gpus | wc -l)
c=0
for trial in {1..5}; do
    for acqf in ${ACQF_ARRAY[@]}; do
        log_dir=log/$exp_name && \
        data_dir="${2:-./inputs}" && \
        # ((c+=1))
        cache_root=.cachestore/${acqf}/${RANDOM}_trial_${trial} && \
        python optimize_multi.py \
            --exp-name $exp_name --trial $trial --cache-root \
            $cache_root --acqf $acqf --data-dir $data_dir &>> ${log_dir}.log && rm -rf $cache_root

        if [ $(($c%$num_gpus)) -eq 0 ]; then
            wait # Wait for the inner loop to complete before continuing
        fi
    done

done

# wait