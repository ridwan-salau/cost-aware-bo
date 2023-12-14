#! /bin/bash
set -e

mkdir
ACQF_ARRAY=(EEIPU EI CArBO EIPS MS_CArBO)
exp_name=t5-pipeline-multi-tuun

c=0
for trial in {1..5}; do
    for acqf in ${ACQF_ARRAY[@]}; do
        log_dir=log/$exp_name
        mkdir -p $log_dir
        data_dir="${2:-./inputs}"
        # ((c+=1))
        cache_root=.cachestore/${acqf}/${RANDOM}_trial_${trial}
        python optimize_multi.py \
            --exp-name $exp_name --trial $trial --cache-root \
            $cache_root --acqf $acqf --data-dir $data_dir &>> ${log_dir}.log & rm -rf $cache_root

        # if [ $(($c%1)) -eq 0 ]; then
        #     wait # Wait for the inner loop to complete before continuing
        # fi
    # done

done

# wait