#! /bin/bash
set -e

ACQF_ARRAY=(EEIPU EI CArBO EIPS)

python_path=/home/ridwan/miniconda3/envs/t5env/bin/python

c=0
for trial in {1..1}; do
    # for acqf in ${ACQF_ARRAY[@]}; do
        acqf="EEIPU"
        # ((c+=1))
        cache_root=.cachestore/${acqf}/${RANDOM}_trial_${trial}
        $python_path optimize.py \
            --exp-name t5-pipeline-tuun --trial $trial --cache-root \
            $cache_root --acqf $acqf && rm -rf $cache_root

        # if [ $(($c%1)) -eq 0 ]; then
        #     wait # Wait for the inner loop to complete before continuing
        # fi
    # done

done

# wait