#! /bin/bash
set -e

ACQF_ARRAY=(EEIPU CArBO MS_BO MS_CArBO EIPS EI) #LaMBO

c=0
for acqf in ${ACQF_ARRAY[@]}; do
    for trial in {1..2}; do
        ((c+=1))
        cache_root=.cachestore/${acqf}/${RANDOM}_trial_${trial}
        python modelling.py \
            --exp-name 2-stg-stacking-new --trial $trial --cache-root \
            $cache_root --acqf $acqf && rm -rf $cache_root &

        if [ $(($c%2)) -eq 0 ]; then
            wait # Wait for the inner loop to complete before continuing
        fi
    done

done

wait