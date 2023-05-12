# kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
exp_group=$(date '+%Y%m%d-%H%M')
ACQF_ARRAY=("EIPU" "EEIPU" "EIPU-MEMO")
DEC_FACS=(2)
INIT_ETAS=(0.95)

for eta in ${INIT_ETAS[@]}; do
    for dec in ${DEC_FACS[@]}; do
        for acqf in ${ACQF_ARRAY[@]}; do
            for trial in {1..5}; do
                python main.py --trial-num $trial --exp-group $exp_group --acqf $acqf --init-eta $eta --decay-factor $dec &
            done;
        done;
    done;
done;

wait