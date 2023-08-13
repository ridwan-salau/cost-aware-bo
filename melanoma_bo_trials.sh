#!/bin/bash
set -e 

EXP_GROUP=HPCLAB
ACQF_ARRAY=("EEIPU" "EI")
MODE="${1:-online}"

# Function to execute the Python script
run_python_script() {
    local GPU_IP=$1
    local EXP_NAME=$(date +%Y%m%d_%H%M%S)_$RANDOM
    local DIR=/home/ridwan.salahuddeen/Documents/research/cost-aware-bo # Modify this to point to your experiment dir
    ssh "$GPU_IP" "cd $DIR \
        && conda activate melanoma \
        && WANDB_MODE=$MODE \
        python3 ./melanoma_hparams.py \
            --trial $trial \
            --exp-group $EXP_GROUP \
            --acqf $acqf \
            --exp-name $EXP_NAME"
    sleep 2
}

IN_USE_MACHINES=()
IN_USE_MACHINES_LOCK=/tmp/in_use_machines.lock
# prev_used() {
#     local IN_USE_MACHINES=$1
#     echo Previously used array $IN_USE_MACHINES >&2
#     local found=false
#     for element in "${IN_USE_MACHINES[@]}"; do
#         if [ "$element" == "$ip" ]; then
#             found=true
#             break
#         fi
#     done
#     echo "$found"
# }

prev_used() {
    local ip="$1"
    local IN_USE_MACHINES="$2"
    for element in "${IN_USE_MACHINES[@]}"; do
        if [ "$element" == "$ip" ]; then
            return 0  # Found
        fi
    done
    return 1  # Not found
}

is_machine_free() {
    GPU_IP=$1
    echo Checking if $GPU_IP is available  >&2
    if echo $(ssh $GPU_IP pgrep -a python) | grep -q "melanoma.py"; then
        # echo $(ssh $GPU_IP pgrep -a python) >&2
        return 1
    else
        echo $GPU_IP is NOT available >&2
        return 0
    fi
}

get_free_gpu() {
    local min_memory=$1
    local end=50
    local c=11
    local IN_USE_MACHINES=$2
    while [ "$c" -lt "$end" ]; do
        local ip="10.127.30.$c"
        echo "Checking machine $ip..." >&2
        gpu=$(ssh -oBatchMode=yes -o ConnectTimeout=5 "$ip" nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

        echo "IN_USE_MACHINES contents 2:" >&2
        for element in "${IN_USE_MACHINES[@]}"; do
            echo - "$element" >&2
        done
        if [ $? -eq 0 ] && [ "$gpu" -gt "$min_memory" ]; then
            if ! prev_used "$ip" "$IN_USE_MACHINES" && is_machine_free "$ip" ; then
                echo "Found a large GPU: $ip" >&2
                echo "$ip"
                break
            fi

        fi

        echo "GPU $ip has $gpu Mb available; Expecting $min_memory" >&2

        if [ "$c" -eq $((end-1)) ]; then
            echo "starting search for free GPU all over" >&2
            c=11
        fi
        ((c++))
    done
}



# Function to atomically update IN_USE_MACHINES array
update_in_use_machines() {
    local new_ip="$1"
    echo "Adding $new_ip to used machines array" 
    flock "$IN_USE_MACHINES_LOCK" -c "echo $new_ip >> IN_USE_MACHINES"
    echo After adding ... >&2
    for element in "${IN_USE_MACHINES[@]}"; do
        echo - "$element"  >&2
    done
}

min_required_memory=20000

trial=1
gpu_ip="${2:-gpu-14}"
while [ "$trial" -le 10 ]; do
    for acqf in "${ACQF_ARRAY[@]}"; do
        # echo Before getting the gpu_ip ...
        for element in "${IN_USE_MACHINES[@]}"; do
            echo - "$element" 
        done
        # gpu_ip=$(get_free_gpu "$min_required_memory" "$IN_USE_MACHINES")
        

        if [ -n "$gpu_ip" ]; then
            # update_in_use_machines "$gpu_ip"
            # IN_USE_MACHINES+=("$gpu_ip")
            run_python_script "$gpu_ip" 
            # sleep 4
        # else
        #     echo "No available GPU found. Waiting..."
        #     sleep 10
        fi
    done

    ((trial++))
done

wait  # Wait for all remaining background processes to finish
# find ~/Documents/research/input/preprocessed/ -type f -print0 | xargs -0 -n1 -P20 rm
