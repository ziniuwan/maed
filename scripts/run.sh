#!/bin/bash

### rand a 5 digit port number
function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))
}
export MASTER_PORT=$(rand 10000 20000)
echo "MASTER_PORT="$MASTER_PORT

export PYTHONPATH="./:$PYTHONPATH"
srun \
    --mpi=pmi2 \
    --partition=innova \
    --nodes=$1 \
    --ntasks-per-node=$2 \
    --gres=gpu:$2 \
    --kill-on-bad-exit=1 \
    python train.py --cfg=$3 --pretrained=$4
