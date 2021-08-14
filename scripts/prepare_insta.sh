#!/bin/bash
n=1
for ((i=0;i<$n;i++))
do
    srun \
        --job-name=insta_data \
        --kill-on-bad-exit=1 \
        python lib/data_utils/insta_utils_imgs.py --inp_dir ./data/insta_variety --n $n --i $i >log/$i.log 2>&1 &
done
