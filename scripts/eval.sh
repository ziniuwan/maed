export PYTHONPATH="./:$PYTHONPATH"
srun \
    --partition=innova \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:1 \
        python eval.py --cfg $1 --pretrained $2 --eval_ds $3 --eval_set $4
