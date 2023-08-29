CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --master_port=${MASTER_PORT:-$RANDOM} \
    --nproc_per_node=2 trainer/train.py \
    --lr 1e-5 --epochs 150 --batch_size 90 \
    --num_hproxies 512 --clip_r 2.3 \
    --warmup_epochs 5 \
    --weight_decay 1e-4 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 5 \
    --topk 20 --mrg 0.1 \
    --dataset SOP --model deit_small_distilled_patch16_224 \
    --IPC 2 --loss seeproxy --emb 128  --use_lastnorm True \
    --optimizer adamw --fc_lr_scale 1e2 \
	--use-schedule True --max-thresh 0.4  --min-thresh 0.0  --lambd 0.2  --type 1 --n-expan 8\
	--run_name single_gpu 

