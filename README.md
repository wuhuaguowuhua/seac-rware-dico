env \
  WANDB_API_KEY="..." \
  WANDB_ENTITY="wuhuaguowuhua-university-of-exeter" \
  WANDB_PROJECT="seac-rware-dico-slow" \
  WANDB_DIR="$PWD/results/wandb" \
  python train_new.py \
    with env_name=rware-tiny-4ag-easy-v1 seed=0 dummy_vecenv=False \
         device=cuda use_wandb=True wandb_project=seac-rware \
         wandb_run_name="tiny4ag-longrun-w2snd-$(date +%m%d_%H%M)" \
         algorithm.num_processes=8 algorithm.num_steps=128 \
         algorithm.snd_metric=w2 algorithm.snd_coef=0.10 algorithm.seac_coef=1.0 \
         algorithm.snd_warmup_updates=5000 algorithm.seac_warmup_updates=5000 \
         num_env_steps=5000000 log_interval=50 save_interval=200000 eval_interval=0
