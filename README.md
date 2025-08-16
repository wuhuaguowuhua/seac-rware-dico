pkill -f train_new.py || true
python train_new.py \
  with env_name=rware-tiny-4ag-easy-v1 seed=0 dummy_vecenv=True \
       algorithm.num_processes=1 algorithm.num_steps=5 algorithm.device=cuda \
       algorithm.snd_coef=0.10 algorithm.seac_coef=1.0 \
       num_env_steps=5000000 log_interval=2000 save_interval=250000 eval_interval=0 \
       use_wandb=True wandb_project=seac-rware-dico
