cd /home/dl01e26/DATA/vla-dl01
conda activate vla

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/lerobot_pick_cube \
  --dataset.root=data/lerobot_pick_cube \
  '--rename_map={"observation.image": "observation.images.camera1"}' \
  --output_dir=/home/dl01e26/DATA/vla-dl01/checkpoints/smolvla_pickcube_60ep_8k \
  --batch_size=4 \
  --optimizer.lr=1e-4 \
  --steps=8000 \
  --save_freq=1000 \
  --log_freq=20 \
  --wandb.enable=false