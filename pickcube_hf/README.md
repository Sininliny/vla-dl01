---
pipeline_tag: robotics
tags:
- robotics
- manipulation
- smolvla
- lerobot
library_name: lerobot
---

# PickCube-v1 Dataset

Robot manipulation dataset for pick and place tasks, converted for SmolVLA training.

## Dataset Info
- Episodes: 5
- Frames: 250  
- Image size: 256x256
- State dim: 6
- Action dim: 6

## Usage

```bash
python -m lerobot.scripts.train \
    --dataset.repo_id=dancher00/pickcube-v1-smolvla \
    --policy.path=lerobot/smolvla_base \
    --batch_size=4 \
    --steps=1000
```
