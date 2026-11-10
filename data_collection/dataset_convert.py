
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC = Path(
    "/home/dl01e26/DATA/vla-dl01/data/pick_cube_demos/trajectory_dataset.h5"
)
ROOT = Path(
    "/home/dl01e26/DATA/vla-dl01/data/lerobot_pick_cube"
)
REPO_ID = "local/pick_cube_rebalanced"

# ManiSkill's default 20 Hz。
FPS = 20

TASK = "Pick up the red cube and place it at the goal location."


def to_hwc(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    if image.shape[-1] == 3:
        pass
    elif image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    else:
        raise ValueError(f"Cannot identify RGB channels: {image.shape}")

    if image.dtype != np.uint8:
        if image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


if not SRC.is_file():
    raise FileNotFoundError(f"Missing source dataset: {SRC}")

if ROOT.exists() and any(ROOT.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {ROOT}\n"
        "Inspect it first; do not overwrite blindly."
    )

with h5py.File(SRC, "r") as h5:
    if "data" not in h5:
        raise ValueError("HDF5 is missing the expected 'data' group.")

    episodes = sorted(
        [k for k in h5["data"].keys() if k.startswith("episode_")],
        key=lambda k: int(k.split("_")[-1]),
    )

    if not episodes:
        raise ValueError("No episodes found in the HDF5 dataset.")

    first = h5["data"][episodes[0]]

    sample_image = to_hwc(first["observation/image"][0])
    state_dim = first["observation/state"].shape[-1]
    action_dim = first["action"].shape[-1]

    print("Source:", SRC)
    print("Episodes:", len(episodes))
    print("Image shape:", sample_image.shape)
    print("State dimension:", state_dim)
    print("Action dimension:", action_dim)

    features = {
        "observation.image": {
            "dtype": "image",
            "shape": sample_image.shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=ROOT,
        fps=FPS,
        robot_type="panda",
        features=features,
        use_videos=False,
    )

    total_frames = 0

    try:
        for ep_name in episodes:
            ep = h5["data"][ep_name]

            images = ep["observation/image"]
            states = ep["observation/state"]
            actions = ep["action"]

            n = len(images)

            if n == 0:
                print(f"Skipping empty episode: {ep_name}")
                continue

            if not (n == len(states) == len(actions)):
                raise ValueError(
                    f"{ep_name}: image/state/action lengths differ"
                )

            for i in range(n):
                state = np.asarray(states[i], dtype=np.float32)
                action = np.asarray(actions[i], dtype=np.float32)
                image = to_hwc(images[i])

                if state.shape != (state_dim,):
                    raise ValueError(
                        f"{ep_name}, frame {i}: bad state shape {state.shape}"
                    )

                if action.shape != (action_dim,):
                    raise ValueError(
                        f"{ep_name}, frame {i}: bad action shape {action.shape}"
                    )

                dataset.add_frame({
                    "observation.image": image,
                    "observation.state": state,
                    "action": action,
                    "task": TASK,
                })

            dataset.save_episode()
            total_frames += n

            print(f"Converted {ep_name}: {n} frames")

    finally:
        dataset.finalize()

print("\nConversion complete.")
print("Output:", ROOT)
print("Episodes:", len(episodes))
print("Frames:", total_frames)

