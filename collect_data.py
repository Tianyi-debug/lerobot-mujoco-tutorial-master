import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env import SimpleEnv
from mujoco_env.script_auto import ScriptPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset



# If you want to randomize the object positions, set this to None
# If you fix the seed, the object positions will be the same every time
SEED = 0 
# SEED = None <- Uncomment this line to randomize the object positions

REPO_NAME = 'omy_pnp'
NUM_DEMO = 5 # Number of demonstrations to collect
# ROOT = "./demo_data" # The root directory to save the demonstrations

import time
ROOT = f"./demo_data_{time.strftime('%Y%m%d_%H%M%S')}"


TASK_NAME = 'Put mug cup on the plate' 
xml_path = './asset/example_scene_y.xml'
# Define the environment
PnPEnv = SimpleEnv(xml_path, seed = SEED, state_type = 'joint_angle')



create_new = True
# if os.path.exists(ROOT):
#     print(f"Directory {ROOT} already exists.")
#     ans = input("Do you want to delete it? (y/n) ")
#     if ans == 'y':
#         import shutil
#         shutil.rmtree(ROOT)
#     else:
#         create_new = False


if create_new:
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root = ROOT, 
                robot_type="omy",
                use_videos=True,
                fps=20, # 20 frames per second
                features={
                    "observation.image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["state"], # x, y, z, roll, pitch, yaw
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"], # 6 joint angles and 1 gripper
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["obj_init"], # just the initial position of the object. Not used in training.
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
        )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

policy = ScriptPolicy()
policy.reset()

action = np.zeros(7)
episode_id = 0
record_flag = False # Start recording when the robot starts moving
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # check if the episode is done
        # --- 1) 用脚本策略产生 action（替换 teleop_robot）---
        action, episode_end = policy(PnPEnv)   # episode_end=True 表示脚本认为该 episode 结束

        # --- 2) 如果脚本结束：保存 episode -> reset env + reset policy -> 进入下一条 ---
        if episode_end:
            if record_flag:
                dataset.save_episode()
                episode_id += 1
            PnPEnv.reset(seed=SEED)
            policy.reset()
            dataset.clear_episode_buffer()
            record_flag = False
            action = np.zeros(7, dtype=np.float32)  # reset 当步不再动作

        if not record_flag and np.any(action != 0):
            record_flag = True
            print("Start recording")

        # Step the environment
        # Get the end-effector pose and images
        ee_pose = PnPEnv.get_ee_pose()
        agent_image,wrist_image = PnPEnv.grab_image()
        # # resize to 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        joint_q = PnPEnv.step(action)
        if record_flag:
            # Add the frame to the dataset
            dataset.add_frame( {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose, 
                    "action": joint_q,
                    "obj_init": PnPEnv.obj_init_pose,
                    # "task": TASK_NAME,
                }, task = TASK_NAME
            )
        PnPEnv.render(teleop=True)



PnPEnv.env.close_viewer()

# Clean up the images folder
import shutil
shutil.rmtree(dataset.root / 'images')