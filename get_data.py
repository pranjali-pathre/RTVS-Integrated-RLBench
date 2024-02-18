import numpy as np
from PIL import Image
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import OpenDoor
from rlbench.tasks import PlaceShapeInShapeSorter
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a data generation script for InstructPix2Pix.")
    parser.add_argument(
        "--num",
        type=str,
        default="0",
        required=True,
        help="variation num",
    )
    args = parser.parse_args()
    return args

live_demos = False
DATASET = '' if live_demos else '../'

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(False, 'world'), gripper_action_mode=Discrete())

env = Environment(
    action_mode, DATASET, obs_config, headless=False, static_positions=True)
env.launch()

task = env.get_task(PlaceShapeInShapeSorter)
task.set_variation(0)
demos = task.get_demos(1, live_demos=live_demos)

training_steps = 120
episode_length = 120
obs = None
for demo in demos:
    task.reset()
    for obs in demo:
        # vel = obs.joint_velocities
        # pos = obs.joint_positions
        # print("Current joint vel: ", vel, "\nCurrent joint pos: ", pos)
        action = [0, 0, -0.1, 0, 0, 0, 1.0, 1.0]
        obs, reward, terminate = task.step(action)
        # print("New joint vel: ", obs.joint_positions, "New joint pos: ", obs.joint_positions)

env.shutdown()