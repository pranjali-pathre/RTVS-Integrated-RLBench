from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PlaceShapeInShapeSorter
import numpy as np
import time
from PIL import Image
import os
import shutil
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    live_demos = False
    DATASET = '' if live_demos else './'
    # absolute = True
    absolute = True
    folder = "./temp/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    else:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
    itr = 0
    if absolute == True:
        env = Environment(
                action_mode=MoveArmThenGripper(
                    arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode = True, frame = 'world'), gripper_action_mode=Discrete()),
                dataset_root = DATASET,
                obs_config=ObservationConfig(),
                static_positions=False,
                headless=False)
        
        task = env.get_task(PlaceShapeInShapeSorter)
        demos = task.get_demos(amount=-1, random_selection=False)
        _, obs = task.reset_to_demo(demo=demos[0])
        Image.fromarray(obs.wrist_rgb).save(folder + "%06d.png" % (itr))

        for i in range(1):
            itr+=1
            init_pose = obs.gripper_pose
            new_pose = np.append(init_pose, 1.0)  # for gripper

            new_pose[0] -= 0.1  # 10cm back (down)
            # new_pose[1] -= 0.05  # 10cm right (right) 
            # new_pose[2] -= 0.05  # 10cm down (forward)
            angles = R.from_quat(obs.gripper_pose[3:]).as_euler("xyz", degrees=True)
            # angles[0] += 10 # look right 10 deg
            # angles[1] += 10 # look up 10 deg
            angles[2] += 5 # look anti 10 deg
            
            new_pose[3:7] = R.from_euler("xyz", angles, degrees=True).as_quat()
            # print("init_pos = ", init_pose, "\nnew_pose = ", new_pose)
            print("\nnew_pose = ", new_pose)

            time.sleep(2)
            obs, reward, term = task.step(new_pose)
            img_src, depth_src = obs.wrist_rgb, obs.wrist_depth
            Image.fromarray(img_src).save(folder + "%06d.png" % (itr))

        env.shutdown()

    else:
        env = Environment(
                action_mode=MoveArmThenGripper(
                    arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode = False, frame = 'world'), gripper_action_mode=Discrete()),
                dataset_root = DATASET,
                obs_config=ObservationConfig(),
                static_positions=True,
                headless=False)
        
        task = env.get_task(PlaceShapeInShapeSorter)
        _, obs = task.reset()
        init_pose = obs.gripper_pose
        new_pose = [0, 0, -0.1, 0, 0, 0, 1.0, 1.0]  # 10cm down
        expected_pose = list(init_pose)
        expected_pose[2] -= 0.1
        print("init_pos = ", init_pose, "\nnew_pose = ", new_pose, "\nexpected_pose = ", expected_pose)
        time.sleep(5)
        obs, reward, term = task.step(new_pose)

        env.shutdown()