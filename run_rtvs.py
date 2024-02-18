from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PlaceShapeInShapeSorter

from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import shutil
from utils.logger import logger
from PIL import Image
from utils.photo_error import mse_
import matplotlib.pyplot as plt

class RTVS:
    def __init__(
        self,
        des_img="./temp/000001.png",
        initial_pos = np.array([0, 0, 0, 0. , 0. , 0.70710678, 0.70710678, 1.0]),
        results_folder="./imgs/",
        controller_type="RLBenchController",
    ):
    
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        live_demos = False
        DATASET = '' if live_demos else './'

        # Setting up rlbench env
        self.env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode = True, frame = 'world'), gripper_action_mode=Discrete()),
            dataset_root = DATASET,
            obs_config=ObservationConfig(),
            static_positions=False,
            headless=False)
        self.env.launch()

        # Setting up task and the agent
        self.task = self.env.get_task(PlaceShapeInShapeSorter)
        self.demos = self.task.get_demos(amount=-1, random_selection=False)
        

        self.img_goal_path = des_img
        self.initial_pos = initial_pos
        self.folder = results_folder
        self.controller_type = controller_type
        
        
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        else:
            shutil.rmtree(self.folder, ignore_errors=True)
            os.makedirs(self.folder, exist_ok=True)
        if not os.path.isdir("./flow/"):
            os.mkdir("./flow/")
        else:
            shutil.rmtree("./flow/", ignore_errors=True)
            os.makedirs("./flow/", exist_ok=True)

    def _set_controller(self):
        logger.info(controller_type=self.controller_type)

        if self.controller_type == "RLBenchController":
            from controllers.rtvs import RLBenchController

            self.controller = RLBenchController(
                np.asarray(Image.open(self.img_goal_path).convert("RGB")),
                self.cam_to_gt_R,
                self.cam_int,
            )
        else:
            logger.info("Controller not found!")
    
    def step(self, v):
        V = v[0]
        dt = 1./250
        new_pose = np.append(self.curr_pose_, 1.0)  # for gripper
        new_pose[0] += -dt*V[1]  
        new_pose[1] += -dt*V[0]  
        new_pose[2] += -dt*V[2] 

        angles = R.from_quat(new_pose[3:7]).as_euler("xyz")
        angles[1] += dt*V[3]
        angles[0] += dt*V[4]
        angles[2] += dt*V[5]
        new_pose[3:7] = R.from_euler("xyz", angles).as_quat()

        self.curr_pose = new_pose
        obs, reward, terminate = self.task.step(self.curr_pose)
        self.curr_pose_ = obs.gripper_pose
        
        return obs, reward, terminate
    
    def run(self):
        _, obs = self.task.reset_to_demo(demo=self.demos[0])

        self.cam_int = np.array(obs.misc['wrist_camera_intrinsics'])
        self.cam_to_gt_R = np.array(obs.misc['wrist_camera_extrinsics'])
        logger.info("Extrinsix matrix: ", self.cam_to_gt_R)
        self.cam_to_gt_R = R.from_matrix(self.cam_to_gt_R[:3,:3])
        logger.info("quaternion_camera: ", self.cam_to_gt_R.as_quat())

        self.init_pose = obs.gripper_pose
        self.curr_pose_ = self.init_pose
        logger.info("Run start")
        logger.info("Initial pose : ", self.curr_pose_)

        self._set_controller()

        self.itr = 0
        img_src, depth_src = obs.wrist_rgb, obs.wrist_depth
        Image.fromarray(img_src).save(self.folder + "%06d.png" % (self.itr))
        pre_img_src = img_src
        img_goal = np.asarray(Image.open(self.img_goal_path).convert("RGB"))
        photo_error_val = mse_(img_src, img_goal)
        perrors = []

        logger.info("Initial Photometric Error: ", photo_error_val)

        step = 0
        while photo_error_val > 670 and step < 300:
            self.itr += 1 
            step+=1
            photo_error_val = mse_(img_src, img_goal)
            perrors.append(photo_error_val)

            vel = self.controller.get_vel(img_src, pre_img_src, depth_src)
            obs, _, _ = self.step([vel])

            logger.info("Step Number: ", step)
            logger.info("Velocity : ", vel.round(4))
            logger.info("Current : ", self.curr_pose)
            logger.info("Photometric Error : ", photo_error_val)

            pre_img_src = img_src
            img_src, depth_src = obs.wrist_rgb, obs.wrist_depth
            Image.fromarray(img_src).save(self.folder + "%06d.png" % (self.itr))
            
        plt.plot(perrors)
        plt.savefig("./error.png")
        self.env.shutdown()   

def main():
    env = RTVS()
    env.run()

if __name__ == "__main__":
    main()

# new_pose =  [ 2.28415591e-01 -8.15832987e-03  1.47203374e+00 -6.93235233e-06
#   9.92657840e-01 -2.14731972e-06  1.20956145e-01  1.00000000e+00]

# new_pose =  [ 1.78963375e-01 -8.14795867e-03  1.47153497e+00 -1.95684406e-05
#   9.92732525e-01 -3.43632064e-06  1.20342128e-01  1.00000000e+00]

# new_pose =  [ 1.29656520e-01 -8.14079493e-03  1.47103858e+00 -2.75553739e-05
#   9.92805123e-01  3.56345299e-06  1.19741954e-01  1.00000000e+00]

# new_pose =  [ 8.03185374e-02 -8.13926384e-03  1.47066998e+00 -2.75850980e-05
#   9.92865324e-01  1.16063165e-05  1.19240917e-01  1.00000000e+00]

# new_pose =  [ 3.12997445e-02 -8.13465938e-03  1.47033668e+00 -2.37107233e-05
#   9.92920458e-01  4.45745536e-06  1.18781775e-01  1.00000000e+00]
