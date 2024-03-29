import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.logger import logger

from ..base_controller import Controller
from .ours import Ours


class OursController(Controller):
    def __init__(
        self,
        grasp_time: float,
        post_grasp_dest,
        box_size,
        conveyor_level,
        ee_pos_scale,
        ours: Ours,
        cam_to_gt_R: R,
        max_speed=0.5,
    ):
        super().__init__(
            grasp_time,
            post_grasp_dest,
            box_size,
            conveyor_level,
            ee_pos_scale,
            max_speed,
        )
        self.ours = ours
        self.cam_to_gt_R = cam_to_gt_R

        self.real_grasp_time = None

    def _get_ee_val(self, obj_vel, rgb_img, depth_img, prev_rgb_img):
        ee_vel_cam, err, photo_err = self.ours.get_vel(
            rgb_img, 0*obj_vel, depth=depth_img, pre_img_src=prev_rgb_img
        )
        ee_vel_cam = ee_vel_cam[:3]
        ee_vel_cam += obj_vel
        ee_vel_gt = self.cam_to_gt_R.apply(ee_vel_cam)
        speed = min(self.max_speed, np.linalg.norm(ee_vel_gt))
        vel = ee_vel_gt * (
            speed / np.linalg.norm(ee_vel_gt) if not np.isclose(speed, 0) else 1
        )
        if err > 0.9:
            self.ready_to_grasp = True

        logger.debug(
            "controller (gt frame):",
            pred_vel=vel,
            pred_speed=np.linalg.norm(vel),
            photo_err=photo_err,
        )
        return vel

    def get_action(self, observations: dict):
        rgb_img = observations["rgb_img"]
        ee_pos = observations["ee_pos"]
        cur_t = observations["cur_t"]
        obj_vel = observations["obj_vel"]
        depth_img = observations.get("depth_img", None)
        prev_rgb_img = observations.get("prev_rgb_img", None)
        obj_vel_cam = self.cam_to_gt_R.inv().apply(obj_vel)

        action = np.zeros(5)
        if cur_t <= self.grasp_time and not self.ready_to_grasp:
            action[4] = -1
            action[:3] = self._get_ee_val(obj_vel_cam, rgb_img, depth_img, prev_rgb_img)
            if cur_t <= 0.6 * self.grasp_time:
                tpos = self._action_vel_to_target_pos(action[:3], ee_pos)
                # tpos[2] = max(tpos[2], self.conveyor_level + self.box_size[2] + 0.005)
                action[2] = self._target_pos_to_action_vel(tpos, ee_pos)[2]
        else:
            action[4] = 1
            if self.real_grasp_time is None:
                self.real_grasp_time = cur_t
            if cur_t <= self.real_grasp_time + 0.5:
                action[:3] = self._get_ee_val(
                    obj_vel_cam, rgb_img, depth_img, prev_rgb_img
                )
            elif cur_t <= self.real_grasp_time + 1.0:
                action[:3] = [0, 0, 0.5]
            else:
                action[:3] = self.post_grasp_dest - ee_pos
        return action, self.ours.get_iou(rgb_img)
