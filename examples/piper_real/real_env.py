# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import time
from typing import Optional, List
import dm_env
import numpy as np
import cv2

from examples.piper_real import robot_utils
from configuration_realsense import RealSenseCameraConfig
from camera_realsense import RealSenseCamera


class RealEnv:
    def __init__(self):
        """"
            Piper + Realsense Camera 真机环境
        """
        config_realsense = RealSenseCameraConfig(serial_number_or_name="140222070379", warmup_s=2.0)
        self.realsense_cam = RealSenseCamera(config=config_realsense)
        self.realsense_cam.connect()
        self.piper_follower = robot_utils.PiPERMotorsBus(robot_utils.PiPERMotorsBusConfig())
        self.piper_follower.connect()

    def get_images(self):
        return self.realsense_cam.async_read()

    def get_observation(self):
        obs = collections.OrderedDict()
        state = self.piper_follower.read()
        state = np.array([state[f"joint_{i+1}"] for i in range(6)] + [state["gripper"]])
        obs["state"] = state
        rgb_image = self.get_images()
        # rgb_image = rgb_image[:448,20:468]
        # rgb_image = cv2.resize(rgb_image, (224,224))
        obs["images"] = rgb_image
        return obs

    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    def step(self, action):
        self.piper_follower.write(target_joint=action)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )


def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)



