# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import os
import time
from typing import Optional, List
import dm_env
import numpy as np
import cv2

from . import robot_utils
from .configuration_realsense import RealSenseCameraConfig
from .camera_realsense import RealSenseCamera


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
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("realsense_rgb", rgb_image)
        cv2.waitKey(1)
        return obs

    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Non-finite action values: {action}")

        # Heuristic unit fix:
        # - PiPERMotorsBus.write expects radians for joints 1-6 and gripper in [0, 0.08].
        # - Some policies/datasets emit joint commands in the device's milli-degree-scaled units
        #   (i.e., already multiplied by factor ~= 57295). In that case, dividing converts to radians.
        # joints = action[:6]
        # if np.max(np.abs(joints)) > 50.0:
        #     joints = joints / float(self.piper_follower.factor)
        # # Safety guard: refuse obviously unsafe joint magnitudes.
        # if np.max(np.abs(joints)) > 4.0:
        #     raise ValueError(f"Refusing to execute unsafe joint command (rad): {joints}")

        # gripper = float(action[6])
        # gripper = float(np.clip(gripper, 0.0, 0.08))

        # safe_action = np.concatenate([joints, [gripper]]).tolist()
        self.piper_follower.write(target_joint=action)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )


def make_real_env() -> RealEnv:
    return RealEnv()



