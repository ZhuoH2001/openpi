import logging
import time
from functools import cached_property
from typing import Any
from dataclasses import dataclass
from piper_sdk import *
from errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from camera_realsense import CameraRealSense
logger = logging.getLogger(__name__)

@dataclass
class PiPERMotorsBusConfig:
    can_name: str = "piper_follower"
    motors: dict[str, tuple[int, str]] = {
                                            "joint_1": (1, "agilex_piper"),
                                            "joint_2": (2, "agilex_piper"),
                                            "joint_3": (3, "agilex_piper"),
                                            "joint_4": (4, "agilex_piper"),
                                            "joint_5": (5, "agilex_piper"),
                                            "joint_6": (6, "agilex_piper"),
                                            "gripper": (7, "agilex_piper"),
                                        }

class PiPERMotorsBus():
    def __init__(self, config: PiPERMotorsBusConfig):
        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()
        self.motors = config.motors
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [6 joints + 1 gripper] * 0.0
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]
        self.factor = 57295.780490 #1000*180/3.1416926
        self.is_connected = False
    
    def connect(self):
        while (not self.piper.EnablePiper()):
             time.sleep(0.03)
        self.is_connected = True
        self.go_zero()

    def disconnect(self):
        while(self.piper.DisablePiper()):
            time.sleep(0.03)
        self.is_connected = False

    def go_zero(self):
        """
            移动到零位
        """
        self.write(target_joint=self.init_joint_position)
        
    def read(self):
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state

        return {
            "joint_1": joint_state.joint_1,
            "joint_2": joint_state.joint_2,
            "joint_3": joint_state.joint_3,
            "joint_4": joint_state.joint_4,
            "joint_5": joint_state.joint_5,
            "joint_6": joint_state.joint_6,
            "gripper": gripper_state.grippers_angle
        }
    
    def write(self, target_joint:list):
        """"
        Joint control
        - target joint: in radians
            joint_1(float):关节1角度 -92000~92000 / 57295.780490 #1000*180/3.1416926
            joint_2(float):关节2角度 -2400~120000 / 57295.780490 #1000*180/3.1416926
            joint_3(float):关节3角度 3000~-110000 / 57295.780490 #1000*180/3.1416926 
            joint_4(float):关节4角度 -90000~90000 / 57295.780490 #1000*180/3.1416926
            joint_5(float):关节5角度 80000~-80000 / 57295.780490 #1000*180/3.1416926
            joint_6(float):关节6角度 -90000~90000 / 57295.780490 #1000*180/3.1416926
            gripper_range:夹爪角度 0~0.08
        """
        joint_0 = round(target_joint[0]*self.factor)
        joint_1 = round(target_joint[1]*self.factor)
        joint_2 = round(target_joint[2]*self.factor)
        joint_3 = round(target_joint[3]*self.factor)
        joint_4 = round(target_joint[4]*self.factor)
        joint_5 = round(target_joint[5]*self.factor)
        gripper_range = round(target_joint[6]*1000*1000)

        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0) # 单位: 0.001°
        time.sleep(0.005)

    def safe_disconnect(self):
        """ 
            Move to safe disconnect position
        """
        self.write(target_joint=self.safe_disable_position)