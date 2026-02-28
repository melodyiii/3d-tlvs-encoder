# from src/lerobot/robots/robot.py

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import builtins
from pathlib import Path

import draccus

from lerobot.motors import MotorCalibration
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

from .config import RobotConfig


import numpy as np
import serial
import threading
import cv2
import time
from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt
# import seaborn as sns
# os.system('cls')





























# TODO(aliberts): action/obs typing such as Generic[ObsType, ActType] similar to gym.Env ?
# https://github.com/Farama-Foundation/Gymnasium/blob/3287c869f9a48d99454306b0d4b4ec537f0f35e3/gymnasium/core.py#L23
class Robot(abc.ABC):
    """
    The base abstract class for all LeRobot-compatible robots.

    This class provides a standardized interface for interacting with physical robots.
    Subclasses must implement all abstract methods and properties to be usable.

    Attributes:
        config_class (RobotConfig): The expected configuration class for this robot.
        name (str): The unique robot name used to identify this robot type.
    """

    # Set these in ALL subclasses
    config_class: builtins.type[RobotConfig]
    name: str


    def __init__(self, config: RobotConfig):
        self.robot_type = self.name
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir if config.calibration_dir else HF_LEROBOT_CALIBRATION / ROBOTS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()
        
        self.THRESHOLD =12
        self.NOISE_SCALE =60

        # self.PORT = "left_gripper_right_finger"
        # self.PORT ='/dev/ttyUSB4'
        self.PORT = 'COM5'
        self.BAUD = 2000000
    
        self.exitThread = False
    
        self.contact_data_norm = np.zeros((16,16))
        WINDOW_WIDTH = self.contact_data_norm.shape[1]*30
        WINDOW_HEIGHT = self.contact_data_norm.shape[0]*30
        cv2.namedWindow("Contact Data_left", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Contact Data_left",WINDOW_WIDTH, WINDOW_HEIGHT)


    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    def __enter__(self):
        """
        Context manager entry.
        Automatically connects to the camera.
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Context manager exit.
        Automatically disconnects, ensuring resources are released even on error.
        """
        self.disconnect()

    def __del__(self) -> None:
        """
        Destructor safety net.
        Attempts to disconnect if the object is garbage collected without cleanup.
        """
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass



    def readThread(self,serDev):
        data_tac = []
        num = 0
        t1=0
        backup = None
        self.flag=False
        current = None
        while True:
            if serDev.in_waiting > 0:
                try:
                    line = serDev.readline().decode('utf-8').strip()
                except:
                    line = ""
                if len(line) < 10:
                    if current is not None and len(current) == 16:
                        backup = np.array(current)
                        print("fps",1/(time.time()-t1+0.000001))
                        t1 =time.time()
                        data_tac.append(backup)
                        num += 1
                        if num > 30:
                            break
                    current = []
                    continue
                if current is not None:
                    str_values = line.split()
                    int_values = [int(val) for val in str_values]
                    matrix_row = int_values
                    current.append(matrix_row) 
    
        data_tac = np.array(data_tac)
        median = np.median(data_tac, axis=0)
        self.flag=True
        print("Finish Initialization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        
        while True:
            if serDev.in_waiting > 0:
                try:
                    line = serDev.readline().decode('utf-8').strip()
                    # print("fps",1/(time.time()-t1))
                    # t1 =time.time()
                except:
                    line = ""
                if len(line) < 10:
                    if current is not None and len(current) == 16:
                        backup = np.array(current)
                        # print(backup)
                    current = []
                    if backup is not None:
                        contact_data= backup-median-self.THRESHOLD
                        contact_data = np.clip(contact_data, 0, 100)
                        
                        if np.max(contact_data) < self.THRESHOLD:
                            self.contact_data_norm = contact_data /self.NOISE_SCALE
                        else:
                            # self.contact_data_norm = np.log(contact_data + 1) / np.log(2.0)
                            self.contact_data_norm = contact_data / np.max(contact_data)
    
                    continue
                if current is not None:
                    str_values = line.split()
                    int_values = [int(val) for val in str_values]
                    matrix_row = int_values
                    current.append(matrix_row) 
                        
                    continue
        
    def apply_gaussian_blur(contact_map, sigma=0.1):
        return gaussian_filter(contact_map, sigma=sigma)

    def temporal_filter(new_frame, prev_frame, alpha=0.2):
        """
        Apply temporal smoothing filter.
        'alpha' determines the blending factor.
        A higher alpha gives more weight to the current frame, while a lower alpha gives more weight to the previous frame.
        """
        return alpha * new_frame + (1 - alpha) * prev_frame
    
    def act_as_main(self):
        print('receive data test')

        while True:
            temp_filtered_data = self.temporal_filter(self.contact_data_norm, prev_frame)
            self.output_frame = prev_frame = temp_filtered_data
                    
    
            # Scale to 0-255 and convert to uint8
            temp_filtered_data_scaled = (temp_filtered_data * 255).astype(np.uint8)

            # Apply color map
            colormap = cv2.applyColorMap(temp_filtered_data_scaled, cv2.COLORMAP_VIRIDIS)

            cv2.imshow("Contact Data_left", colormap)
            cv2.waitKey(1)
            time.sleep(0.01)

    
    # TODO(aliberts): create a proper Feature class for this that links with datasets
    @property
    @abc.abstractmethod
    
    def observation_features(self) -> dict:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not. If `False`, calling :pymeth:`get_observation` or
        :pymeth:`send_action` should raise an error.
        """
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the robot.

        Args:
            calibrate (bool): If True, automatically calibrate the robot after connecting if it's not
                calibrated or needs calibration (this is hardware-dependant).
        """
        serDev = serial.Serial(self.PORT,self.BAUD) 
        # serDev = serial.Serial('/dev/ttyUSB0',2000000)
        serDev.flush()
        serialThread = threading.Thread(target=self.readThread, args=(serDev,))
        serialThread.daemon = True
        serialThread.start()
        self.prev_frame = np.zeros_like(self.contact_data_norm)
        
        while self.flag == False: # caution error may SUSPEND WHOLE PROGRAM
            time.sleep(0.01)
        
        serialThread2 = threading.Thread(target=self.act_as_main)
        serialThread2.daemon = True
        serialThread2.start()
        

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """
        Calibrate the robot if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """
        Helper to load calibration data from the specified file.

        Args:
            fpath (Path | None): Optional path to the calibration file. Defaults to `self.calibration_fpath`.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """
        Helper to save calibration data to the specified file.

        Args:
            fpath (Path | None): Optional path to save the calibration file. Defaults to `self.calibration_fpath`.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    @abc.abstractmethod
    def get_observation(self) -> RobotObservation:
        """
        Retrieve the current observation from the robot.

        Returns:
            RobotObservation: A flat dictionary representing the robot's current sensory state. Its structure
                should match :pymeth:`observation_features`.
        """

        # 16*16,float,0~1
        return self.output_frame;

    @abc.abstractmethod
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send an action command to the robot.

        Args:
            action (RobotAction): Dictionary representing the desired action. Its structure should match
                :pymeth:`action_features`.

        Returns:
            RobotAction: The action actually sent to the motors potentially clipped or modified, e.g. by
                safety limits on velocity.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot and perform any necessary cleanup."""
        pass
