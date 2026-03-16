from PIL import Image
from datetime import datetime
import time
import json
import os
import numpy as np
from picamera2 import Picamera2

class Sensor:
    def __init__(self, cam=0, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)["sensor"]
        
        self.cam = cam
        self.picam2 = Picamera2(cam)
        sensor_resolution = self.picam2.sensor_resolution
        scale = self.config.get("resolution_scale", 0.5)
        sensor_resolution = [int(i * scale) for i in sensor_resolution]
        still_config = self.picam2.create_still_configuration(
            main={"size": sensor_resolution},
            )
        self.picam2.configure(still_config)
        self.picam2.start()
        time.sleep(self.config.get("startup_sleep", 2.0))
    
    def adjust_focus(self, lens_position):
        """Set the lens position.

        Args:
            lens_position (float): Lens position in "diopters" (units of 1/meters). See  https://github.com/raspberrypi/picamera2/issues/978#issue-2179641066 
        """
        self.picam2.set_controls(
                {
                    "AfMode": self.config.get("af_mode", 0), 
                    "LensPosition": lens_position
                }
            ) 

    def take_photo(self,save=False):
            image = self.picam2.capture_array() # image compatible with cv2
            if save:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                time_seconds = time.time()
                filename = f"cam{self.cam}_{timestamp}_{time_seconds}.jpg"
                # PIL expects RGB, Picamera2 default is also RGB
                Image.fromarray(image).save(filename)
            return image

class MultiSensor:
    def __init__(self, sensors):
        self.sensors = sensors
    def adjust_focus(self, lens_position):
        for sensor in self.sensors:
            sensor.adjust_focus(lens_position)
    def take_photo(self,save=False):
        images = []
        for sensor in self.sensors:
            images.append(sensor.take_photo(save=save))
        return np.concatenate(images, axis=1)

if __name__ == "__main__":
    print("I am here in sensor.py")
