import numpy as np
import cv2
import random
from datetime import datetime
from gpiozero import Servo
import os
from picamera2 import Picamera2
import time

class Sensor:
    def __init__(self, min_angle=0, max_angle=np.pi, servo_pin = 18):
        self.picam2 = Picamera2()
        sensor_resolution = self.picam2.sensor_resolution
        still_config = self.picam2.create_still_configuration(main={"size": sensor_resolution})
        self.picam2.configure(still_config)
        self.picam2.start()
        time.sleep(2)
        self.min_angle = min_angle
        self.max_angle = max_angle
        # Initialize servo on GPIO pin 18 with custom pulse widths
        self.servo = Servo(
            servo_pin, 
            min_pulse_width=500 / 1_000_000, 
            max_pulse_width=2500 / 1_000_000
        )
    
    def normalize_angle(self, angle):
        n = (angle - self.min_angle) / (self.max_angle - self.min_angle)
        n = (n*2) - 1
        return n
        
    def rotate_gear(self, n):
         self.servo.value = n   
         
    def delete_file(self,filename):
            os.system(f"rm '{filename}'")
    
    def take_photo(self):
            # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # random_number = random.random()
            # filename = f"{timestamp}__{random_number}.jpg"
            # os.system(f"rpicam-still -n --immediate -o '{filename}'")
            # image = cv2.imread(filename)
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # self.delete_file(filename)
            image = self.picam2.capture_array() # image compatible with cv2
            return image
            
            
    def extract_face(self, yaw):
        # Dummy implementation, replace with actual sensor code
        # For example, capture an image from a camera at the given yaw angle
        n = self.normalize_angle(yaw)
        self.rotate_gear(n)
        #image = np.zeros((512, 512, 3), dtype=np.uint8)
        time.sleep(.2)
        image = self.take_photo()
        return image
