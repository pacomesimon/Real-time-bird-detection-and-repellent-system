import os
import cv2
import numpy as np
import time

from sensor import Sensor
from agent import Agent
from siren.play import play_buzzer
from PIL import Image


sensor = Sensor()
pano_agent = Agent(sensor = sensor)


def get_angles_from_bboxes(bboxes, yaw):
    # [{'category_id': 0, 'bbox': [2420.0, 1121.0, 63.0, 63.0]}, {'category_id': 0, 'bbox': [2377.0, 1120.0, 63.0, 63.0]}]
    all_bboxes_list = []
    for bbox_dict in bboxes:
        all_bboxes_list.append(bbox_dict['bbox'])
    all_bboxes = np.array(all_bboxes_list)
    coordinates_x = all_bboxes[:,0] + all_bboxes[:,2]/2
    coordinates_y = all_bboxes[:,1] + all_bboxes[:,3]/2

    yaw = (yaw * 180 / np.pi)
    horizontal_FOV = 66 # degrees 
    horizontal_bias = yaw - horizontal_FOV/2
    horizontal_resolution = 4056
    horizontal_angles = (horizontal_resolution - coordinates_x) * (horizontal_FOV/horizontal_resolution) + horizontal_bias

    vertical_FOV = 52.3 # degrees
    vertical_bias = 45
    vertical_resolution = 3040
    vertical_angles = coordinates_y * (vertical_FOV/vertical_resolution) + vertical_bias

    all_angles = np.zeros_like(all_bboxes[:,:2])
    all_angles[:,0] = horizontal_angles
    all_angles[:,1] = vertical_angles
    return all_angles   

save_images_bool_txt_path = "../pi_online_evaluation/save_images_bool.txt"
with open(save_images_bool_txt_path, "w") as f:
            f.write('0')

def handle_prediction_log(current_image, prev_image, coco_bboxes,yaw):
      # Read save_images_bool.txt setting
      with open(save_images_bool_txt_path, "r") as f:
              save_images_setting = f.read().strip()

      if save_images_setting == "1":
          # Save current_image and prev_image
          timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
          pair_timestamp = time.time()
          curr_img_path = os.path.join("cached_images", f"{timestamp}_{pair_timestamp}_1.jpg")
          prev_img_path = os.path.join("cached_images", f"{timestamp}_{pair_timestamp}_0.jpg")
          
          Image.fromarray(current_image).save(curr_img_path)
          Image.fromarray(prev_image).save(prev_img_path)

      
      timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
      if len(coco_bboxes)==0:
        return timestamp, None
      play_buzzer()
      all_angles = get_angles_from_bboxes(
        coco_bboxes,
        yaw
      )
      logs = []
      for angles in all_angles:
        logs.append(
            f"{timestamp} {angles[0]} {angles[1]}"
        )
      return "\n".join(logs)

pano_agent.handle_prediction = handle_prediction_log

with open("predictions_bboxes.txt", "a") as f:
  for prediction in pano_agent.work():
    print(prediction)
    if type(prediction) != tuple:
      f.write(str(prediction) + "\n")