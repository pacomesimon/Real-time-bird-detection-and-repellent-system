import cv2
import numpy as np
import time
import json
import os
import random
import torch
from torchvision import transforms
import torch.nn as nn

with open("config.json", "r") as f:
    CONFIG = json.load(f)

DEVICE = CONFIG["model"]["device"]
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# a simple custom model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 
        self.classifier = nn.Sequential(
            nn.Linear(1 * 56 * 56, 32), # Adjust input features based on actual output size
            nn.ReLU(),
            nn.Linear(32, 2) # Assuming 2 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten the tensor for the fully connected layer
        x = self.classifier(x)
        return x

model = CustomCNN().to(DEVICE)

model.load_state_dict(torch.load(CONFIG["model"]["path"],
                      map_location=torch.device(DEVICE)
                      )
                      )

quant_dtype = CONFIG["model"]["quantization"]
if quant_dtype == "qint8":
    model = torch.quantization.quantize_dynamic(
        model,
        None,
        dtype=torch.qint8
    )
model.eval()
print("model loaded")

class BirdDetection():
    def __init__(self, 
                 threshold=CONFIG["bird_detection"]["threshold"],
                 percentile=CONFIG["bird_detection"]["percentile"],
                 classification_model=model,
                 horizon_height_ratio=CONFIG["bird_detection"]["horizon_height_ratio"],
                 min_bbox_dim=CONFIG["bird_detection"]["min_bbox_dim"],
                 max_bboxes = CONFIG["bird_detection"]["max_bboxes"],
                 ):
        self.threshold = threshold
        self.percentile = percentile
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize images to a fixed size
            transforms.ToTensor(),         # Convert images to PyTorch tensors
        ])
        self.classification_model = classification_model
        self.change_inf_time = []
        self.contour_inf_time = []
        self.class_inf_time = []
        self.horizon_height_ratio = horizon_height_ratio
        self.min_bbox_dim = min_bbox_dim
        self.max_bboxes = max_bboxes
    def binary_image_to_coco_bboxes(self, binary_img):
      """
      Args:
          binary_img (np.ndarray): Binary image (2D array with values 0 and 255 or 0 and 1)

      Returns:
          List[Dict]: A list of dictionaries each with keys:
                      {'category_id': 0, 'bbox': [x_min, y_min, width, height]}
      """
      # Ensure binary image is in the correct format for contour detection
      img_height = binary_img.shape[0]
      horizon_height = int(img_height * self.horizon_height_ratio)
      binary_img[horizon_height:, :] = 0
      binary_img = (binary_img * 255).astype(np.uint8)

      contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE
                                     )

      coco_bboxes = []
      for contour in contours:
          x, y, w, h = cv2.boundingRect(contour)
          if w < self.min_bbox_dim:
            x = x - ((self.min_bbox_dim-w)//2)
            w = self.min_bbox_dim
          if h < self.min_bbox_dim:
            y = y - ((self.min_bbox_dim-h)//2)
            h = self.min_bbox_dim
          coco_bboxes.append({
              'category_id': 0,
              'bbox': [float(x), float(y), float(w), float(h)]
          })

      return coco_bboxes

    def classify_image_bboxes(self,img, coco_bboxes):
      if len(coco_bboxes) == 0:
        return []
      cropped_list = []
      for bbox in coco_bboxes:
        x, y, w, h = bbox['bbox']
        cropped_img = img[int(y):int(y+h), int(x):int(x+w)]
        cropped_list.append(
            self.transform(cropped_img)
            )
      batch = torch.stack(cropped_list).to(DEVICE)
      outputs = self.classification_model(batch)
      _, predicted = torch.max(outputs.data, 1)
      correct_bboxes = []
      for i, pred in enumerate(predicted):
        if pred == 1:
          correct_bboxes.append(coco_bboxes[i])
      return correct_bboxes

    def __call__(self, img,prev_img,torelable_min = None,
                 torelable_max = None):
      t1 = time.time()
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

      diff = cv2.absdiff(img_gray, prev_img_gray)

      diff_min, diff_max = diff.min(), diff.max()
      diff = (diff - diff_min) / (diff_max - diff_min)
      diff_mean = diff.mean()
      self.change_inf_time.append(time.time()-t1)

      if True:
        t1 = time.time()
        mask_threshold = np.percentile(diff[diff >= self.threshold],
                                       self.percentile)

        coco_bboxes = self.binary_image_to_coco_bboxes(diff >= mask_threshold)
        total_n_boxes = len(coco_bboxes)
        self.contour_inf_time.append((total_n_boxes,time.time()-t1))
        t1 = time.time()
        coco_bboxes_raw = coco_bboxes.copy()
        if len(coco_bboxes_raw)>self.max_bboxes:
            coco_bboxes_raw = random.choices(coco_bboxes_raw,k=self.max_bboxes)
        coco_bboxes = self.classify_image_bboxes(img, coco_bboxes_raw) #TODO
        coco_bboxes += self.classify_image_bboxes(prev_img, coco_bboxes_raw)
        self.class_inf_time.append((total_n_boxes,time.time()-t1))
      return diff, coco_bboxes

def draw_coco_bboxes(image, annotations, color=(255, 0, 0), thickness=50):
    """
    Args:
        image (np.ndarray): The input image (BGR format).
        annotations (List[Dict]): List of annotations with COCO bboxes:
                                  [{'category_id': int, 'bbox': [x, y, w, h]}, ...]
        color (Tuple[int, int, int]): BGR color for the boxes (default is red).
        thickness (int): Line thickness of the boxes.

    Returns:
        np.ndarray: The image with bounding boxes drawn on it.
    """
    img_copy = image.copy()

    for ann in annotations:
        x, y, w, h = ann['bbox']
        top_left = (int(round(x)), int(round(y)))
        bottom_right = (int(round(x + w)), int(round(y + h)))
        cv2.rectangle(img_copy, top_left, bottom_right, color, thickness)

    return img_copy


bird_detector = BirdDetection()

class Agent:
    def __init__(self,sensor, 
                  min_angle=CONFIG["agent"]["min_angle"], 
                  max_angle=CONFIG["agent"]["max_angle"], 
                  stride=CONFIG["agent"]["stride"],
                  yaws = [0,0],
                  bird_detector=None
                  ):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.stride = stride
        self.yaws = yaws
        if bird_detector is None:
            self.bird_detector = BirdDetection()
        else:
            self.bird_detector = bird_detector
        self.sensor = sensor

    def extract_face(self, yaw=0.0):
        face = self.sensor.take_photo()
        return face

    def predict(self,current_image, prev_image):
      diff, coco_bboxes = self.bird_detector(current_image,prev_image)
      return coco_bboxes
    
    def handle_prediction(self,current_image, prev_image, coco_bboxes, yaw):
      viz_config = CONFIG["agent"]["visualization"]
      image = draw_coco_bboxes(
        current_image,
        coco_bboxes,
        tuple(viz_config["bbox_color"]),
        viz_config["bbox_thickness"]
      )
      preview_size = tuple(viz_config["preview_size"])
      return cv2.resize(image, preview_size, interpolation=cv2.INTER_NEAREST) 

    def work(self):
        turn = 1
        while True:
          yaws = self.yaws[::turn]
          turn = -turn
          for yaw in (yaws[1:]):
              prev_image = self.extract_face(yaw)
              current_image = self.extract_face(yaw)
              coco_bboxes = self.predict(current_image, prev_image)
              artifacts = self.handle_prediction(current_image, prev_image, coco_bboxes,yaw)
              yield artifacts
