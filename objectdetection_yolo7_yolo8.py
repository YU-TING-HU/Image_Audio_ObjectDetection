# -*- coding: utf-8 -*-
"""
# YOLOv7
"""
# Clone the YOLOv7 GitHub repository
!git clone https://github.com/WongKinYiu/yolov7.git
# Download the pre-trained YOLOv7 model weights
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt


"""## Prepare Data & Label"""

import pandas as pd
import numpy as np
import seaborn as sns
import os
import shutil
import xml.etree.ElementTree as ET
import glob
import json

# Create folders for storing images and labels
def create_folder(path):
  """
  Create a folder if it does not exist.

  Args:
  path (str): The directory path to create.
  """
  if not os.path.exists(path):
    os.makedirs(path)

# Create directory structure for training, validation, and test datasets
create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/train/images')
create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/train/labels')
create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/val/images')
create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/val/labels')
create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/images')
create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/labels')

# Function to convert XML bounding boxes to YOLO format
# Source: https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5
def xml_to_yolo_bbox(bbox, w, h):
  """
  Convert Pascal VOC bounding box format to YOLO format.

  Args:
  bbox (list): List containing xmin, ymin, xmax, ymax coordinates.
  w (int): Width of the image.
  h (int): Height of the image.

  Returns:
  list: YOLO formatted bounding box [x_center, y_center, width, height].
  """
  # xmin, ymin, xmax, ymax
  x_center = ((bbox[2] + bbox[0]) / 2) / w
  y_center = ((bbox[3] + bbox[1]) / 2) / h
  width = (bbox[2] - bbox[0]) / w
  height = (bbox[3] - bbox[1]) / h
  return [x_center, y_center, width, height]

# Path to the folder containing images
img_folder = '/content/drive/MyDrive/ObjectDetection_Yolo7/images'
_, _, files = next(os.walk(img_folder))
pos = 0

# Iterate over each image file to distribute it into train, val, and test datasets
for f in files:
  source_img = os.path.join(img_folder, f)
  if pos < 700:
    dest_folder = '/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/train'
  elif (pos >= 700 and pos < 800):
    dest_folder = '/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/val'
  else:
    dest_folder = '/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test'
  destination_img = os.path.join(dest_folder,'images', f)
  shutil.copy(source_img, destination_img)

  # Define the corresponding label file paths
  label_file_basename = os.path.splitext(f)[0]
  label_source_file = f"{label_file_basename}.xml"
  label_dest_file = f"{label_file_basename}.txt"

  label_source_path = os.path.join('/content/drive/MyDrive/ObjectDetection_Yolo7/annotations', label_source_file)
  label_dest_path = os.path.join(dest_folder, 'labels', label_dest_file)
  
  # If the XML label file exists, convert it to YOLO format and save it
  if os.path.exists(label_source_path):
    # parse the content of the xml file
    tree = ET.parse(label_source_path)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)
    classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    result = []

    for obj in root.findall('object'):
      label = obj.find("name").text
      # check for new classes and append to list
      index = classes.index(label)
      pil_bbox = [int(x.text) for x in obj.find("bndbox")]
      yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
      # convert data to string
      bbox_string = " ".join([str(x) for x in yolo_bbox])
      result.append(f"{index} {bbox_string}")
      if result:

        # generate a YOLO format text file for each xml file
        with open(label_dest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(result))
  pos += 1

# number of data
print(f"Number of images processed: {pos}")


"""## Training"""
# Train YOLOv7 model using the specified configuration and data
!python /content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/train.py --weights 'yolov7-e6e.pt' --data /content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/data/masks.yaml --workers 1 --batch-size 4 --img 416 --cfg /content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/cfg/training/yolov7-masks.yaml --name yolov7 --epochs 10


"""## Detection"""
# Perform detection using the trained YOLOv7 model
!python /content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/detect.py --weights /content/runs/train/yolov7/weights/best.pt --conf 0.4 --img-size 640 --source /content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/images

# Display the detection results
import glob
from IPython.display import Image, display
i = 0
limit = 1 # max number of images to print
for imageName in glob.glob('/content/runs/detect/exp/*.png'):
    if i < limit:
      display(Image(filename=imageName))
      print("\n")
    i = i + 1



"""# YOLOv8"""

!pip install ultralytics
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")
# Perform detection using YOLOv8 on a specific image
result = model("/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/images/maksssksksss89.png")
result
# Another way to perform detection using YOLOv8
!yolo detect predict model=yolov8n.pt source="/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/images/maksssksksss89.png" conf=0.3

# Display the detection result
display(Image(filename="/content/runs/detect/predict/maksssksksss89.png"))


"""## Training"""

# Build a new YOLOv8 model from scratch or load a pre-trained one
model = YOLO("yolov8n.yaml")
# load a pretrained model
model = YOLO("yolov8n.pt")
# Train the YOLOv8 model
results = model.train(data="/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/data/masks.yaml", epochs=1, imgsz=512, batch=4, verbose=True, device='gpu')
# Export the trained YOLOv8 model
model.export()


"""## Detection"""

# Perform detection using the trained YOLOv8 model
!yolo detect predict model=/content/runs/detect/train/weights/best.pt source="/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/images/maksssksksss89.png" conf=0.4
# Display the detection result
display(Image(filename="/content/runs/detect/predict2/maksssksksss89.png"))