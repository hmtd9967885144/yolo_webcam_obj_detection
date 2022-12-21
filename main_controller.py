# import darknet functions to perform object detections
from IPython.display import display, Javascript, Image

from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time


from darknet import *

cfg_dir = '/cfg/yolov3.cfg'
data_dir = '/cfg/coco.data'
weights_dir = '/yolov3.weights'


network, class_names, class_colors = load_network(cfg_dir, data_dir, weights_dir)

width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio