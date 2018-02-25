import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

os.system('mkdir detect4video')
os.system('rm detect4video/*')
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import tensorflow as tf
from PIL import Image
from file_helper import read_lines
from object_detection.utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


lines = read_lines('oid_cattle.txt')
examples = list()
last_path = ''
for line in lines:
    infos = line.split()
    img_path = infos[0]
    box = list()
    for bi in infos[1:5]:
        box.append(float(bi))
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)
    vis_util.draw_bounding_box_on_image(image, box[1], box[0], box[3], box[2])
    image.show()
    raw_input(img_path)