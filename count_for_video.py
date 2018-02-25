import os

import shutil

from new_object_discover import is_new, accum_frame_padding, frame_shift
from overlap_box_filter import overlap_box_filter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from file_helper import write
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

os.system('mkdir detect4video')
os.system('rm detect4video/*')
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import tensorflow as tf
from PIL import Image


def add_str_on_img(image, total_cnt):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 48)
    except IOError:
        font = ImageFont.load_default()
    font.size = 48
    display_str = '%d' % total_cnt
    text_width, text_height = font.getsize(display_str)
    im_width, im_height = image.size
    draw.text(
        (im_width - text_width - 50, 50),
            '%d' % total_cnt,
            fill = 'green',
           font = font)


if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'model/frozen'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = '/home/cwh/coding/cow_count/r10'
# 335
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '%08d.jpg' % i) for i in range(90, 1860) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

last_boxes = list()
sum_padding = 0
avg_height = 0
avg_padding = 0
all_boxes = list()

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    total_cnt = 0
    empty_cnt = 0
    for image_path in TEST_IMAGE_PATHS:
      if not os.path.exists(image_path):
          continue
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (t_boxes, t_scores, t_classes, t_num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      # boxes = t_boxes
      # scores = t_scores
      # classes = t_classes
      # num = t_num
      boxes = list()
      scores = list()
      classes = list()
      num = 0
      # remove exclude cow
      score_thresh = 0.45
      for box, score, cl in zip(np.squeeze(t_boxes), np.squeeze(t_scores), np.squeeze(t_classes).astype(np.int32)):
          if cl == 21 or cl == 45 or cl == 19 or cl == 76 or cl == 546 or cl == 32:
              if score > score_thresh:
                  boxes.append(box)
                  scores.append(score)
                  classes.append(cl)
                  num += 1
          # elif score > 0.5:
          #     boxes.append(box)
          #     scores.append(score)
          #     classes.append(cl)
          #     num += 1

      valid_boxes, valid_box_flags = overlap_box_filter(boxes)
      valid_scores = list()
      valid_classes = list()
      num = 0
      for i, flag in enumerate(valid_box_flags):
        if flag:
          valid_scores.append(scores[i])
          valid_classes.append(classes[i])
          num += 1

      # new object discover

      if num > 0:
          print '%s: %d object detected' % (image_path, num)
          shift_boxes = frame_shift(-sum_padding, avg_height, -avg_padding, valid_boxes)
          for i, box in enumerate(shift_boxes):
              if valid_classes[i] != 19 and is_new(valid_boxes[i], last_boxes) and is_new(box, all_boxes) :
                  valid_classes[i] = 1
                  total_cnt += 1
                  print 'new box detect'
          for i, box in enumerate(shift_boxes):
              if classes[i] != 19:
                all_boxes.append(shift_boxes[i])
      # # double check
      # last2_boxes = [box for box in last_boxes]
      # last_boxes = valid_boxes

      avg_padding, avg_height = accum_frame_padding(avg_padding, avg_height, valid_boxes, last_boxes)
      print '%s avg_padding: %f, avg_height: %f, sum_padding: %f' % (image_path, avg_padding, avg_height, sum_padding)
      # last_boxes = [box for box in valid_boxes]
      last_boxes = list()
      for i, box in enumerate(valid_boxes):
          if valid_classes[i] == 1 or valid_classes[i] == 21:
              last_boxes.append(box)
      sum_padding += avg_padding


      add_str_on_img(image, total_cnt)
      image_np = load_image_into_numpy_array(image)
      if num > 0:
          # Visualization of the results of a detection.
          result = vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(valid_boxes).reshape(num, 4),
              np.squeeze(valid_classes).astype(np.int32).reshape(num,),
              np.squeeze(valid_scores).reshape(num,),
              category_index,
              use_normalized_coordinates=True,
              min_score_thresh=score_thresh,
              line_thickness=8)

          Image.fromarray(result).save('detect4video/' + image_path.split('/')[-1])
          # total_cnt += num


      else:
          if num < 3:
            write('no_target.txt', image_path+'\n')
          image.save('detect4video/' + image_path.split('/')[-1])
      # plt.figure(figsize=IMAGE_SIZE)
      if num == 0:
        print image_path
    print total_cnt
  os.system('mkdir detect4video2')
  os.system('rm detect4video2/*')
  for i, image_name in enumerate(sorted(os.listdir('detect4video'))):
      shutil.copy('detect4video/' + image_name, 'detect4video2/%05d.jpg' % i)