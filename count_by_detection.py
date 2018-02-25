import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.system('rm oid_cattle.txt detect_rst/*')
import numpy as np
import tensorflow as tf
from PIL import Image

from file_helper import write_line

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'model/faster_rcnn_inception_resnet_v2_atrous_oid_2017_11_08'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'oid_bbox_trainable_label_map.pbtxt')

NUM_CLASSES = 545

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

PATH_TO_TEST_IMAGES_DIR = 'r10'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '%d.jpg' % i) for i in range(71, 1860) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

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
    for image_path in TEST_IMAGE_PATHS:
      if not os.path.exists(image_path):
          continue
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      width = image_np.shape[1]
      height = image_np.shape[0]
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
      for box, score, cl in zip(np.squeeze(t_boxes), np.squeeze(t_scores), np.squeeze(t_classes).astype(np.int32)):
          if cl == 150 or cl == 45 or cl == 13 or cl == 76:
              if score > 0.4:
                  boxes.append(box)
                  scores.append(score)
                  classes.append(cl)
                  num += 1
                  write_line('oid_cattle.txt', '%s %4f %4f %4f %4f' % (image_path, box[1], box[0], box[3], box[2]))
          # elif score > 0.5:
          #     boxes.append(box)
          #     scores.append(score)
          #     classes.append(cl)
          #     num += 1

          if cl != 150 and score > 0.5:
              print '%s: %d' % (image_path, cl )

      if num > 0:
          # Visualization of the results of a detection.
          result = vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes).reshape(num, 4),
              np.squeeze(classes).astype(np.int32).reshape(num,),
              np.squeeze(scores).reshape(num,),
              category_index,
              use_normalized_coordinates=True,
              min_score_thresh=0.4,
              line_thickness=8)
          Image.fromarray(result).save('detect_rst/' + image_path.split('/')[-1])
          total_cnt += num
          print total_cnt
          # for i, score in enumerate(np.squeeze(scores).reshape(num, )):
          #     if score > 0.2:
          #         print np.squeeze(classes).reshape(num,)[i]

      # plt.figure(figsize=IMAGE_SIZE)
      if num == 0:
        print image_path
        write_line('hard_cattle.txt', '%s %4f %4f %4f %4f' % (image_path, box[1], box[0], box[3], box[2]))
    print total_cnt
