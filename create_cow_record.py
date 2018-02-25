import os
from random import shuffle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system("rm data/*.rcd")
import tensorflow as tf

from file_helper import read_lines
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', 'data/cattles.rcd', 'Path to output TFRecord')
flags.DEFINE_string('head_path', 'hard_head.txt', 'Path to input label')
flags.DEFINE_string('body_path', 'hard_body.txt', 'Path to input label')
FLAGS = flags.FLAGS


def create_tf_example(example):
    # TODO(user): Populate the following variables from your example.
    height = 640  # Image height
    width = 360  # Image width
    filename = example['img_path']  # Filename of the image. Empty if image is not from file
    # encoded_image_data = imutils.resize(cv2.imread(filename), width=360).tobytes() # Encoded image bytes
    encoded_image_data = tf.gfile.FastGFile(filename, 'rb').read()
    decoded_image = tf.image.decode_jpeg(encoded_image_data)
    decoded_image_resized = tf.cast(tf.image.resize_images(decoded_image, [height, width]), tf.uint8)
    encoded_image_data = tf.image.encode_jpeg(decoded_image_resized)
    encoded_image_data = tf.Session().run(encoded_image_data)
    image_format = 'jpeg'  # b'jpeg' or b'png'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    for i in range(len(example['boxes'])):
        xmins.append(float(example['boxes'][i][0]))
        xmaxs.append(float(example['boxes'][i][2]))
        ymins.append(float(example['boxes'][i][1]))
        ymaxs.append(float(example['boxes'][i][3]))

    classes_text = list()
    classes = list()
    for i in range(len(example['boxes'])):
        # classes_text.append('/m/01kb5c')
        # classes.append(546)
        if example['class'] == 21:
            classes.append(21)
            classes_text.append('/m/01xq0k1')
        elif example['class'] == 19:
            classes.append(19)
            classes_text.append('/m/03k3r')
        # if 'strange' in filename:
        #     classes_text.append('/m/0jbk')
        #     classes.append(13)
        # else:
        #     classes_text.append('/m/01xq0k1')
        #     classes.append(13)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def add_to_examples(head_path, examples, cl):
    lines = read_lines(head_path)

    last_path = ''
    for line in lines:
        infos = line.split()
        if last_path != infos[0]:
            examples.append(
                {
                    'img_path': infos[0],
                    'boxes': list(),
                    'class': cl,
                }
            )
        examples[-1]['boxes'].append(infos[1:5])
    return examples

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    examples = list()
    examples = add_to_examples(FLAGS.head_path, examples, 21)
    # examples = add_to_examples('oid_cattle.txt', examples, 21)
    examples = add_to_examples(FLAGS.body_path, examples, 19)
    # examples = add_to_examples('animal_box.txt', examples, 21)
    shuffle_idxes = range(len(examples))
    shuffle(shuffle_idxes)
    for i in shuffle_idxes:
        tf_example = create_tf_example(examples[i])
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
