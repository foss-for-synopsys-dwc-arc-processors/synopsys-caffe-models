## Change Log
### Conversion
This model generated from cnn_models/tensorflow/object_detection/ssd_mobilenet_v2_coco_2018_03_29.pb with evconvert (2020.01)
* command  
`evconvert tf2ev -g ssd_mobilenet_v2_coco_2018_03_29.pb -p image_tensor 1 300 300 3 -f Preprocessor/sub -l concat concat_1 -o ssd_mobilenet_v2_coco_2018_03_29`  

## Models
### evconvert converted models
* convert_optimized.prototxt  
* convert_optimized.caffemodel  

### manually optimized models  
- convert_optimized_detection_yx.prototxt  
  - We add an extra input layer `pbox` for priorbox generation, the data can be found in `anchor_yx.pkl` in this folder.
We also added extra `detection_out` layer to produce the same detection output as the TensorFlow model,
which means the detection box order is `[ymin, xmin, ymax, xmax]`.  
  - This model is not supported by mapping tools
- convert_optimized_yx.prototxt  
  - We add extra layers including `mbox_loc`, `mbox_conf_flatten`, `mbox_priorbox` and `detection_out` to produce the same detection output as the TensorFlow model.
  - This model could be supported by mapping tools  
- ssd_mobilenet_v2_yx_updated.prototxt (similar transformation as for SSD-MobileNet (v1)


## Compare script
- script to compare `convert_optimized_detection_yx.prototxt` and pb model
```
import os
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import pickle
import argparse

from tensorflow.python.platform import gfile
import caffe


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

def load_tf_model(tf_model_pb):
    tf.reset_default_graph()
    gpu_options=tf.GPUOptions(allow_growth=True) #, visible_device_list='1'

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with gfile.FastGFile(tf_model_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    sess.run(tf.global_variables_initializer())
    return sess

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_tf_model_inference(sess, img_path, out_op_names=None):
    input_op = sess.graph.get_tensor_by_name('image_tensor:0')
    if out_op_names is None:
        out_op_names = ['Preprocessor/sub', 'concat', 'concat_1', \
                   'detection_boxes', 'detection_scores', 'num_detections', 'detection_classes', \
                       'Postprocessor/ExpandDims']

    out_ops = []
    for op_name in out_op_names:
        out_op = sess.graph.get_tensor_by_name(op_name + ':0')
        out_ops.append(out_op)

    image = Image.open(img_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    feed_input = image_np_expanded

    out = sess.run(out_ops, feed_dict={input_op: feed_input})

    return out

def parse_tf_detections(out):
    num_detections = int(out[5][0])
    detection_boxes = out[3][0]
    detection_scores = out[4][0]
    detection_classes = out[6][0]
    detections = np.zeros((1, 1, num_detections, 7))
    for i in range(num_detections):
        detections[0][0][i][1] = detection_classes[i]
        detections[0][0][i][2] = detection_scores[i]
        detections[0][0][i][3] = detection_boxes[i][1]
        detections[0][0][i][4] = detection_boxes[i][0]
        detections[0][0][i][5] = detection_boxes[i][3]
        detections[0][0][i][6] = detection_boxes[i][2]
    return detections

#Only used when caffe detection output format is [ymin, xmin, ymax, xmax]
def parse_caffe_detections(detections):
    num_detections = len(detections[0][0])
    new_detections = detections.copy()
    # [ymin, xmin, ymax, xmax] -> [xmin, ymin, ymax, xmax]
    for i in range(num_detections):
        new_detections[0][0][i][3] = detections[0][0][i][4]
        new_detections[0][0][i][4] = detections[0][0][i][3]
        new_detections[0][0][i][5] = detections[0][0][i][6]
        new_detections[0][0][i][6] = detections[0][0][i][5]
    return new_detections

def run_caffe_model_inference(net, input_data, pbox):
    net.blobs['Preprocessor/sub'].data[...] = input_data
    net.blobs['pbox'].data[...] = pbox
    net.forward()
    return net.blobs['detection_out'].data

def run_inference_all(sess, net, img_path, pbox_pkl):
    tf_outs = run_tf_model_inference(sess, img_path)
    tf_Preprocessor_sub_data = tf_outs[0]
    tf_detections = parse_tf_detections(tf_outs)
    with open(pbox_pkl, 'rb') as F:
        pbox = pickle.load(F, encoding='iso-8859-1')
    caffe_detections = run_caffe_model_inference(net, tf_Preprocessor_sub_data.transpose(0, 3, 1, 2), pbox)
    return tf_detections, caffe_detections

def run_caffe_model_inference_only(net, img_path, pbox_pkl):
    mean_values = [127.5, 127.5, 127.5]
    scale = 0.00784313771874
    rgb_color_order = True

    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300), interpolation = cv2.INTER_LINEAR)
    img = img.astype(dtype=np.float32)
    img = np.rollaxis(img, 2) # convert hwc_to_chw
    img[0] = img[0] - mean_values[0]
    img[1] = img[1] - mean_values[1]
    img[2] = img[2] - mean_values[2]
    img = img * scale
    if rgb_color_order:
        img = np.flipud(img) # convert bgr_to_rgb
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

    with open(pbox_pkl, 'rb') as F:
        pbox = pickle.load(F, encoding='iso-8859-1')

    net.blobs['Preprocessor/sub'].data[...] = img
    net.blobs['pbox'].data[...] = pbox# # img#
    net.forward()
    return net.blobs['detection_out'].data

def compare_tf_caffe_detections(tf_detections, caffe_detections):
    if tf_detections.shape == caffe_detections.shape:
        tf_detections = np.sort(tf_detections.reshape(-1))
        caffe_detections = np.sort(caffe_detections.reshape(-1))
        diff = np.max(tf_detections - caffe_detections)
    else:
        if caffe_detections.shape == (1, 1, 1, 7) and \
                int(caffe_detections[0][0][0][1]) == -1 and \
                tf_detections.shape == (1, 1, 0, 7):
            diff = 0
        else:
            diff = 1000
    return diff

#Only used when caffe detection output format is [ymin, xmin, ymax, xmax]
def parse_caffe_detections(detections):
    num_detections = len(detections[0][0])
    new_detections = detections.copy()
    # [ymin, xmin, ymax, xmax] -> [xmin, ymin, ymax, xmax]
    for i in range(num_detections):
        new_detections[0][0][i][3] = detections[0][0][i][4]
        new_detections[0][0][i][4] = detections[0][0][i][3]
        new_detections[0][0][i][5] = detections[0][0][i][6]
        new_detections[0][0][i][6] = detections[0][0][i][5]
    return new_detections

def print_detections(detections):
    detection_count = len(detections[0][0])
    if detection_count == 0:
        print("None detections found!")
        return
    if detection_count == 1:
        if int(detections[0][0][0][1]) == -1:
            print("None detections found!")
            return
    print("{} detections found!".format(detection_count))
    print ("label, score, xmin, ymin, xmax, ymax")
    for i in range(detection_count):
        print("%d, %.4f, %.4f, %.4f, %.4f %.4f" %(detections[0][0][i][1], detections[0][0][i][2], \
                detections[0][0][i][3], detections[0][0][i][4], \
                detections[0][0][i][5], detections[0][0][i][6]))

def FILE_EXIST(file):
    if not os.path.exists(file):
        raise argparse.ArgumentTypeError("File {} doesn't exist.".format(file))
    else:
        return file

def main():

    parser = argparse.ArgumentParser(description='Caffe VS TF')
    parser.add_argument('--prototxt', type=FILE_EXIST, default=None, required=True, help='Prototxt file.')
    parser.add_argument('--caffemodel', type=FILE_EXIST, default=None, required=True, help='Trained weight file [.caffemodel].')
    parser.add_argument('--pb', type=FILE_EXIST, help='Frozen PB.')
    parser.add_argument('--pbox_pkl', type=FILE_EXIST, help='Priorbox Pickle File')
    parser.add_argument('--image', type=FILE_EXIST, help='Image to be tested')

    args = parser.parse_args()

    sess = load_tf_model(args.pb)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    tf_detections, caffe_detections = run_inference_all(sess, net, args.image, args.pbox_pkl)
#     print("TensorFlow detections: {}".format(tf_detections))
#     print("Caffe detections: {}".format(caffe_detections))
    diff = compare_tf_caffe_detections(tf_detections, caffe_detections)
    print("TensorFlow vs Caffe detections difference: {}".format(diff))
    
    # Convert yx to xy box format
    caffe_detections = parse_caffe_detections(caffe_detections)
    print("1. TensorFlow detections")
    print_detections(tf_detections)
    print("2. Caffe detections")
    print_detections(caffe_detections)

if __name__ == '__main__':
    main()
```
- script to compare `convert_optimized_yx.prototxt` and pb model
```
import os
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import pickle
import argparse

from tensorflow.python.platform import gfile
import caffe

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

def load_tf_model(tf_model_pb):
    tf.reset_default_graph()
    gpu_options=tf.GPUOptions(allow_growth=True) #, visible_device_list='1'

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with gfile.FastGFile(tf_model_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    sess.run(tf.global_variables_initializer())
    return sess

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_tf_model_inference(sess, img_path, out_op_names=None):
    input_op = sess.graph.get_tensor_by_name('image_tensor:0')
    if out_op_names is None:
        out_op_names = ['Preprocessor/sub', 'concat', 'concat_1', \
                   'detection_boxes', 'detection_scores', 'num_detections', 'detection_classes', \
                       'Postprocessor/ExpandDims']

    out_ops = []
    for op_name in out_op_names:
        out_op = sess.graph.get_tensor_by_name(op_name + ':0')
        out_ops.append(out_op)

    image = Image.open(img_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    feed_input = image_np_expanded

    out = sess.run(out_ops, feed_dict={input_op: feed_input})

    return out

def parse_tf_detections(out):
    num_detections = int(out[5][0])
    detection_boxes = out[3][0]
    detection_scores = out[4][0]
    detection_classes = out[6][0]
    detections = np.zeros((1, 1, num_detections, 7))
    for i in range(num_detections):
        detections[0][0][i][1] = detection_classes[i]
        detections[0][0][i][2] = detection_scores[i]
        detections[0][0][i][3] = detection_boxes[i][1]
        detections[0][0][i][4] = detection_boxes[i][0]386281
        detections[0][0][i][5] = detection_boxes[i][3]
        detections[0][0][i][6] = detection_boxes[i][2]
    return detections

#Only used when caffe detection output format is [ymin, xmin, ymax, xmax]
def parse_caffe_detections(detections):
    num_detections = len(detections[0][0])
    new_detections = detections.copy()
    # [ymin, xmin, ymax, xmax] -> [xmin, ymin, ymax, xmax]
    for i in range(num_detections):
        new_detections[0][0][i][3] = detections[0][0][i][4]
        new_detections[0][0][i][4] = detections[0][0][i][3]
        new_detections[0][0][i][5] = detections[0][0][i][6]
        new_detections[0][0][i][6] = detections[0][0][i][5]
    return new_detections

def run_caffe_model_inference(net, input_data):
    net.blobs['Preprocessor/sub'].data[...] = input_data
    net.forward()
    return net.blobs['detection_out'].data

def run_inference_all(sess, net, img_path):
    tf_outs = run_tf_model_inference(sess, img_path)
    tf_Preprocessor_sub_data = tf_outs[0]
    tf_detections = parse_tf_detections(tf_outs)
    caffe_detections = run_caffe_model_inference(net, tf_Preprocessor_sub_data.transpose(0, 3, 1, 2))
    return tf_detections, caffe_detections

def run_caffe_model_inference_only(net, img_path):
    mean_values = [127.5, 127.5, 127.5]
    scale = 0.00784313771874
    rgb_color_order = True

    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300), interpolation = cv2.INTER_LINEAR)
    img = img.astype(dtype=np.float32)
    img = np.rollaxis(img, 2) # convert hwc_to_chw
    img[0] = img[0] - mean_values[0]
    img[1] = img[1] - mean_values[1]
    img[2] = img[2] - mean_values[2]
    img = img * scale
    if rgb_color_order:
        img = np.flipud(img) # convert bgr_to_rgb
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

    net.blobs['Preprocessor/sub'].data[...] = img
    net.forward()
    return net.blobs['detection_out'].data

def compare_tf_caffe_detections(tf_detections, caffe_detections):
    if tf_detections.shape == caffe_detections.shape:
        tf_detections = np.sort(tf_detections.reshape(-1))
        caffe_detections = np.sort(caffe_detections.reshape(-1))
        diff = np.max(np.absolute(tf_detections - caffe_detections))
    else:
        if caffe_detections.shape == (1, 1, 1, 7) and \
                int(caffe_detections[0][0][0][1]) == -1 and \
                tf_detections.shape == (1, 1, 0, 7):
            diff = 0
        else:
            diff = 1000
    return diff

def print_detections(detections):
    detection_count = len(detections[0][0])
    if detection_count == 0:
        print("None detections found!")
        return
    if detection_count == 1:
        if int(detections[0][0][0][1]) == -1:
            print("None detections found!")
            return
    print("{} detections found!".format(detection_count))
    print ("label, score, xmin, ymin, xmax, ymax")
    for i in range(detection_count):
        print("%d, %.4f, %.4f, %.4f, %.4f %.4f" %(detections[0][0][i][1], detections[0][0][i][2], \
                detections[0][0][i][3], detections[0][0][i][4], \
                detections[0][0][i][5], detections[0][0][i][6]))


def FILE_EXIST(file):
    if not os.path.exists(file):
        raise argparse.ArgumentTypeError("File {} doesn't exist.".format(file))
    else:
        return file

def main():

    parser = argparse.ArgumentParser(description='Caffe VS TF')
    parser.add_argument('--prototxt', type=FILE_EXIST, default=None, required=True, help='Prototxt file.')
    parser.add_argument('--caffemodel', type=FILE_EXIST, default=None, required=True, help='Trained weight file [.caffemodel].')
    parser.add_argument('--pb', type=FILE_EXIST, help='Frozen PB.')
    parser.add_argument('--image', type=FILE_EXIST, help='Image to be tested')

    args = parser.parse_args()

    sess = load_tf_model(args.pb)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    tf_detections, caffe_detections = run_inference_all(sess, net, args.image)
    print("Compare TF and Converted Caffemodel on image {}, image is preprocessed in TensorFlow".format(args.image))
    # Convert yx to xy box format
    caffe_detections = parse_caffe_detections(caffe_detections)
    print("1. TensorFlow detections")
    print_detections(tf_detections)
    print("2. Caffe detections")
    print_detections(caffe_detections)
    diff = compare_tf_caffe_detections(tf_detections, caffe_detections)
    print("Max TensorFlow vs Caffe detections difference: {}".format(diff))


if __name__ == '__main__':
    main()
```

