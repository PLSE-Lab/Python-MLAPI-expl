#!/usr/bin/env python
# coding: utf-8

# # Transfer learning for Object detection models

# In the first part of our notebook, we created annotated images of people from LFW dataset. In this notebook, we will train an existing model to recognize those 62 people using Tensorflow object detection library.  
# P.S. refer to version 17 for the quantized model; this is a float type model.

# In[ ]:


import os
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print(os.listdir("/kaggle/input//"))
# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
tf.__version__
#cv2.__version__


# In[ ]:


get_ipython().run_line_magic('cd', '/kaggle/working/')
# changing the directory for installation of below modules
# tensorflow models section has an object detection library
# piping logs to text files, otherwise the notebook is not very readable
get_ipython().system('git clone --quiet https://github.com/tensorflow/models.git > models.txt')
# protobuf is needed for creating py files from models library above
get_ipython().system('apt-get install -qq protobuf-compiler > proto.txt ')
# pycoco for coco scores
get_ipython().system('pip install -q pycocotools > pycoco.txt')
# creating py files from protos
get_ipython().run_line_magic('cd', '/kaggle/working/models/research')
get_ipython().system('protoc object_detection/protos/*.proto --python_out=.')
# setting path, if not set, python can not use object detection library (from models)
import os
os.environ['PYTHONPATH'] += ':/kaggle/working/models/research/:/kaggle/working/models/research/slim/'
# if prints OK, then installation and environment are set up correctly 
get_ipython().system('python object_detection/builders/model_builder_test.py')


# In[ ]:


# copy coco config to the working dir for editing
get_ipython().system('cp /kaggle/input/my-training/nssd_mobilenet_v2_coco.config /kaggle/working/')


# In[ ]:


# modified from https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/
# changing config file by changing the path to records and setting variables
import re
pipeline_fname="/kaggle/working/nssd_mobilenet_v2_coco.config"
fine_tune_checkpoint="/kaggle/input/my-training/model.ckpt"
train_record_fname="/kaggle/input/face-recognition-part-1/train.tfrecord"
test_record_fname="/kaggle/input/face-recognition-part-1/test.tfrecord"
label_map_pbtxt_fname="/kaggle/input/face-recognition-part-1/object_label.pbtxt"
batch_size=64
num_steps=20000 # more steps of training gives higher accuracy
num_classes=62 # specify classes
num_examples=5000 # generate examples by augmenting existing images in tfrecords

with open(pipeline_fname) as f:
    s = f.read()
with open(pipeline_fname, 'w') as f:
    
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
    
    # tfrecord files both train and test.
    s = re.sub(
        '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(test.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)
    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)
    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)
    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    s = re.sub('num_examples: [0-9]+',
               'num_examples: {}'.format(num_examples), s) 
    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    f.write(s)


# In[ ]:


# folder for saving trained model
#!rm -r /kaggle/working/training
os.mkdir('/kaggle/working/training')


# In[ ]:


#!ls /kaggle/working/
train='/kaggle/working/training/'


# In[ ]:


# if you are training for the first time, you can remove this cell!!!
# this is for further training from my last checkpoint
get_ipython().system('cp /kaggle/input/float-trained16k/* /kaggle/working/training/')
#opening checkpoint text file to edit the last step:
ch=open('/kaggle/working/training/checkpoint','w')
ch.write('model_checkpoint_path: "model.ckpt-16000"\nall_model_checkpoint_paths: "model.ckpt-16000"')
ch.close() # without checkpoint step, training does not continue


# In[ ]:


#training starts by running model_main.py and passing the paths
get_ipython().system('python /kaggle/working/models/research/object_detection/model_main.py     --pipeline_config_path={pipeline_fname}     --model_dir=/kaggle/working/training/     --alsologtostderr > /kaggle/working/train.txt')
print("Finished training")


# In[ ]:


get_ipython().system('ls /kaggle/working/training')
#!cat {pipeline_fname}


# In[ ]:


get_ipython().system('cp {pipeline_fname} /kaggle/working/training/')
conf_path='/kaggle/working/training/nssd_mobilenet_v2_coco.config'


# In[ ]:


#save frozen graph of the model for inference later (for notebook usage only, not for tflite converting)
import re
import numpy as np

output_directory = '/kaggle/working/trained_model'

lst = os.listdir(train)
lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
last_model = lst[steps.argmax()].replace('.meta', '')
last_model_path = os.path.join(train, last_model)

print(last_model_path)
get_ipython().system('python /kaggle/working/models/research/object_detection/export_inference_graph.py     --input_type=image_tensor     --pipeline_config_path={pipeline_fname}     --output_directory={output_directory}     --trained_checkpoint_prefix={last_model_path}     > /kaggle/working/graph.txt')
print('Finished exporting')


# In[ ]:


get_ipython().system('ls /kaggle/working/trained_model/')


# In[ ]:


os.mkdir('/kaggle/working/freezetflite')
outd='/kaggle/working/freezetflite'


# In[ ]:


# freezing graph for tensorflow lite for android use
get_ipython().system('python /kaggle/working/models/research/object_detection/export_tflite_ssd_graph.py --pipeline_config_path={conf_path} --trained_checkpoint_prefix={last_model_path} --output_directory={outd} --add_postprocessing_op=true')


# In[ ]:


# converting frozen graph to obtain tflite, used on Android later
get_ipython().system("tflite_convert  --graph_def_file=/kaggle/working/freezetflite/tflite_graph.pb --output_file=/kaggle/working/freezetflite/62faces_float.tflite --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --input_shape=1,300,300,3 --allow_custom_ops ")


# Our model has been trained. Now we will run inference on some images to check whether it is detecting and recognizing faces.

# In[ ]:


# modified from https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/
get_ipython().run_line_magic('cd', '/kaggle/working/models/research/object_detection')
import warnings
warnings.filterwarnings('ignore')

PATH_TO_CKPT='/kaggle/working/trained_model/frozen_inference_graph.pb'
PATH_TO_LABELS = label_map_pbtxt_fname
PATH_DIR ='/kaggle/input/faces-data/'
TEST_IMAGE_PATHS =[os.path.join(PATH_DIR+i) for i in os.listdir(PATH_DIR)]

#import matplotlib; matplotlib.use('Agg')
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt

from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops


# This is needed to display the images.
#%matplotlib inline


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=5)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)


# In[ ]:


import shutil
shutil.rmtree('/kaggle/working/models')


# In[ ]:


get_ipython().system('ls /kaggle/working/training/')


# In[ ]:


get_ipython().system('ls /kaggle/working/')


# In[ ]:


get_ipython().system('ls /kaggle/working/trained_model/')


# In[ ]:




