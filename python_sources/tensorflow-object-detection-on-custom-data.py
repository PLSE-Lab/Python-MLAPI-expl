#!/usr/bin/env python
# coding: utf-8

# # TF Object Detection on Custom Data
# In this notebook we will train a TensorFlow Object Detection model with a (large) custom dataset. We will cover the following steps:  
# * Install TensorFlow and TF Object Detection API
# * Fetch a pre-trained model from the TensorFlow detection model zoo
# * Configure the model and run training with the custom dataset
# * Make predictions with the trained model
# 
# Now, the TensorFlow Object Detection API is not for the faint of heart to get started on, but once a few tweaks are in place, it is mostly smooth sailing.

# ## Install TF Object Detection API
# The [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is at the time of writing not compatible with TF2 , so we need to install TF1.14 first. This notebook produces quite a lot of local files, and to keep a tidy house any large files not required will be removed (`rm -fr`).

# In[ ]:


get_ipython().run_line_magic('cd', '/kaggle/working')


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'HAVE_GPU = True # change according to environment\nif HAVE_GPU:\n    !pip install --user tensorflow-gpu==1.14 -q\nelse:\n    !pip install --user tensorflow==1.14 -q\n# never mind the `ERROR: tensorflow 2.1...` message below')


# In[ ]:


# make sure we the required packages
get_ipython().system('pip install --user Cython -q')
get_ipython().system('pip install --user contextlib2 -q')
get_ipython().system('pip install --user pillow -q')
get_ipython().system('pip install --user lxml -q')
get_ipython().system('pip install --user matplotlib -q')


# We need to install the `protoc` compiler. On windows, you can get [precompiled binaries here](https://github.com/protocolbuffers/protobuf/releases).

# In[ ]:


get_ipython().system('wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip -q')
get_ipython().system('unzip -o protobuf.zip')
get_ipython().system('rm protobuf.zip')


# Time to fetch the Object Detection API.
# <div class="alert alert-block alert-info">
# <b>Tip:</b> Move up one level to avoid kernel crash when cloning repositories with deep folder structure.
# </div>

# In[ ]:


get_ipython().run_line_magic('cd', '/kaggle')
get_ipython().system('rm -fr models')
get_ipython().system('git clone https://github.com/tensorflow/models.git')
get_ipython().system('rm -fr models/.git')


# Then compile the protocol buffer messages needed by the API.

# In[ ]:


# compile ProtoBuffers
get_ipython().run_line_magic('cd', 'models/research')
get_ipython().system('../../working/bin/protoc object_detection/protos/*.proto --python_out=.')


# In[ ]:


import os

os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONPATH']=os.environ['PYTHONPATH']+':/kaggle/models/research/slim:/kaggle/models/research'
os.environ['PYTHONPATH']


# That's it! We can now test our setup by running `model_builder_test.py`.

# In[ ]:


get_ipython().system('pwd')
get_ipython().system('python object_detection/builders/model_builder_test.py')


# Yohoo, it works!  
# Now, we did not install the Coco API, since we will be using the Pascal VOC evaluation metric. Unfortunately, there are some hardcoded references to the Coco API that needs to be commented out. Alternatively, just install the Coco API.

# In[ ]:


def disable_coco(file):
    with open(file,'r') as f:
        file_str = f.read()
    file_str=file_str.replace('from object_detection.metrics import coco_evaluation',
                    '#from object_detection.metrics import coco_evaluation')
    file_str=file_str.replace('object_detection.metrics import coco_tools',
                    '#object_detection.metrics import coco_tools')
    file_str=file_str.replace('\'coco_detection_metrics\':', '#\'coco_detection_metrics\':')
    file_str=file_str.replace('coco_evaluation.CocoDetectionEvaluator,', '#coco_evaluation.CocoDetectionEvaluator,')
    file_str=file_str.replace('\'coco_mask_metrics\':','#\'coco_mask_metrics\':')
    file_str=file_str.replace('coco_evaluation.CocoMaskEvaluator,','#coco_evaluation.CocoMaskEvaluator,')
    with open(file,'w') as f:
        f.write(file_str)

disable_coco('./object_detection/eval_util.py')


# ## Fetch a model from the zoo
# We will start with a pre-trained model from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Our custom dataset is the [ArTaxOr dataset](https://www.kaggle.com/mistag/arthropod-taxonomy-orders-object-detection-dataset), which contains images of invertebrate animals. Thus it makes sense to choose one of the iNaturalist Species-trained model. 

# In[ ]:


get_ipython().run_line_magic('cd', 'object_detection')
get_ipython().system('wget -O faster_rcnn_resnet50_fgvc_2018_07_19.tar.gz http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_fgvc_2018_07_19.tar.gz -q')
get_ipython().system('tar xvzf faster_rcnn_resnet50_fgvc_2018_07_19.tar.gz')
get_ipython().system('rm faster_rcnn_resnet50_fgvc_2018_07_19.tar.gz')
get_ipython().run_line_magic('cd', '..')


# <div class="alert alert-block alert-info">
# <b>Tip:</b> Remove the <b>checkpoint</b> file, otherwise training will fail while loading graph.
# </div>

# In[ ]:


get_ipython().system('rm object_detection/faster_rcnn_resnet50_fgvc_2018_07_19/checkpoint')


# Furthermore, create a directory for saved models (otherwise an error will occur when training is finished).

# In[ ]:


get_ipython().run_line_magic('cd', 'object_detection/faster_rcnn_resnet50_fgvc_2018_07_19')
get_ipython().system('mkdir export')
get_ipython().run_line_magic('cd', 'export')
get_ipython().system('mkdir Servo')
get_ipython().run_line_magic('cd', '../../..')


# ### Config file
# Then we need to define the model `.config` file. Here we set up paths to the dataset and a few other parameters. Thankfully, TFRecords for the ArTaxOr dataset has been created in [this notebook](https://www.kaggle.com/mistag/tensorflow-tfrecords-demystified) so we can link directly to its output files. We will use a 80-20 split for training and evaluation, and since the dataset is sharded in 50 files, we can simply select 10 of them (arbitrary) to go into the evaluation set. We also need to determine how many images there are in the evaluation set to configure the evaluation stage correctly:

# In[ ]:


#import tensorflow as tf

#input_pattern='/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????1-of-00050.tfrecord;/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????7-of-00050.tfrecord'
#input_files = tf.io.gfile.glob(input_pattern)
#data_set = tf.data.TFRecordDataset(input_files)
#records_n = sum(1 for record in data_set)
records_n = 3075 # takes a long time to run this, so cheating here
print("records_n = {}".format(records_n))


# In[ ]:


import sys

os.environ['DATA_PATH']='/kaggle/input/tensorflow-tfrecords-demystified'
os.environ['MODEL_PATH']='object_detection/faster_rcnn_resnet50_fgvc_2018_07_19'


# In[ ]:


get_ipython().run_cell_magic('writefile', "'object_detection/faster_rcnn_resnet50_fgvc_2018_07_19/ArTaxOr.config'", 'model {\n  faster_rcnn {\n    num_classes: 7 # ArTaxOr has 7 classes currently\n    image_resizer {\n      keep_aspect_ratio_resizer {\n        min_dimension: 600\n        max_dimension: 1024\n      }\n    }\n    feature_extractor {\n      type: \'faster_rcnn_resnet50\'\n      first_stage_features_stride: 16\n    }\n    first_stage_anchor_generator {\n      grid_anchor_generator {\n        scales: [0.25, 0.5, 1.0, 2.0]\n        aspect_ratios: [0.5, 1.0, 2.0]\n        height_stride: 16\n        width_stride: 16\n      }\n    }\n    first_stage_box_predictor_conv_hyperparams {\n      op: CONV\n      regularizer {\n        l2_regularizer {\n          weight: 0.0\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n          stddev: 0.01\n        }\n      }\n    }\n    first_stage_nms_score_threshold: 0.0\n    first_stage_nms_iou_threshold: 0.7\n    first_stage_max_proposals: 300\n    first_stage_localization_loss_weight: 2.0\n    first_stage_objectness_loss_weight: 1.0\n    initial_crop_size: 14\n    maxpool_kernel_size: 2\n    maxpool_stride: 2\n    second_stage_batch_size: 32\n    second_stage_box_predictor {\n      mask_rcnn_box_predictor {\n        use_dropout: false\n        dropout_keep_probability: 1.0\n        fc_hyperparams {\n          op: FC\n          regularizer {\n            l2_regularizer {\n              weight: 0.0\n            }\n          }\n          initializer {\n            variance_scaling_initializer {\n              factor: 1.0\n              uniform: true\n              mode: FAN_AVG\n            }\n          }\n        }\n      }\n    }\n    second_stage_post_processing {\n      batch_non_max_suppression {\n        score_threshold: 0.0\n        iou_threshold: 0.6\n        max_detections_per_class: 50\n        max_total_detections: 100\n      }\n      score_converter: SOFTMAX\n    }\n    second_stage_localization_loss_weight: 2.0\n    second_stage_classification_loss_weight: 1.0\n  }\n}\n\ntrain_config: {\n  batch_size: 1\n  num_steps: 4000000\n  optimizer {\n    momentum_optimizer: {\n      learning_rate: {\n        manual_step_learning_rate {\n          initial_learning_rate: 0.0002\n          schedule {\n            step: 20000\n            learning_rate: .00002\n          }\n          schedule {\n            step: 50000\n            learning_rate: .000002\n          }\n        }\n      }\n      momentum_optimizer_value: 0.9\n    }\n    use_moving_average: false\n  }\n  gradient_clipping_by_norm: 10.0\n  fine_tune_checkpoint: "/kaggle/models/research/object_detection/faster_rcnn_resnet50_fgvc_2018_07_19/model.ckpt"\n  from_detection_checkpoint: true\n  load_all_detection_checkpoint_vars: true\n  data_augmentation_options {\n    random_horizontal_flip {\n    }\n  }\n}\n\ntrain_input_reader: {\n  label_map_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr.pbtxt"\n  tf_record_input_reader {\n    input_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????0-of-00050.tfrecord"\n    input_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????2-of-00050.tfrecord"\n    input_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????3-of-00050.tfrecord"\n    input_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????4-of-00050.tfrecord"\n    input_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????5-of-00050.tfrecord"\n    input_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????6-of-00050.tfrecord"\n    input_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????8-of-00050.tfrecord"\n    input_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????9-of-00050.tfrecord"\n  }\n}\n\neval_config: {\n  metrics_set: "pascal_voc_detection_metrics"\n  #use_moving_averages: false\n  num_examples: 3075\n}\n\neval_input_reader: {\n  label_map_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr.pbtxt"\n  shuffle: false\n  num_readers: 1\n  tf_record_input_reader {\n    input_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????1-of-00050.tfrecord"\n    input_path: "/kaggle/input/tensorflow-tfrecords-demystified/ArTaxOr-????7-of-00050.tfrecord"\n  }\n}')


# In[ ]:


get_ipython().system('pwd')


# ## Training

# In[ ]:


# Note! Tensorboard only works in editor mode (kernel running), so we will not be using it here.
#%load_ext tensorboard
#%tensorboard --logdir=object_detection/faster_rcnn_resnet50_fgvc_2018_07_19


# 20000 steps take about 2h to run. Training will output large amounts of text, and once things are working it is better to dump it to a file rather than having to scroll down past thousands of lines.

# In[ ]:


old_stdout = sys.stdout
sys.stdout = open('/kaggle/working/train.log', 'w')
get_ipython().system('python object_detection/model_main.py     --pipeline_config_path=object_detection/faster_rcnn_resnet50_fgvc_2018_07_19/ArTaxOr.config     --model_dir=object_detection/faster_rcnn_resnet50_fgvc_2018_07_19     --num_train_steps=60000     --sample_1_of_n_eval_examples=1     --alsologtostderr=False')
sys.stdout = old_stdout


# Export the trained model to working directory using the supplied script.

# In[ ]:


get_ipython().run_cell_magic('capture', 'cap_out --no-stderr', '!mkdir /kaggle/working/trained\n!python object_detection/export_inference_graph.py \\\n    --input_type image_tensor \\\n    --pipeline_config_path object_detection/faster_rcnn_resnet50_fgvc_2018_07_19/ArTaxOr.config \\\n    --trained_checkpoint_prefix object_detection/faster_rcnn_resnet50_fgvc_2018_07_19/model.ckpt-60000 \\\n    --output_directory /kaggle/working/trained')


# In[ ]:


get_ipython().system('ls -al /kaggle/working/trained')


# Zip it for easy download.

# In[ ]:


get_ipython().system('tar -cvzf /kaggle/working/trained_model.tar /kaggle/working/trained')
get_ipython().system('gzip /kaggle/working/trained_model.tar')


# Let's check precision vs. training steps by parsing data from the log file.

# In[ ]:


get_ipython().system('pip install --user parse -q')
from parse import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

with open('/kaggle/working/train.log', 'r') as f:
    data=f.read()


# In[ ]:


loss=[]
for r in findall("Loss/RPNLoss/localization_loss = {:f}", data):
    loss.append(r[0])
mAP=[]
for r in findall("/mAP@0.5IOU = {:f}", data):
    mAP.append(r[0])
step=[]
for r in findall("global_step = {:d}", data):
    step.append(r[0])
plt.figure(figsize=(16, 8))
plt.plot(step,mAP)
plt.xlabel('Global step')
plt.legend(['mAP@0.5IOU']);


# Let's have a look at precision for each class:

# In[ ]:


APAraneae=[]
for r in findall("AP@0.5IOU/Araneae = {:f}", data):
    APAraneae.append(r[0])
APColeoptera=[]
for r in findall("AP@0.5IOU/Coleoptera = {:f}", data):
    APColeoptera.append(r[0])
APDiptera=[]
for r in findall("AP@0.5IOU/Diptera = {:f}", data):
    APDiptera.append(r[0])
APHemiptera=[]
for r in findall("AP@0.5IOU/Hemiptera = {:f}", data):
    APHemiptera.append(r[0])
APHymenoptera=[]
for r in findall("AP@0.5IOU/Hymenoptera = {:f}", data):
    APHymenoptera.append(r[0])
APLepidoptera=[]
for r in findall("AP@0.5IOU/Lepidoptera = {:f}", data):
    APLepidoptera.append(r[0])
APOdonata=[]
for r in findall("AP@0.5IOU/Odonata = {:f}", data):
    APOdonata.append(r[0])
plt.figure(figsize=(16, 8))
plt.plot(step,APAraneae)
plt.plot(step,APColeoptera)
plt.plot(step,APDiptera)
plt.plot(step,APHemiptera)
plt.plot(step,APHymenoptera)
plt.plot(step,APLepidoptera)
plt.plot(step,APOdonata)
plt.xlabel('Global step')
plt.legend(['AP Araneae', 'AP Coleoptera', 'AP Diptera', 'AP Hemiptera', 'AP Hymenoptera', 'AP Lepidoptera', 'AP Odonata']);


# Odonata (dragonflies and damselflies) is the class with the highest score, while Hymenoptera (bees, wasps, ants) is the class the model struggles most with.

# ## Prediction (detections)
# The easiest way to make predictions (or detections) with the trained model is to use the API supplied script `infer_detections.py`, which expects images in a TFRecord file. Note that this script is difficult on Windows machines. We will make predictions on the [ArTaxOr TestSet](https://www.kaggle.com/mistag/arthropod-taxonomy-orders-object-detection-testset). The [starter kernel](https://www.kaggle.com/mistag/starter-arthropod-taxonomy-orders-testset) outputs a TFRecord file, so we can simply link to that. The detections are output in a separate TFRecord file, which we will process further down.

# In[ ]:


get_ipython().run_cell_magic('capture', 'cap_out --no-stderr', '!python object_detection/inference/infer_detections.py \\\n  --input_tfrecord_paths=/kaggle/input/starter-arthropod-taxonomy-orders-testset/ArTaxOr_TestSet.tfrecord \\\n  --output_tfrecord_path=/kaggle/working/ArTaxOr_detections.tfrecord \\\n  --inference_graph=/kaggle/working/trained/frozen_inference_graph.pb \\\n  --discard_image_pixels')


# In[ ]:


get_ipython().system('ls /kaggle/working')


# First, we'll import pickled annotation data from the [ArtAxOr TestSet Starter notebook](https://www.kaggle.com/mistag/starter-arthropod-taxonomy-orders-testset). Then we read in the TFRecord with the detections, and create a Pandas frame with the detected bounding boxes.

# In[ ]:


get_ipython().run_cell_magic('capture', '', "import pandas as pd\nimport tensorflow as tf\n\nlabels=pd.read_pickle('/kaggle/input/starter-arthropod-taxonomy-orders-testset/testset_labels.pkl')\ndf=pd.read_pickle('/kaggle/input/starter-arthropod-taxonomy-orders-testset/testset_filelist.pkl')\nanno=pd.read_pickle('/kaggle/input/starter-arthropod-taxonomy-orders-testset/testset_objects.pkl')")


# In[ ]:


pdf=pd.DataFrame(columns=['score', 'label_idx', 'left', 'top', 'right', 'bottom', 'by', 'filename'])
example = tf.train.Example()
for record in tf.compat.v1.io.tf_record_iterator('/kaggle/working/ArTaxOr_detections.tfrecord'):
    example.ParseFromString(record)
    f = example.features.feature
    score = f['image/detection/score'].float_list.value
    score = [x for x in score if x >= 0.60]
    l = len(score)
    pdf=pdf.append({'score': score,
                    'label_idx': f['image/detection/label'].int64_list.value[:l],
                    'left': f['image/detection/bbox/xmin'].float_list.value[:l],
                    'top': f['image/detection/bbox/ymin'].float_list.value[:l],
                    'right': f['image/detection/bbox/xmax'].float_list.value[:l],
                    'bottom': f['image/detection/bbox/ymax'].float_list.value[:l],
                    'by': f['image/by'].bytes_list.value[0].decode(),
                    'filename': f['image/filename'].bytes_list.value[0].decode()}, ignore_index=True)


# In[ ]:


pdf.head()


# Then we define a few helper functions for plotting the test images and bounding boxes.

# In[ ]:


get_ipython().system('pip install --user python-resize-image -q')


# In[ ]:


from PIL import Image, ImageFont, ImageDraw
from resizeimage import resizeimage
import numpy as np

TSET_PATH = '/kaggle/input/arthropod-taxonomy-orders-object-detection-testset/ArTaxOr_TestSet/'

#fontname = 'C:/Windows/fonts/micross.ttf' # Windows
fontname = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf' # Linux
font = ImageFont.truetype(fontname, 20) if os.path.isfile(fontname) else ImageFont.load_default()

def resize_image(file, width, height, stretch=False):
    with Image.open(file) as im:
        img = im.resize((width, height)) if stretch else resizeimage.resize_contain(im, [width, height])
    img=img.convert("RGB")    
    return img

#draw boundary box
def bbox(img, xmin, ymin, xmax, ymax, color, width, label, score):
    draw = ImageDraw.Draw(img)
    xres, yres = img.size[0], img.size[1]
    box = np.multiply([xmin, ymin, xmax, ymax], [xres, yres, xres, yres]).astype(int).tolist()
    txt = " {}: {}%" if score >= 0. else " {}"
    txt = txt.format(label, round(score, 1))
    ts = draw.textsize(txt, font=font)
    draw.rectangle(box, outline=color, width=width)
    if len(label) > 0:
        if box[1] >= ts[1]+3:
            xsmin, ysmin = box[0], box[1]-ts[1]-3
            xsmax, ysmax = box[0]+ts[0]+2, box[1]
        else:
            xsmin, ysmin = box[0], box[3]
            xsmax, ysmax = box[0]+ts[0]+2, box[3]+ts[1]+1
        draw.rectangle([xsmin, ysmin, xsmax, ysmax], fill=color)
        draw.text((xsmin, ysmin), txt, font=font, fill='white')
    
#prediction
def plot_img_pred(img, xres, yres, axes, scores, xmin, ymin, xmax, ymax, classes, title, by=''):
    wscale = min(1,xres/yres)
    hscale = min(1,yres/xres)
    for i in range(len(scores)):
        if scores[i]> 0.5 and classes[i]>0:
            label = labels.name.iloc[int(classes[i]-1)]
            color=labels.color.iloc[int(classes[i]-1)]
            width, height = xmax[i]-xmin[i], ymax[i]-ymin[i]
            xcenter, ycenter = xmin[i] + width/2., ymin[i] + height/2.
            sxmin = .5+(xcenter-.5)*wscale-.5*wscale*width
            symin = .5+(ycenter-.5)*hscale-.5*hscale*height
            sxmax = .5+(xcenter-.5)*wscale+.5*wscale*width
            symax = .5+(ycenter-.5)*hscale+.5*hscale*height
            bbox(img, sxmin, symin, sxmax, symax, color, 2, label, 100*scores[i])
    plt.setp(axes, xticks=[], yticks=[])
    axes.set_title(title) if by == '' else axes.set_title(title+'\n'+by)
    plt.imshow(img)

#ground truth
def plot_img_gt(img, axes, boxes, stretch, title, by=''):
    wscale = 1. if stretch else min(1,boxes.xres.iloc[0]/boxes.yres.iloc[0])
    hscale = 1. if stretch else min(1,boxes.yres.iloc[0]/boxes.xres.iloc[0])
    for i in range(len(boxes)):
        label = boxes.label.iloc[i]
        color=labels.color.iloc[boxes.label_idx.iloc[i]]
        xmin = .5+(boxes.xcenter.iloc[i]-.5)*wscale-.5*wscale*boxes.width.iloc[i]
        ymin = .5+(boxes.ycenter.iloc[i]-.5)*hscale-.5*hscale*boxes.height.iloc[i]
        xmax = .5+(boxes.xcenter.iloc[i]-.5)*wscale+.5*wscale*boxes.width.iloc[i]
        ymax = .5+(boxes.ycenter.iloc[i]-.5)*hscale+.5*hscale*boxes.height.iloc[i]
        bbox(img, xmin, ymin, xmax, ymax, color, 2, label, -1)
    plt.setp(axes, xticks=[], yticks=[])
    axes.set_title(title) if by == '' else axes.set_title(title+'\n'+by)
    plt.imshow(img)

def pred_batch(idx):
    if idx + 2 < len(pdf):
        rows = 3
    else:
        rows = len(pdf) - idx
    fig = plt.figure(figsize=(16,rows*8))
    for i in range(rows):
        img = resize_image(TSET_PATH+'positives/'+pdf.filename.iloc[i+idx], 512, 512, False)
        by = pdf.by.iloc[i+idx]
        axes = fig.add_subplot(rows, 2, 1+i*2)
        boxes = anno[anno.id == df.id.iloc[i+idx]][['label', 'label_idx', 'xres', 'yres', 'xcenter', 'ycenter', 'width', 'height']]
        plot_img_gt(img, axes, boxes, False, 'Ground truth', by)
        img = resize_image(TSET_PATH+'positives/'+pdf.filename.iloc[i+idx], 512, 512, False)
        axes = fig.add_subplot(rows, 2, 2+i*2)
        plot_img_pred(img, boxes.xres.iloc[0], boxes.yres.iloc[0], axes, pdf.score[i+idx], pdf.left[i+idx], pdf.top[i+idx], 
                      pdf.right[i+idx], pdf.bottom[i+idx],
                      pdf.label_idx[i+idx], 'Detections', '')


# Finally we can show some images, with ground truth to the left and detections to the right. Some detections are overlapping, and additional non-max suppression seems to be needed here.

# In[ ]:


pred_batch(0)


# In[ ]:


pred_batch(3)


# In[ ]:


pred_batch(6)


# In[ ]:


pred_batch(9)


# In[ ]:


pred_batch(12)


# In[ ]:


pred_batch(15)


# In[ ]:


pred_batch(18)


# ## Summary
# Any object detection framework that has all files located in a directory called `research` should ring a few alarm bells when it comes to expectations of a slick user experience. However, we have seen that a few tweaks are all that is needed to get going with the TensorFlow Object Detection API. What about other options for object detection? PyTorch Detectron2 is the only other framework that has a pretrained model zoo, but currently it does not run on Kaggle (and no Windows support). TensorFlow Hub has several pre-trained models, and otherwise one would have to engage in detail implementation of models like YOLO etc. What we really need is to get object detection from research level and into mainstream.
