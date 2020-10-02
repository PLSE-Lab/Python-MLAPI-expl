#!/usr/bin/env python
# coding: utf-8

# # Object Detection Video
# In this notebook we will run object detection on a video, frame by frame, using a model trained in another notebook. Then we will render a new video with the detections.  
# 
# This notebook builds on the work from the following sources:  
# * [The ArTaxOr dataset](https://www.kaggle.com/mistag/arthropod-taxonomy-orders-object-detection-dataset) - the hardest part, creating a dataset
# * [ArTaxOr starter kernel](https://www.kaggle.com/mistag/starter-arthropod-taxonomy-orders-data-exploring)
# * [Create TFRecords of the dataset](https://www.kaggle.com/mistag/tensorflow-tfrecords-demystified)
# * [Train a object detection model](https://www.kaggle.com/mistag/tensorflow-object-detection-on-custom-data)

# ## Install TensorFlow & Object Detection API
# The TensorFlow Object Detection API is still TF1.x only, se we need to install TF1.14 plus the API and related tools.

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'HAVE_GPU = True # change according to environment\nif HAVE_GPU:\n    !pip install --user tensorflow-gpu==1.14 -q\nelse:\n    !pip install --user tensorflow==1.14 -q\n# never mind the `ERROR: tensorflow 2.1...` message below')


# In[ ]:


# make sure we the required packages
get_ipython().system('pip install --user Cython -q')
get_ipython().system('pip install --user contextlib2 -q')
get_ipython().system('pip install --user pillow -q')
get_ipython().system('pip install --user lxml -q')
get_ipython().system('pip install --user matplotlib -q')


# We need to install the protoc compiler.

# In[ ]:


get_ipython().system('wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip -q')
get_ipython().system('unzip -o protobuf.zip')
get_ipython().system('rm protobuf.zip')


# Time to fetch the Object Detection API.

# In[ ]:


get_ipython().run_line_magic('cd', '/kaggle')
get_ipython().system('rm -fr models')
get_ipython().system('git clone https://github.com/tensorflow/models.git')
get_ipython().system('rm -fr models/.git')


# Then compile the protocol buffer messages needed by the API.

# In[ ]:


# compile ProtoBuffers
get_ipython().run_line_magic('cd', 'models/research')
get_ipython().system('/kaggle/working/bin/protoc object_detection/protos/*.proto --python_out=.')


# In[ ]:


import os

os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONPATH']=os.environ['PYTHONPATH']+':/kaggle/models/research/slim:/kaggle/models/research'
os.environ['PYTHONPATH']


# That's it! We can now test our setup by running model_builder_test.py.

# In[ ]:


get_ipython().system('pwd')
get_ipython().system('python object_detection/builders/model_builder_test.py')


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


# ## Convert input video to TFRecord
# The easiest way to make predictions (or detections) with the trained model is to use the API supplied script `infer_detections.py`, which expects images in a TFRecord file. We will run detections on a butterfly video from the [Short Videos dataset](https://www.kaggle.com/mistag/short-videos).

# In[ ]:


get_ipython().system('ls /kaggle/input/short-videos/insects')


# We will use OpenCV for video processing. Note that OpenCV uses BGR representation, while Pillow uses RGB representation, so there are a few conversions back and forth between these two representations.

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import cv2\nimport numpy as np\nimport pandas as pd\nimport tensorflow as tf\nimport matplotlib.pyplot as plt\nfrom io import BytesIO\nfrom PIL import Image, ImageFont, ImageDraw\n%matplotlib inline')


# In[ ]:


# this function creates a TFRecord of one frame
def create_tf_example(img, fcount):  
    height = img.shape[0] # Image height
    width = img.shape[1] # Image width
    im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    encoded_image_data = cv2.imencode('.jpg', frame)[1].tostring()
    image_format = b'jpeg'

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'video/frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[fcount]))
    }))
    return tf_record


# In[ ]:


get_ipython().run_cell_magic('time', '', "vid='/kaggle/input/short-videos/insects/butterflies_960p.mp4'\ntrec='/kaggle/working/butterflies_960p.tfrecord'\ncap = cv2.VideoCapture(vid)\nvideo_length, fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0\nwith tf.io.TFRecordWriter(trec) as writer:\n    for i in range(video_length):\n        ret , frame = cap.read()\n        if ret:\n            tf_record = create_tf_example(frame, fcount)\n            writer.write(tf_record.SerializeToString())\n            fcount+=1\ncap.release()")


# Let's check we have our TFRecord file:

# In[ ]:


get_ipython().system('ls -al /kaggle/working/*.tfrecord')


# ## Run detections
# We are all set to run detections. The detections will be output to a separate TFRecord file. We will use the model we trained in [this kernel](https://www.kaggle.com/mistag/tensorflow-object-detection-on-custom-data):

# In[ ]:


get_ipython().system('ls /kaggle/input/tensorflow-object-detection-on-custom-data/trained')


# In[ ]:


get_ipython().run_cell_magic('capture', 'cap_out --no-stderr', '!python object_detection/inference/infer_detections.py \\\n  --input_tfrecord_paths=/kaggle/working/butterflies_960p.tfrecord \\\n  --output_tfrecord_path=/kaggle/working/detections.tfrecord \\\n  --inference_graph=/kaggle/input/tensorflow-object-detection-on-custom-data/trained/frozen_inference_graph.pb \\\n  --discard_image_pixels')


# In[ ]:


get_ipython().system('ls -al /kaggle/working/*.tfrecord')


# ## Convert detections into DataFrame
# To make it easier to process the detections, we convert `detections.tfrecord` into a Pandas frame.

# In[ ]:


tf.enable_eager_execution() # only for TF1.x
labels = pd.read_pickle('/kaggle/input/starter-arthropod-taxonomy-orders-data-exploring/ArTaxOr_labels.pkl')
detections = tf.data.TFRecordDataset('/kaggle/working/detections.tfrecord')
pdf = pd.DataFrame(columns=['score', 'label_idx', 'left', 'top', 'right', 'bottom', 'frame'])
for record in detections:
    det = tf.train.Example.FromString(record.numpy())
    height = det.features.feature['image/height'].int64_list.value[0]
    width = det.features.feature['image/width'].int64_list.value[0]
    score = det.features.feature['image/detection/score'].float_list.value
    score = [x for x in score if x >= 0.60]
    for i in range(len(score)):
        pdf=pdf.append({'score': score[i],
                        'label_idx': det.features.feature['image/detection/label'].int64_list.value[i],
                        'left': det.features.feature['image/detection/bbox/xmin'].float_list.value[i],
                        'top': det.features.feature['image/detection/bbox/ymin'].float_list.value[i],
                        'right': det.features.feature['image/detection/bbox/xmax'].float_list.value[i],
                        'bottom': det.features.feature['image/detection/bbox/ymax'].float_list.value[i],
                        'frame': det.features.feature['video/frame'].int64_list.value[0]}, ignore_index=True)


# In[ ]:


pdf.head()


# ## Render video
# Finally we can render a new video with detections.

# In[ ]:


fontname = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(fontname, 15) if os.path.isfile(fontname) else ImageFont.load_default()

def frame_idx(img, fcnt):
    draw = ImageDraw.Draw(img)
    xres, yres = img.size[0], img.size[1]
    txt='Frame:{0:04d}'.format(fcnt)
    draw.text((5, yres-25), txt, font=font, fill='white')
    
def bbox(img, xmin, ymin, xmax, ymax, color, label, score):
    draw = ImageDraw.Draw(img)
    xres, yres = img.size[0], img.size[1]
    box = np.multiply([xmin, ymin, xmax, ymax], [xres, yres, xres, yres]).astype(int).tolist()
    txt = " {}: {}%" if score >= 0. else " {}"
    txt = txt.format(label, round(score, 1))
    ts = draw.textsize(txt, font=font)
    draw.rectangle(box, outline=color, width=3)
    if len(label) > 0:
        if box[1] >= ts[1]+3:
            xsmin, ysmin = box[0], box[1]-ts[1]-3
            xsmax, ysmax = box[0]+ts[0]+2, box[1]
        else:
            xsmin, ysmin = box[0], box[3]
            xsmax, ysmax = box[0]+ts[0]+2, box[3]+ts[1]+1
        draw.rectangle([xsmin, ysmin, xsmax, ysmax], fill=color)
        draw.text((xsmin, ysmin), txt, font=font, fill='white')


# In[ ]:


get_ipython().run_cell_magic('time', '', "vout = cv2.VideoWriter('/kaggle/working/butterflies.mp4', 0x7634706d, 30, (width,height))\ndataset = tf.data.TFRecordDataset(trec)\nfor img_example in dataset:\n    img_parsed = tf.train.Example.FromString(img_example.numpy())\n    fcnt=img_parsed.features.feature['video/frame'].int64_list.value[0]\n    img_encoded=img_parsed.features.feature['image/encoded'].bytes_list.value[0]\n    img = Image.open(BytesIO(img_encoded))\n    fdet = pdf[pdf.frame == fcnt]\n    for i in range(len(fdet)):\n        bbox(img, fdet.left.iloc[i], fdet.top.iloc[i], fdet.right.iloc[i], fdet.bottom.iloc[i], \n             labels.color.iloc[int(fdet.label_idx.iloc[i])-1], \n             labels.name.iloc[int(fdet.label_idx.iloc[i])-1], \n             int(fdet.score.iloc[i]*100.))\n    frame_idx(img, fcnt) # add frame counter\n    buff=np.frombuffer(img.tobytes(), dtype=np.uint8)\n    buff=buff.reshape(height, width, 3)\n    buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)\n    vout.write(buff)\nvout.release()")


# In[ ]:


get_ipython().run_line_magic('cd', '/kaggle/working')
get_ipython().system('rm -f *.proto')
get_ipython().system('ls -l *.mp4')


# I have not figured out how to display the video directly from the notebook, so I had to put it on YouTube first. And here is the result:

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo('ZjDRcgXQsR0', width=830, height=467)


# ## Summary
# In this notebook we have seen how to run object detection on a video, and then render a new video with detections. The trained model we used needs some improvement it looks. Although objects are detected pretty well, the object class is wrong quite often.
