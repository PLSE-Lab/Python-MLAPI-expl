#!/usr/bin/env python
# coding: utf-8

# # Novice Approach to Deepfake Detection Challenge

# ## Imports

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


import cv2
import json
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from skimage.measure import compare_ssim
from sklearn.metrics import log_loss, classification_report
import subprocess
import sys
import tensorflow as tf
import time
from tqdm.notebook import tqdm


# ## Variables

# In[5]:


KAGGLE = os.getenv('KAGGLE_KERNEL_RUN_TYPE') != None

PATH = [Path('/home/jupyter/.fastai/data/dfdc'),
            Path('/kaggle/input/deepfake-detection-challenge')][KAGGLE]

VIDEOS = [PATH/'test/videos', PATH/'test_videos'][KAGGLE]

DATASET_DIR = [Path('./dfdc-na'),
               Path('/kaggle/input/dfdc-na')][KAGGLE]

sys.path.append(str(DATASET_DIR))

TPU_NAME = 'dfdc-2'

# Scale factor applied to video frames
FRAME_SCALE = 1.0

# Size of faces to be returned from videos
IMAGE_SIZE = (260, 260)

BUCKET = None
if BUCKET is not None:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.get_bucket(BUCKET)

TFRECORD_DIR = [f'gs://{BUCKET}', '.'][1]
TFRECORD_PREFIX = 'na_videos_faces'

#this is the meta data from the json files with a few extra columns appended
df_meta = pd.read_pickle(f'{DATASET_DIR}/df_meta.pkl')
df_meta.cluster = df_meta.cluster.astype(np.int64)

#these are the bounding box predictions from the detected faces in the original videos
df_boxes = pd.read_pickle(f'{DATASET_DIR}/df_boxes.pkl')
df_boxes['zip_no'] = df_boxes.index.get_level_values(0).map(df_meta.zip_no)
df_boxes['cluster'] = df_boxes.index.get_level_values(0).map(df_meta.cluster)

# this filters the boxes to only those of high probability for the purpose of creating
# the tranining data
df_box_probs = df_boxes.reset_index().groupby(['filename', 'box_idx']).agg({'frame_idx': 'count', 'prob': 'mean'})
df_box_probs = df_box_probs[(df_box_probs.prob > 0.98) & (df_box_probs.frame_idx > 15)]

# This filters  the test videos to fake videos and their corresponding original videos
# in test set so that the book runs with access to only the test videos.
test_names = [t.name for t in VIDEOS.glob('*.mp4')]

df_test_meta = df_meta.loc[df_meta.index.isin(test_names)
                           & df_meta.original.isin(test_names)
                           & (df_meta.original.isin(df_box_probs.index.get_level_values(0)))].copy()

# this is a simplistic split for illustrative purposes only
df_test_meta['test_split'] = np.random.random(len(df_test_meta)) > 0.8
df_test_meta.test_split = df_test_meta.test_split.map({True: 'valid', False: 'train'})


# ## Functions 

# In[6]:


def crop(ndarray, box):
    """Crops an ndarray to a given bounding box."""
    new_nd_array = np.zeros(ndarray.shape)
    left, upper, right, lower = tuple(map(lambda x: int(max(0, x)), box))
    return ndarray[slice(upper, lower),slice(left, right),:]

def iou(pred_record):
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(pred_record.mask_actual,
                   pred_record.mask_pred)
    return m.result().numpy()

def get_fake_faces(vid_name, idxs=None, n_frames=None, orig_vid=None, box_idx=0):
    """Returns records of faces from fake videos."""
    if orig_vid is None:
        orig_name = df_meta.loc[vid_name].original
        orig_frames, orig_idxs = get_frames(orig_name, n_frames=n_frames, idxs=idxs)
    else:
        orig_name, orig_frames, orig_idxs = orig_vid

    vid_frames, idxs = get_frames(vid_name, idxs=orig_idxs)
    orig_boxes = df_boxes.loc[orig_name]
    orig_boxes = orig_boxes.loc[orig_boxes.index.get_level_values(1)==box_idx]

    # makes it possible to get boxes from frames different than the
    # original ones that are in the df_boxes dataframe
    vid_boxes = pd.merge_asof(pd.Series(idxs, name='new_frame_idx'),
                     orig_boxes, left_on='new_frame_idx',
                     right_on='frame_idx', direction='nearest').set_index('new_frame_idx')

    faces_list = []
    for frame_idx, frame, orig_frame in zip(idxs, vid_frames, orig_frames):            
        orig_box = scale_box(vid_boxes.loc[frame_idx].square_box)
        box = scale_center_box(orig_box)

        face = crop(frame, box)
        orig_face = crop(orig_frame, box)

        thresh = get_thresh(face, orig_face)

        face_mask = resize_pad_square(thresh, IMAGE_SIZE[0])
        face_mask = np.ndarray.astype(face_mask > 1, np.bool)
        mask_rle = dense_to_brle(face_mask.flatten())
        fake_face = resize_pad_square(face, IMAGE_SIZE[0])
        real_face = resize_pad_square(orig_face, IMAGE_SIZE[0])

        faces_list.append({'face': fake_face,
                           'orig_face': real_face,
                           'mask_rle': mask_rle,
                           'mask': face_mask,
                           'frame_idx': frame_idx,
                           'scale': FRAME_SCALE,
                           'name': vid_name,
                           'orig_name': orig_name,
                           'box': list(box),
                           'box_idx': box_idx,
                           'label': df_meta.loc[vid_name].label,
                           'label_code': df_meta.loc[vid_name].label_code,
                           'cluster': df_meta.loc[vid_name].cluster,
                           'zip_no': df_meta.loc[vid_name].zip_no
                          })

    return faces_list

def get_frames(vid_name, n_frames=20, start_frame=0, end_frame=None, idxs=None):
    """Get frames from a given video name."""
    vid_path = str(VIDEOS/vid_name)

    v_cap = FileVideoStream(vid_path, n_frames=n_frames,
                            start_frame=start_frame, end_frame=end_frame,
                            idxs=idxs,transform=transform).start()
    frames = []
    while v_cap.running():
        frame, idx = v_cap.read()
        frames.append(frame)
    return frames, v_cap.idxs

def get_real_faces(vid_name, n_frames=None, idxs=None, box_idx=0):
    """Returns records of faces from real videos."""
    vid_frames, idxs = get_frames(vid_name, idxs=idxs, n_frames=n_frames)
    orig_boxes = df_boxes.loc[vid_name]
    orig_boxes = orig_boxes.loc[orig_boxes.index.get_level_values(1)==box_idx]

    # makes it possible to get boxes from frames different than the
    # original ones that are in the df_boxes dataframe
    vid_boxes = pd.merge_asof(pd.Series(idxs, name='new_frame_idx'),
                     orig_boxes, left_on='new_frame_idx',
                     right_on='frame_idx', direction='nearest').set_index('new_frame_idx')

    faces_list = []
    for frame_idx, orig_frame in zip(idxs, vid_frames):
        box = scale_box(vid_boxes.loc[frame_idx].square_box)
        box = scale_center_box(box)
        
        face = crop(orig_frame, box)
        face = resize_pad_square(face, IMAGE_SIZE[0])
        mask_rle = np.array((0, IMAGE_SIZE[0] * IMAGE_SIZE[1]), np.int64)
        
        faces_list.append({'face': face,
                           'mask_rle': mask_rle,
                           'frame_idx': frame_idx,
                           'scale': FRAME_SCALE,
                           'name': vid_name,
                           'orig_name': vid_name,
                           'box': list(box),
                           'box_idx': box_idx,
                           'label': df_meta.loc[vid_name].label,
                           'label_code': df_meta.loc[vid_name].label_code,
                           'cluster': df_meta.loc[vid_name].cluster,
                           'zip_no': df_meta.loc[vid_name].zip_no
                          })

    return faces_list

def get_thresh(frame, orig_frame, min_diff=210):
    """Returns a thresholded difference between two rgb images."""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    orig_frame_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(frame_gray, orig_frame_gray, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, min_diff, 255, cv2.THRESH_BINARY_INV)[1]
    return thresh

def plot_learning_curves(history):
    """Plots losses and metrics from keras model.fit object."""
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    
def print_output(output):
    """Prints output from string."""
    for l in output.split('\n'):
        print(l)
        
def print_pred_metrics(label_actual, label_pred, pred_prob):
    """Prints prediction evaluation metrics and report."""
    print(f'log loss: {log_loss(label_actual, pred_prob.clip(0.001, 0.999))}')
    print(classification_report(label_actual, label_pred))
    print(pd.crosstab(label_actual, label_pred, margins=True))

def resize_pad_square(ndarray, size, box=None):
    """Crops ndarray to box if provided, pads square and then resizes to square size."""
    if box != None:
        left, upper, right, lower = box
        new_ndarray = ndarray.copy()[max(upper,0):lower, max(left,0):right]
    else:
        new_ndarray = ndarray.copy()    

    # pad square
    r, c = new_ndarray.shape[:2]
    left_pad = top_pad = right_pad = bottom_pad = 0
    
    if r < c:
        top_pad = int((c - r) // 2)
        bottom_pad = int(c - r - top_pad)
    if c < r:
        left_pad = int((r - c) // 2)
        right_pad = int(r - c - left_pad) 
    
    if sum((left_pad, top_pad, right_pad, bottom_pad)) > 0:
        pads = ((top_pad, bottom_pad), (left_pad, right_pad), (0,0))

        if np.ndim(new_ndarray) == 2:
            pads = pads[:2]            
            
        new_ndarray = np.pad(new_ndarray, pads, mode='reflect')
    
    # resize
    shrink = min(new_ndarray.shape[:2]) < size
    interpolation = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    
    resized_array = cv2.resize(new_ndarray, (size,size), interpolation=interpolation)

    return resized_array
        
def run_command(command):
    """Runs command line command as a subprocess returning output as string."""
    STDOUT = subprocess.PIPE
    process = subprocess.run(command, shell=True, check=False,
                             stdout=STDOUT, stderr=STDOUT, universal_newlines=True)    
    return process.stdout
    
def scale_box(box):
    """Used to adjust scale of box to apply to frame scaled by specified factor."""
    return tuple(map(lambda x: int(x * FRAME_SCALE), box))
    
def scale_center_box(box, scale=1.3):
    """Scales box by specified factor about its center."""
    left, upper, right, lower = box
    row_delta = int((lower - upper) * (1 - scale) / 2)
    col_delta = int((right - left) * (1 - scale) / 2)
    left += col_delta
    right -= col_delta
    upper += row_delta
    lower -= row_delta
    return left, upper, right, lower
        
def show_images(imgs, titles=None, hw=(4,6), rc=(3,3)):
    """Show list of images with optiona list of titles."""
    h, w = hw
    r, c = rc
    fig=plt.figure(figsize=(w*c, h*r))
    gs1 = gridspec.GridSpec(r, c, fig, hspace=0.2, wspace=0.05)
    for i in range(r*c):
        img = imgs[i].squeeze()
        ax = fig.add_subplot(gs1[i])
        if titles != None:
            ax.set_title(titles[i], {'fontsize': 10})
        plt.imshow(img)
        plt.axis('off')
    plt.show()

def show_pred(r):
    """Show segmentation prediction."""
    print(r.name, f'label: {r.label}', f'prob: {r.prob:0.4f}', f'iou: {r.iou:0.4f}')
    show_images([r.face, r.mask_actual, r.mask_pred],
                ['face', 'actual mask', 'predicted_mask'], hw=(3,3), rc=(1,3))
    
def transform(frame):
    """Transform applied to frames returned from FileVideoStream."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h = int(frame.shape[0] * FRAME_SCALE)
    w = int(frame.shape[1] * FRAME_SCALE)    
    frame = cv2.resize(frame, (w,h))
    return frame


# ## More Imports 

# In[9]:


if KAGGLE:
    for package in ['classification_models', 'segmentation_models',
                    'efficientnet', 'run_length_encoding']:
        print_output(run_command(f"pip install {str(DATASET_DIR/package)}"))


# In[10]:


sys.path.append(str(DATASET_DIR))
sys.path.append(str(DATASET_DIR/'vggface2_Keras/src'))

from efficientnet import tfkeras as efn
from filevideostream_na import FileVideoStream
from rle.tf_impl import brle_to_dense
from rle.np_impl import dense_to_brle, rle_length

import segmentation_models as sm
sm.set_framework('tf.keras')

from vggface2_Keras.src.model import Vggface2_ResNet50


# # Overview

# I spent more hours than I'd care to disclose competing in this challenge. I wasn't able to create a model on my own that produced a good leaderboard score. I did, however, learn a ton about creating data pipelines, Tensorflow and TPUs and trained a lot of models. In an effort to solidify my learnings and in hopes of passing along something useful from my trials and tribulations, this book provides a few details on my approach to key parts of the challenge.
# 
# I thought using TPUs would potentially be a competitive advantage in this competition given the large size of the dataset. Thank you Google for the generous grant, which once getting past the learning curve of tfrecords, definitely led to faster training and idea-iteration.
# 
# **training and validation splits**
# 
# Developing good training and validation datasets were important elements of this challenge. I tried to develop a training set comprised of correctly labeled faces, meaning that if they were labeled as fake, they in fact had altereed pixels in them (I didn't do anything with the audio). For the training and validation splits, I also wanted to create sets that were independent with respect to actor. I used some simplistic approaches early on and then a clustering analysis provided in [this kernel](https://www.kaggle.com/hmendonca/proper-clustering-with-facenet-embeddings-eda).
# 
# **tfrecords**
# 
# The first section below details my tfrecod set up, which ended up being really cool. [Tfrecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) are basically serialized training examples that can be read back into [tf.data.Datasets](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) super efficiently. In order to use TPUs, you create tfrecords and store them on gcs so they can be fed to the TPUs. For my tfrecord set up, I created a single tfrecord for each face detected in each video, storing each face from each frame of the video as a sequence.
# 
# **datasets**
# 
# The second section focuses on the datasets that get created from the tfrecords. I really enjoyed the [tf.data.Datasets](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) api. It made it really easy to do thinks like balancing the datasets between fake and real examples, perform augmentation and preprocessing. It also made it possible to use a single set of tfrecords to feed single face cnn, sequence and segmentation models.
# 
# **models**
# 
# The third section provides basic versions of the three types of models I used. There is really nothing special in the first two, a single face cnn and an multi-face LSTM. I used the excellent Efficientnet implementatios in the [qubvel/efficientnet](https://github.com/qubvel/efficientnet) repo for the cnn. I also tried using a pretrained resnet50 model trained on the vggface2 dataset from [WeidiXie/
# Keras-VGGFace2-ResNet50](https://github.com/WeidiXie/Keras-VGGFace2-ResNet50), unfreezing the weights in the last convolution block, as illustrated in the LSTM example.
# 
# The third model I tried was the one I was most optimistic about, but in the end couldn't make it work. The idea was to create segmentation masks showing the altered pixels in the fake faces by thresholding the difference between the original and fake frames and then train a segmentation model to detect the altered pixels. My thinking was that this was valuable additional information the model could use to develop relationships beyond simple binary classification. I used the excelellent implementation of a Unet with Efficientnet encoder in the [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models) repo. I added a few additional layers after the segmentation output to calculate the binary classification loss and then weighted the segmentation loss and the binary classification loss. I thought this might actually help get to a better binary classification model by focusing the model's attention on the altered pixels and that it would also be useful to get predictions of which pixels were fake in addition a prediction of binary classification.

# In[6]:


fake_name = df_test_meta[df_test_meta.label == 'FAKE'].index[0]

frame_idxs = [0, 30, 60, 90]
faces = get_fake_faces(fake_name, idxs=frame_idxs)

face_list = []
title_list = []
for n in range(len(frame_idxs)):
    face = faces[n]
    keys = ['orig_face', 'face', 'mask']
    title_list.extend([f"fram: {face['frame_idx']}, box: {face['box_idx']} - {key[:4]}" for key in keys])
    face_list.extend([faces[n][key] for key in keys])

show_images(face_list, title_list, hw=(3,3), rc=(2,6))


# **predictions**
# 
# The last section details my approach to predictions. A couple of things worth noting. First off, the determination of a video's classification from potentially multiple faces from multiple frames was not as straightforward as simply taking the average of all of the predictions for a given video. The predictions roll up from a face detected in a frame through multiple frames and ultimately lead to a prediction for a video. I ended up pulling faces from 16 frames per video and then deciding that a video would be predicted as fake if at least three faces were predicted as such with a probability of at least 80%. I then converted that back into a probability that a video was fake for purposes of the log loss evaluation by dividing the actual number of faces over 80% by 4.5, such that a video with three frames over the 80% threshold would be considered to be fake with 67% probability and a video with 1 face over 80% would be considered to have a 22% probability of being fake.
# 
# I only provide predictions for the segmentation model. Again, I couldn't make the results work, but the predictions are really cool in a lot of cases, highlighting where the faces are fake in addition to predicting whether the video is fake.
# 
# **results and key learnings**
# 
# The best leaderboard score I could produce was in excess of 0.5. I was really committed to not just copying kernels with better scores and running them, but really building my own pipelines and models. I would do this differently going forward.
# 
# I tried a ton of different models and datasets. I tried tightly cropped faces versus loosely cropped and even whole frame models. I tried single frame models with only one fake video per original. I tried overweighting the real videos. I tried sequences of frames from short durations of the videos as well as spread throughout the videos. I tried different pretrained weights, including vggface, vggface2 and imagenet. I tried different cnn models, from Resnet50 to different Efficientnets. I tried training my own version of an Efficientnet model on the vggface2 dataset. I tried large batch sizes, small batch sizes, freezing weights and unfreezing weights. I could never seem to make anything produce a good validation result.
# 
# On the one hand, all of those iterations were incredibly useful in just learning how to go from raw data to training a model and making predictions. On the other hand, clearly other people were figuring out how to make models work. There were obviously things I was missing or doing incorrectly that were preventing me from achieving good results. With that in mind, here is what I would have done differently in hindsight.
# 
# 1. I would have taken a **hybrid approach to learning from other kernels**, taking down the ones with better scores, really dissecting them to understand how they were working, and then endeavoring to recreate my own versions, especially as a novice.
# 2. I would have spent more time **making sure that I had a really good validation set**. I used a couple of simplistic splits early on and then someone else's cluster analysis to approximate actor mutual exclusivity, but I was never really confident that I had a great validation set. Consequently as I was training models and watching validation results closely, I always had some doubt about whether I could trust them. Given the number of decision to make, it is really important to be able to trust your validation results, especially given the limit of two leaderboard submissions per day.
# 3. I would do **more testing of my data pipeline**. There was a lot going on between the mp4 video files and the batches of training examples being fed into the models. My initial operating procedure was to write the code for the dataset that I wanted to create, then print out a few batches and if they looked basically right, send them through the model. That obviously leaves a lot of uncertainty as to whether the data that you think is running through the model is actually running through the model. Later in my develop cycle, I started doing more quantitative testing. Real-fake, original video and actor distribution within the dataset all ended up being important considerations, so I would cycle through the dataset to check the actual distribution within batches and across the entire dataset. The pipeline itself also had a lot of moving parts, that frequently broke as I was adding new features and trying new ideas. Having something akin to unit tests would have been much more efficient in the end.
# 4. I would **try joining a team**. Going it alone is obviously beneficial from the standpoint of learning by having to do everything, but as a complete novice, there is a ton to learn and being able to learn some of the basics from people who have done it successfully before is ultimately likely a much faster way to get up the learning curve.

# # Training and validation splits 

# Creating a good training dataset turned out to be a non-trivial exercise, even for models based on frames without detecting faces. The first challenge was that the meta data didn't identify the actor in the original video, so it wasn't possible to create training and validation splits that were mutually exclusive with respect to actors based on the unique original video name. I ended up using the clustering analysis done in [this kernel](https://www.kaggle.com/hmendonca/proper-clustering-with-facenet-embeddings-eda), which essentially used a face recognition model to get an embedding for each original video and then performed an analysis to assign each to one of 700 some odd clusters. The histogram below shows the number of clusters containing the number of videos specified on the x axis.

# In[95]:


df_clusters = pd.read_feather(str(DATASET_DIR/'face_clusters.feather'))
df_clusters.cluster.value_counts().hist(bins=list(range(0,100)));


# The median number of videos per cluster was 10 and there was one cluster with 464 entries in it. I ended up selecting a validation set of approximately 1900 original videos by selecting all clusters with less than 15, but more than five videos in it. Each of the rows below is a separate cluster with each column in each row a face from a different video in the cluster. The clustering looked like it did a decent job of identifying the actor in each video.

# In[8]:


face_list = []
title_list = []
rows, cols = (6,8)
for c in range(rows):
    cluster = (df_test_meta[df_test_meta.label == 'REAL']
               .groupby(['cluster'])
               .count()['label_code']
               .sort_values(ascending=False)
              ).index[c]

    for v in range(cols):
        vid_name = df_test_meta.loc[(df_test_meta.cluster == cluster) & (df_test_meta.label == 'REAL')].index[v]
        faces = get_real_faces(vid_name, idxs=[0])
        face = faces[0]['face']
        title = f'{cluster}: {vid_name[:-4]}'
        title_list.append(title)
        face_list.append(face)
        
show_images(face_list, title_list, hw=(2,2), rc=(rows,cols))


# Another set of challenges to establishing a high quality training dataset were related to the labels of the videos. Not every video labeled as fake actually had altered pixels. Furthermore, some fake videos had some modified frames, and some that were not modified. Lastly, in videos that had more than one face, not all of the faces had modified pixels.
# 
# The faces from the video below illustrate these challenges, where the face from one of the frames has been modified, but the other hasn't. This set of faces also illustrates another, more fundamental challenge that I didn't solve, that of grouping faces of the same person together in sequence. You can see on the first row with faces on the left from box 0 on frame 0 and faces on the right from box 0 on frame 180. In videos with more than one face. Time permitting, I would have run each face through a face identification model, getting an embedding back and then group faces across frames by minimizing the distance from the embedding for the faces from the first frame.

# In[9]:


orig_name = df_box_probs[df_box_probs.index.get_level_values(0)
                         .isin(df_test_meta.index) & (df_box_probs.index.get_level_values(1) == 1)].index[2][0]

for box_idx in range(2):
    fake_vid_name = df_test_meta[(df_test_meta.label == 'FAKE') & (df_test_meta.original == orig_name)].index[0]
    frame_idxs = [0, 180]
    faces = get_fake_faces(fake_vid_name, idxs=frame_idxs, box_idx=box_idx)

    face_list = []
    title_list = []
    for n in range(len(frame_idxs)):
        face = faces[n]
        keys = ['orig_face', 'face', 'mask']
        title_list.extend([f"fram: {face['frame_idx']}, box: {face['box_idx']} - {key[:4]}" for key in keys])
        face_list.extend([faces[n][key] for key in keys])

    show_images(face_list, title_list, hw=(3,3), rc=(1,6))


# # tfrecords

# [Tfrecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) are serialized records of training examples that are used to create datasets that can be read super efficiently by the TPUs from cloud storage. They are essentially serialized dicts of training examples that get read and parsed to create datasets. Tensorflow recommended creating multiple approximately 150MB files so that they could be read in parallel.
# 
# The features have to be stored as lists of either `tf.int64`, `tf.float32` or `tf.string`. The cool thing was that it was possible to store multiple frames or faces in each record and then specify how many to read back at dataset creation time. This made it possible to run both single frame and sequence models from the same dataset.
# 
# I also stored the run length encoding of the mask for each of the fake video faces so that segmentation models could be run from the same set of tfrecords.
# 
# To create the tfrecords, I got all of the bounding boxes for all of the original videos using [facenet-pytorch](https://github.com/timesler/facenet-pytorch), stored them in a dataframe and then looped through each of the corresponding fake videos to grab the faces from them.
# 
# I also used a modified version of `imutils` `FileVideoStream` class to to pull video frames that took either a number of frames or a list of indexes of the frames to speed things up by not reading all of the frames in each video. Another time saver was setting the bounding boxes up to use the boxes from the nearest frame for which a bounding box had previously been detected.

# ## Serialize Records

# This is just boilerplate to create serialized features of the three permissible data types for the tfrecords.

# In[10]:


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    if type(value) != type(list()):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    if type(value) != type(list()):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    
    if type(value) != type(list()):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# This stores the records created from the `get_faces` functions in a single tfrecord example. Notice how multiple faces, masks and frame indexes are serialized.

# In[11]:


def serialize_example(records):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    
    meta = records[0]
    shape = list(meta['face'].shape)
    
    feature = {
        'name':       _bytes_feature(tf.compat.as_bytes(meta['name'])),
        'original':   _bytes_feature(tf.compat.as_bytes(meta['orig_name'])),
        'label':      _bytes_feature(tf.compat.as_bytes(meta['label'])),
        'label_code': _int64_feature(meta['label_code']),
        'scale':      _float_feature(meta['scale']),
        'shape':      _int64_feature(shape),
        'face':       _bytes_feature([tf.io.encode_jpeg(np.array(record['face'])).numpy() for record in records]),
        'mask':       _bytes_feature([record['mask_rle'].tobytes() for record in records]),
        'idx':        _int64_feature([record['frame_idx'] for record in records]),
        'zip_no':     _int64_feature(meta['zip_no']),
        'cluster':    _int64_feature(meta['cluster'])
    }

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


# In[12]:


fake_records = get_fake_faces(fake_name, n_frames=8)
fake_example = serialize_example(fake_records)


# In[13]:


real_records = get_real_faces(orig_name, n_frames=8)
real_example = serialize_example(real_records)


# ## Parse Examples 

# This function is set up to return a parsing function that returns a parsed record with a specified number of frames, with or without corresponding mask.

# In[14]:


def get_parse_face_with_mask_examples_fn(take_n_faces, incl_mask=False):
    def parse_face_examples(examples):          
        features = {
            'name':       tf.io.FixedLenFeature([], tf.string),
            'original':   tf.io.FixedLenFeature([], tf.string),
            'label':      tf.io.FixedLenFeature([], tf.string),
            'label_code': tf.io.FixedLenFeature([], tf.int64),
            'scale':      tf.io.FixedLenFeature([], tf.float32),
            'shape':     tf.io.FixedLenFeature([3], tf.int64),
            'face':     tf.io.VarLenFeature(tf.string),
            'mask':     tf.io.VarLenFeature(tf.string),
            'idx':       tf.io.VarLenFeature(tf.int64),
            'zip_no': tf.io.FixedLenFeature([], tf.int64),
            'cluster': tf.io.FixedLenFeature([], tf.int64),
        }

        parsed = tf.io.parse_example(examples, features)
        
        # make sure label code is consistent with label
        label_code = tf.cast(tf.math.equal(parsed['label'], tf.convert_to_tensor('FAKE', tf.string)), tf.int64)
        parsed['label_code'] = label_code
    
        def brle_to_mask(rle_bytes):
            mask = tf.io.decode_raw(rle_bytes, tf.int64)
            mask = tf.cast(brle_to_dense(mask), tf.uint8)
            mask = tf.reshape(mask, (parsed['shape'][:2]))
            return mask
                                     
        face = tf.map_fn(lambda x: tf.io.decode_jpeg(x), parsed['face'].values[:take_n_faces], tf.uint8)
        parsed['face'] = face

        if incl_mask:
            mask = tf.map_fn(lambda x: brle_to_mask(x), parsed['mask'].values[:take_n_faces], tf.uint8)
            parsed['mask'] = mask

        else:
            del parsed['mask']

        
        idx = tf.sparse.to_dense(parsed['idx'])
        parsed['idx'] = idx[:take_n_faces]

        return parsed

    return parse_face_examples


# These records have gone full cycle, from cropped faces from video frames, to serialized tfecords and then parsed back out again.

# In[92]:


show_n = 8
parse_fn = get_parse_face_with_mask_examples_fn(show_n, incl_mask=True)
print(parse_fn(fake_example).keys())

for example in [fake_example, real_example]:
    parsed_example = parse_fn(example)
    print(parsed_example['name'].numpy(),
          parsed_example['label'].numpy(),
          f"cluster: {parsed_example['cluster'].numpy()}")
    show_images(parsed_example['face'].numpy()[:show_n],
                parsed_example['idx'].numpy()[:show_n].tolist(), hw=(2,2), rc=(1,show_n))
    show_images(parsed_example['mask'].numpy()[:show_n], hw=(2,2), rc=(1,show_n))


# ## Create Record Files

# The tfrecord files are created separately for real and fake videos and for training and validation splits. It was possible to filter tfrecords into training and validation splits after they are parsed, on cluster or zip file number, for example, but that ended up being much slower than doing the splits in writing the files. The real and fake are written separately so that they can be balanced efficiently for training and also because records for the fake videos can be written by looping through fakes for each real video so that frames for each real video only have to be pulled twice, instead of for each fake video (since I create masks from thresholded differences for each face).
# 
# This function just chunks up the original video file names. It includes an argument for number of files, for testing or distributing to other machines.

# In[16]:


def get_chunks(split, label, n_per_chunk=20, n_files=None):
    """Returns a chunks of original video names. Specify n_files > 0 to limit number of files written."""
    
    orig_names = df_test_meta[df_test_meta.test_split == split].original.unique()
    n_chunks = len(orig_names) // n_per_chunk  
    orig_chunks = list(enumerate(np.array_split(np.array(orig_names),n_chunks)))
    sample_chunks = orig_chunks[:n_files]

    print(f'Total {len(orig_chunks)} chunks with videos from {len(orig_chunks[0][1])} original videos in each.')
    print(f'sample_chunks {len(sample_chunks)} chunks with videos from {len(sample_chunks[0][1])} original videos in each.')
    
    return sample_chunks


# This function is where the tfrecords are actually written. It loops the the original video names, pulls the high probability face boxes for each and then creates records for each fake video. It takes an argument for max number of fake videos if you want to limit the number of fake videos per original video in the records.

# In[17]:


def write_tfrecord_files(split, label, chunks, max_fakes=None):
    start = time.perf_counter()
    error_vids = []

    for chunk_idx, chunk in chunks:
        filename = f'{TFRECORD_PREFIX}_{split}_{label}_{chunk_idx:05d}.tfrecords'
        record_file = f'{TFRECORD_DIR}/{filename}'

        with tf.io.TFRecordWriter(record_file) as writer:

            for orig_name in tqdm(chunk):
                if label == 'FAKE':
                    orig_frames, orig_idxs = get_frames(orig_name, n_frames=20)
                    fake_vid_names = df_test_meta[(df_test_meta.label == 'FAKE')
                                                 & (df_test_meta.original == orig_name)].index[:max_fakes]
                    
                    for vid_name in fake_vid_names:
                        for box_idx in df_box_probs.loc[orig_name].index:
                            try:
                                fake_records = get_fake_faces(vid_name,
                                                              orig_vid=(orig_name, orig_frames, orig_idxs),
                                                              box_idx=box_idx)
                                writer.write(serialize_example(fake_records))
                            except:
                                print(vid_name)
                                error_vids.append(vid_name)
                else:
                    for box_idx in df_box_probs.loc[orig_name].index:
                        for box_idx in df_box_probs.loc[orig_name].index:
                            try:
                                real_records = get_real_faces(orig_name, n_frames=20, box_idx=box_idx)
                                writer.write(serialize_example(real_records))
                            except:
                                print(orig_name)
                                error_vids.append(orig_name)

        print(f'{filename} written to {TFRECORD_DIR}')

    duration = time.perf_counter() - start
    n_videos = len(np.concatenate([x[1] for x in chunks]))

    print(f'\nProcessed {n_videos} original videos in {duration:0.1f} seconds - avg of {duration / n_videos:0.1f} seconds per video.')
    print(f'Estimated duration for 19,000 original videos: {duration / n_videos * 19000 / 3600:0.1f} hours.')

    total_size = 0.
    
    if TFRECORD_DIR[:2] == 'gs':
        for b in bucket.list_blobs(prefix=TFRECORD_PREFIX):
            total_size += b.size
            
    else:
        for p in Path(TFRECORD_DIR).glob(f'{TFRECORD_PREFIX}*'):
            total_size += p.lstat().st_size
    #     print(p.name, f'{p.lstat().st_size / 1e6:.1f} MB')

    print(f'\nTotal disk space of {total_size / 1e6:0.1f} MB for {n_videos} videos - avg of {total_size / n_videos / 1e6:0.1f} MB per video.')
    print(f'Estimated disk space for 19,000 original videos: {total_size / n_videos * 19000 / 1e9:0,.1f} GB.')


# In[18]:


if True:
    for s in ['train', 'valid']:
        for l in ['FAKE', 'REAL']:
            print('\n\033[1m', s,l, '\033[0m')
            chunks = get_chunks(s, l, 10, n_files=2)
            write_tfrecord_files(s, l, chunks)


# # Datasets

# These dataset functions are set up to be able to create examples with single faces or multiple faces with and without masks. Augmentation includes random horizontal flip and resize with random crop. Preprocessing includes normalization for three different sets of parameters as well as resizing. Both augmentation and preprocessing are applied to both faces and corresponding masks if they are included in the dataset.

# In[19]:


AUTO = tf.data.experimental.AUTOTUNE


# In[20]:


def flatten_frames(example):
    n_frames = tf.shape(example['face'])[0]
    for k in [k for k in example.keys() if k not in ['face', 'mask', 'idx', 'shape']]:
        example[k] = tf.repeat(example[k], n_frames)
    example['shape'] = tf.tile(tf.expand_dims(example['shape'], axis=0), (n_frames,1))
    return example

def get_augment_fn(zoom=1.0, incl_mask=False):
    def augment(batch):
        new_batch = batch.copy()
        del batch

        face = new_batch['face']
        face = tf.image.central_crop(face, zoom)
        shape = tf.shape(face)

        flip = tf.random.uniform(()) > 0.5

        if flip: 
            face = tf.image.flip_left_right(face)

        resize_pix = 30
        face = tf.image.resize(face, shape[-3:-1] + resize_pix)
        upper = tf.random.uniform((), 0, resize_pix, tf.int32)
        left = tf.random.uniform((), 0, resize_pix, tf.int32)
        face = face[:, upper:shape[1]+upper, left:shape[2]+left, :]

        new_batch['face'] = tf.cast(face, tf.uint8)

        if incl_mask:
            mask = tf.expand_dims(new_batch['mask'], axis=-1)
            mask = tf.image.central_crop(mask, zoom)

            if flip: 
                mask = tf.image.flip_left_right(mask)

            mask = tf.image.resize(mask, shape[-3:-1] + resize_pix, 'nearest')
            mask = mask[:, upper:shape[1]+upper, left:shape[2]+left]
            mask = tf.reshape(mask, shape[:-1])

            new_batch['mask'] = tf.cast(mask, tf.uint8)

        return new_batch

    return augment
    
def get_preprocess_fn(norm_type, batch_size=128, image_size=(224, 224), seq=False, incl_mask=False):   
    def vgg_norm(face):
        mean = tf.constant([91.4953, 103.8827, 131.0912])
        
        return face - mean
    
    def dfdc_norm(face):
        mean = tf.constant([109.733734, 92.62417, 85.35359])
        std = tf.constant([55.618042, 55.17303, 54.191914])

        return (face - mean) / std
    
    def imagenet_norm(face):
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        
        return (face / tf.constant(255, tf.float32) - mean) / std
 
    norm_fn = {'vgg': vgg_norm,
               'dfdc': dfdc_norm,
               'imagenet': imagenet_norm,
               None: dfdc_norm
              }

    def preprocess(batch):
        if seq:
            shape = tf.shape(batch['face'])
            face = tf.map_fn(lambda x: tf.image.resize(x, image_size), batch['face'], tf.float32)
            face = tf.reshape(face, (batch_size, shape[1], *image_size, 3))
        else:
            face = tf.image.resize(batch['face'], image_size)
            face = tf.reshape(face, (batch_size, *image_size, 3))

        face = norm_fn[norm_type](face)

        label_code = tf.cast(batch['label_code'], tf.float32)
        label_code = tf.reshape(label_code, (batch_size,))
        
        if incl_mask:
            mask = tf.cast(tf.expand_dims(batch['mask'], axis=-1), tf.float32)

            if seq:
                mask = tf.map_fn(lambda x: tf.image.resize(x, image_size, 'nearest'), mask, tf.float32)
                mask = tf.reshape(tf.squeeze(mask), (batch_size, shape[1], *image_size))
            else:
                mask = tf.image.resize(mask, image_size, 'nearest')
                mask = tf.reshape(tf.squeeze(mask), (batch_size, *image_size))

            return face,  (mask, label_code)
        
        return face, label_code
    
    return preprocess


# This function actually creates the datasets. One cool thing was that most of the `tf.image` methods work on either three or four dimensional tensors, so by changing the order of the augmentation and batching, it was possible to use the same functions to create both single frame and sequence datasets. Another nice feature was the `tf.data.experimental.sample_from_datasets` method that takes a list of datasets and an optional set of weights to create a balanced dataset.

# In[21]:


def get_ds(split, batch_size=32, n_frames=1, incl_mask=False, seq=False, shuffle=False):
            
    def get_ds_label(label):
        return (tf.data.Dataset.list_files(f'{TFRECORD_DIR}/{TFRECORD_PREFIX}_{split}_{label}_*.tfrecords',
                                           shuffle=shuffle)
                .interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)
                .map(get_parse_face_with_mask_examples_fn(n_frames, incl_mask), num_parallel_calls=AUTO)
                .repeat())
        
    ds_real = get_ds_label('REAL')
    ds_fake = get_ds_label('FAKE')
    
    ds_bal = tf.data.experimental.sample_from_datasets([ds_real, ds_fake], [1., 1.], seed=42)
    
    if seq:
        if split == 'train':
            ds_bal = ds_bal.map(get_augment_fn(incl_mask=incl_mask), num_parallel_calls=AUTO)
        
        ds_bal = ds_bal.batch(batch_size)
        
    else:
        ds_bal = ds_bal.map(flatten_frames, num_parallel_calls=AUTO).unbatch().batch(batch_size)

        if split == 'train':
            ds_bal = ds_bal.map(get_augment_fn(incl_mask=incl_mask), num_parallel_calls=AUTO)
        
    return ds_bal.prefetch(AUTO)


# In[22]:


if TFRECORD_DIR[:2] == 'gs':
    for b in bucket.list_blobs(prefix=TFRECORD_PREFIX):
        print(b.name, f'{b.size / 1e6:.1f} MB')
        
else:
    for p in sorted(Path(TFRECORD_DIR).glob(f'{TFRECORD_PREFIX}*')):
        print(p.name, f'{p.lstat().st_size / 1e6:.1f} MB')


# In[23]:


ds = get_ds('train', seq=False, incl_mask=True)
for b in ds.take(1):
    b=b


# These records have gone the full, full cycle, extracted faces from video frames, serialized to tfrecords and then parsed back out into a dataset.

# In[24]:


show_images(b['face'].numpy(), b['name'].numpy().tolist(), hw=(2,2), rc=(1,8))
show_images(b['mask'].numpy(), b['label'].numpy().tolist(), hw=(2,2), rc=(1,8))


# # Models

# ## Device 

# This allows the models to run on CPU or GPU or TPU if available.

# In[25]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_NAME)
    print('Running on TPU ', tpu.master())
except:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# ## Single Frame CNN 

# This is just a quick example to demonstrate how the datasets can be constructed to train a simple single frame cnn model. To train it on a TPU, the tfrecord files would need to be in cloud storage. There also aren't any callbacks here, most notably for saving model checkpoints. The model and checkpoint directories would also need to be on cloud storage in order to work with TPUs. The batch sizes could obviously be a lot larger on TPUs. I was running batch sizes as large as 2000 in some models.

# In[26]:


image_size = (260, 260)
batch_size = 16
seq = False
incl_mask = False

ds_train = get_ds('train', batch_size=batch_size, n_frames=1,
                      seq=seq, incl_mask=incl_mask, shuffle=True)
ds_valid = get_ds('valid', batch_size=batch_size, n_frames=1,
                      seq=seq, incl_mask=incl_mask)

preprocess = get_preprocess_fn('dfdc', batch_size=batch_size,
                               image_size=image_size, seq=seq, incl_mask=incl_mask)

ds_train_fit = ds_train.map(preprocess, num_parallel_calls=AUTO)
ds_valid_fit = ds_valid.map(preprocess, num_parallel_calls=AUTO)


# In[27]:


with strategy.scope():
    
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    
    cnn = efn.EfficientNetB2(weights=None,include_top=False,pooling='avg', input_shape=(*image_size, 3))
    
    model = tf.keras.Sequential([
        cnn,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='selu', kernel_initializer="lecun_normal"),
        tf.keras.layers.Dense(512, activation='selu', kernel_initializer="lecun_normal"),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss=loss_fn, optimizer=opt, metrics=metrics)
    
model.summary()


# In[28]:


history = model.fit(ds_train_fit,
                    steps_per_epoch=10,
                    epochs=2,
                    validation_data=ds_valid_fit,
                    validation_steps=2
                   )


# In[29]:


plot_learning_curves(history)


# ## Sequence

# Likewise, this is also a simple implementation to demonstrate how the datasets can be configured from the same tfrecord files to train a sequence model. This model starts with a pretrained resnet50 model trained on the vggface2 dataset from [WeidiXie/
# Keras-VGGFace2-ResNet50](https://github.com/WeidiXie/Keras-VGGFace2-ResNet50), unfreezing the weights in the last convolution block.e

# In[96]:


batch_size = 8
image_size = (224, 224)
n_frames = 16
seq = True
incl_mask = False

ds_train = get_ds('train', batch_size=batch_size, n_frames=n_frames,
                      seq=seq, incl_mask=incl_mask, shuffle=True)
ds_valid = get_ds('valid', batch_size=batch_size, n_frames=n_frames,
                      seq=seq, incl_mask=incl_mask)

preprocess = get_preprocess_fn('vgg', batch_size=batch_size, image_size=image_size,
                               seq=seq, incl_mask=incl_mask)

ds_train_fit = ds_train.map(preprocess, num_parallel_calls=AUTO)
ds_valid_fit = ds_valid.map(preprocess, num_parallel_calls=AUTO)


# In[97]:


with strategy.scope():
    
    opt = tf.keras.optimizers.Adam(lr=1e-4)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metrics = [tf.keras.metrics.BinaryAccuracy()]    

    cnn = Vggface2_ResNet50(mode='train')
    cnn.load_weights(str(DATASET_DIR/'vggface2_Keras/model/resnet50_softmax_dim512/weights.h5'))
    inputs = tf.keras.layers.Input(shape=(*image_size, 3))
    cnn = tf.keras.Model(cnn.get_layer('base_input').input, outputs=cnn.get_layer('dim_proj').output)

    cnn.trainable = True
    for l in cnn.layers[:-13]:
        l.trainable = False
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input((n_frames, *image_size, 3)),
        tf.keras.layers.TimeDistributed(cnn),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(128, activation='selu', kernel_initializer='lecun_normal'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(loss=loss_fn, optimizer=opt, metrics=metrics)
    
model.summary()


# In[32]:


history = model.fit(ds_train_fit,
                    steps_per_epoch=10,
                    epochs=2,
                    validation_data=ds_valid_fit,
                    validation_steps=2
                   )


# In[33]:


plot_learning_curves(history)


# ## Segmentation

# This segmentation model uses a Unet from the excellent [qubvel/
# segmentation_models](https://github.com/qubvel/segmentation_models) repo, using the EfficientnetB2 cnn as the encoder. It also adds a layer after the segmentation output to calculate binary classification loss and then weights the segmentation and binary classification losses 4 to 1.
# 
# I train this model for few epochs on the toy tfrecords and then load weights from a checkpoint after more extensive training to make predictions in the next section.

# In[34]:


batch_size = 16
image_size = (256, 256)
n_frames = 1
encoder_weights = 'imagenet'
seq = False
incl_mask = True

ds_train = get_ds('train', batch_size=batch_size, n_frames=n_frames,
                      seq=seq, incl_mask=incl_mask, shuffle=True)
ds_valid = get_ds('valid', batch_size=batch_size, n_frames=n_frames,
                      seq=seq, incl_mask=incl_mask)

preprocess = get_preprocess_fn(encoder_weights, batch_size=batch_size, image_size=image_size,
                               seq=seq, incl_mask=incl_mask)

ds_train_fit = ds_train.map(preprocess, num_parallel_calls=AUTO)
ds_valid_fit = ds_valid.map(preprocess, num_parallel_calls=AUTO)


# In[35]:


with strategy.scope():
    class SegClassModel(tf.keras.Model):

        def __init__(self):
            super(SegClassModel, self).__init__()
            self.seg_model = sm.Unet('efficientnetb2', encoder_weights=None, encoder_freeze=False)
            self.pooling = tf.keras.layers.GlobalAvgPool2D()
            self.flatten = tf.keras.layers.Flatten()
            self.dense0 = tf.keras.layers.Dense(512, activation='selu', kernel_initializer="lecun_normal")
            self.dense1 = tf.keras.layers.Dense(512, activation='selu', kernel_initializer="lecun_normal")
            self.final = tf.keras.layers.Dense(1, activation='sigmoid', name='class_output')

        def call(self, inputs):
            seg_output = self.seg_model(inputs)
            class_output = self.pooling(seg_output)
            class_output = self.flatten(class_output)
            class_output = self.dense0(class_output)
            class_output = self.dense1(class_output)
            class_output = self.final(class_output)

            return seg_output, class_output
    
    opt = tf.keras.optimizers.Adam(lr=1e-5)
    seg_loss_fn = sm.losses.bce_jaccard_loss
    class_loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    seg_metrics = [sm.metrics.iou_score]
    class_metrics = [tf.keras.metrics.BinaryAccuracy()]
    
    model = SegClassModel()

    model.compile(loss=[seg_loss_fn, class_loss_fn],
                  loss_weights=[4., 1.],
                  optimizer=opt,
                  metrics=[seg_metrics, class_metrics])
   
    model.build((batch_size, *image_size, 3))
model.summary()


# In[36]:


history = model.fit(ds_train_fit,
                    steps_per_epoch=10,
                    epochs=2,
                    validation_data=ds_valid_fit,
                    validation_steps=2
                   )


# In[37]:


plot_learning_curves(history)


# Load weights from checkpoint.

# In[38]:


cp_dir = f'{DATASET_DIR}/seg_ckpt'
cp_filename = f'{cp_dir}/cp-0023.ckpt'
with strategy.scope():
    model.load_weights(cp_filename)


# # Predictions

# For the sake of simplicity, I just use the validation set from the segmentation example above, but extend out the number of frames pulled for each video to 16 in order to improve prediction accuracy at the video level.

# In[39]:


batch_size = 16
image_size = (256, 256)
n_frames = 16
encoder_weights = 'imagenet'
seq = False
incl_mask = True

ds_pred = get_ds('valid', batch_size=batch_size, n_frames=n_frames,
                      seq=seq, incl_mask=incl_mask).take(32)

preprocess_pred = get_preprocess_fn(encoder_weights, batch_size=batch_size, image_size=image_size,
                               seq=seq, incl_mask=incl_mask)

ds_pred_pp = ds_pred.map(preprocess_pred)


# In[40]:


# make sure example order is deterministic so we can line up training data with predictions
assert np.array_equal(np.concatenate([b['name'] for b in ds_pred.as_numpy_iterator()]),
               np.concatenate([b['name'] for b in ds_pred.as_numpy_iterator()]))


# In[41]:


predictions = model.predict(ds_pred_pp)
name_list = []
face_list = []
frame_list = []
mask_list = []
for b in ds_pred.as_numpy_iterator():
    face_list.extend(b['face'].squeeze())
    name_list.extend(b['name'].squeeze())
    if not seq:
        frame_list.extend(b['idx'].squeeze())
    if incl_mask:
        b_pp = preprocess_pred(b)
        mask_list.extend(b_pp[1][0].numpy().squeeze())


# In[42]:


df_pred = pd.DataFrame({'name': [n.decode() for n in name_list]})

if incl_mask:
    df_pred['mask_pred'] = [m.squeeze() for m in predictions[0]]
    df_pred['mask_actual'] = [m.squeeze() for m in mask_list]
    df_pred['iou'] = df_pred.apply(iou, axis=1)
    df_pred['prob'] = [p.squeeze() for p in predictions[1]]
else:
    def_pred['prob'] = predictions.squeeze()
    
if not seq:
    df_pred['frame'] = frame_list

df_pred['face'] = face_list
df_pred['actual'] = df_pred.name.map(df_meta.label_code)
df_pred['label'] = df_pred.name.map(df_meta.label)
df_pred['error'] = np.square(df_pred.actual - df_pred.prob)
df_pred['prob_thresh'] = (df_pred.prob > 0.5).astype(int)


# ## Face level classification 

# This just takes every face that was predicted and calculates losses and classification metrics at the face level. The numbers here are obviously terrible since I couldn't make the underlying model work, but the predictions below, where the model got it right are pretty cool.

# In[43]:


print_pred_metrics(df_pred.actual, df_pred.prob_thresh, df_pred.prob)


# Here are some examples of where the model got it right.

# In[99]:


for r in df_pred[df_pred.actual == 1].sort_values('error').to_records()[:5]:
    show_pred(r)


# ## Video level classification 

# This section would get used when running models on individual faces predictions for multiple faces for each video being made. It aggregates together the frames by video and then determines whether a video should be classified as fake if it had a specified number of frames with a predicted probability over a specified frame level probability.

# In[52]:


vid_prob_thresh = 0.8
frame_thresh = 3
df_pred['vid_prob_thresh'] = (df_pred.prob > vid_prob_thresh).astype(int)


# In[53]:


df_pred_vid = df_pred.groupby(['name']).agg({'actual': 'max', 'vid_prob_thresh': 'sum'})
df_pred_vid['vid_frame_thresh'] = (df_pred_vid.vid_prob_thresh > frame_thresh).astype(int)
df_pred_vid['vid_prob'] = (df_pred_vid.vid_prob_thresh / (frame_thresh * 1.5)).clip(0.05, 0.95)


# In[54]:


print_pred_metrics(df_pred_vid.actual, df_pred_vid.vid_frame_thresh, df_pred_vid.vid_prob)


# ## Commit

# ### Update Dataset

# In[81]:


if False:
    if not KAGGLE:
        print_output(run_command(f'kaggle d version -r tar -p {DATASET_DIR} -m "added dependencies"'))


# ### Save Notebook

# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.save_notebook()')


# ### Commit Kernel

# In[12]:


if True:
    if not KAGGLE:

        data = {'id': 'calebeverett/dfdc-novice-approach',
                      'title': 'dfdc-novice-approach',
                      'code_file': 'dfdc-novice-approach.ipynb',
                      'language': 'python',
                      'kernel_type': 'notebook',
                      'is_private': 'false',
                      'enable_gpu': 'false',
                      'enable_internet': 'false',
                      'dataset_sources': ['calebeverett/dfdc-na'],
                      'competition_sources': ['deepfake-detection-challenge'],
                     ' kernel_sources': []}
        
        with open('kernel-metadata.json', 'w') as f:
            json.dump(data, f)

        print_output(run_command('kaggle k push'))

