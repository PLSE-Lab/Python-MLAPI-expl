#!/usr/bin/env python
# coding: utf-8

# Here is just an example of how to use mobilenet face extractor for inference, and also lrcn. 
# 
# 
# I didn't make the weights public, because people(including me) don't like high scoring infernece kernel. 
# 
# 
# **BUT I'm also taking request to make the weights public. I'm OK with making the weights public.**

# In[ ]:


get_ipython().system('pip install ../input/efficientnet/efficientnet-1.0.0-py3-none-any.whl')


# # Import Libraries

# In[ ]:


import pandas as pd
import tensorflow as tf
import cv2
import glob
from tqdm.notebook import tqdm
import numpy as np
import os
import efficientnet.keras as efn
from keras.layers import *
from keras import Model
import matplotlib.pyplot as plt


# # Initialize Face Extractor

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('../input/mobilenet-face/frozen_inference_graph_face.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# In[ ]:


cm = detection_graph.as_default()
cm.__enter__()


# In[ ]:


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(graph=detection_graph, config=config)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


# In[ ]:


def get_img(images):
    global boxes,scores,num_detections
    im_heights,im_widths=[],[]
    imgs=[]
    for image in images:
        (im_height,im_width)=image.shape[:-1]
        imgs.append(image)
        im_heights.append(im_height)
        im_widths.append(im_widths)
    imgs=np.array(imgs)
    (boxes, scores_) = sess.run(
        [boxes_tensor, scores_tensor],
        feed_dict={image_tensor: imgs})
    finals=[]
    for x in range(boxes.shape[0]):
        scores=scores_[x]
        max_=np.where(scores==scores.max())[0][0]
        box=boxes[x][max_]
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        left, right, top, bottom = int(left), int(right), int(top), int(bottom)
        image=imgs[x]
        finals.append(cv2.cvtColor(cv2.resize(image[max([0,top-40]):bottom+80,max([0,left-40]):right+80],(240,240)),cv2.COLOR_BGR2RGB))
    return finals
def detect_video(video):
    frame_count=10
    capture = cv2.VideoCapture(video)
    v_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0,v_len,frame_count, endpoint=False, dtype=np.int)
    imgs=[]
    i=0
    for frame_idx in range(int(v_len)):
        ret = capture.grab()
        if not ret: 
            print("Error grabbing frame %d from movie %s" % (frame_idx, video))
        if frame_idx >= frame_idxs[i]:
            if frame_idx-frame_idxs[i]>20:
                return None
            ret, frame = capture.retrieve()
            if not ret or frame is None:
                print("Error retrieving frame %d from movie %s" % (frame_idx, video))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgs.append(frame)
            i += 1
            if i >= len(frame_idxs):
                break
    imgs=get_img(imgs)
    if len(imgs)<10:
        return None
    return np.hstack(imgs)


# In[ ]:


os.mkdir('./videos/')
for x in tqdm(glob.glob('../input/deepfake-detection-challenge/test_videos/*.mp4')):
    try:
        filename=x.replace('../input/deepfake-detection-challenge/test_videos/','').replace('.mp4','.jpg')
        a=detect_video(x)
        if a is None:
            continue
        cv2.imwrite('./videos/'+filename,a)
    except Exception as err:
        print(err)


# In[ ]:


cm.__exit__(None,Exception,'exit')
sess.close()


# # Initialize Model

# In[ ]:


bottleneck = efn.EfficientNetB1(weights=None,include_top=False,pooling='avg')
inp=Input((10,240,240,3))
x=TimeDistributed(bottleneck)(inp)
x = LSTM(128)(x)
x = Dense(64, activation='elu')(x)
x = Dense(1,activation='sigmoid')(x)


# In[ ]:


model=Model(inp,x)
model.load_weights('../input/dfdc-model3/model.h5')

def get_birghtness(img):
    return img/img.max()
# %% [code]
def process_img(img,flip=False):
    imgs=[]
    for x in range(10):
        if flip:
            imgs.append(get_birghtness(cv2.flip(img[:,x*240:(x+1)*240,:],1)))
        else:
            imgs.append(get_birghtness(img[:,x*240:(x+1)*240,:]))
    return np.array(imgs)


# In[ ]:


sample_submission = pd.read_csv("../input/deepfake-detection-challenge/sample_submission.csv")
test_files=glob.glob('./videos/*.jpg')
submission=pd.DataFrame()
submission['filename']=os.listdir(('../input/deepfake-detection-challenge/test_videos/'))
submission['label']=0.5
filenames=[]
batch=[]
batch1=[]
preds=[]


# In[ ]:


for x in test_files:
    img=process_img(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB))
    if img is None:
        continue
    batch.append(img)
    batch1.append(process_img(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB),True))
    filenames.append(x.replace('./videos/','').replace('.jpg','.mp4'))
    if len(batch)==16:
        preds+=(((0.5*model.predict(np.array(batch))))+((0.5*model.predict(np.array(batch1))))).tolist()
        batch=[]
        batch1=[]
if len(batch)!=0:
    preds+=(((0.5*model.predict(np.array(batch))))+((0.5*model.predict(np.array(batch1))))).tolist()


# In[ ]:


new_preds=[]
for x in preds:
    new_preds.append(x[0])
print(sum(new_preds)/len(new_preds))


# In[ ]:


for x,y in zip(new_preds,filenames):
    submission.loc[submission['filename']==y,'label']=min([max([0.1,x]),0.9])


# In[ ]:


plt.hist(submission['label'])


# In[ ]:


submission.head()


# In[ ]:


np.array(submission['label']).mean()


# In[ ]:


submission.to_csv('submission.csv', index=False)
get_ipython().system('rm -r videos')


# Thanks for reading. Please upvote if you found it helpful.
