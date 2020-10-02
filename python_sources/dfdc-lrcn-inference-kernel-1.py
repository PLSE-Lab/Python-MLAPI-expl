#!/usr/bin/env python
# coding: utf-8

# We started in last five lays and landed onto bronze. Thanks to [Harshit](https://www.kaggle.com/harshitsheoran) and [Shangqiu Li](https://www.kaggle.com/unkownhihi) for their kernels and opensourcing their approaches.<br><br>
# This kernel shows even if you start in the end, there is a scope to get something.

# <h3> Approach</h3>
# + [Sampling and balancing](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/132700#759482) + Flipping the frames
# + [Training](https://www.kaggle.com/unkownhihi/dfdc-lrcn-training)
# + [Inference](https://www.kaggle.com/unkownhihi/dfdc-lrcn-inference)
# 
# We trained multiple models (same data and same procedure) and took their average prediction as output.<br><br>
# The hope is optimizing some hyperparameters may get you much better scores.

# In[ ]:


import time
super_start = time.time()


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
import time


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
def detect_video(video, start_frame):
    frame_count=10
    capture = cv2.VideoCapture(video)
    v_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(start_frame,v_len,frame_count, endpoint=False, dtype=np.int)
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
        a=detect_video(x,0)
        if a is None:
            continue
        cv2.imwrite('./videos/'+filename,a)
    except Exception as err:
        print(err)


# In[ ]:


os.mkdir('./videos_2/')
for x in tqdm(glob.glob('../input/deepfake-detection-challenge/test_videos/*.mp4')):
    try:
        filename=x.replace('../input/deepfake-detection-challenge/test_videos/','').replace('.mp4','.jpg')
        a=detect_video(x,95)
        if a is None:
            continue
        cv2.imwrite('./videos_2/'+filename,a)
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

weights = ['../input/deepfake-20/saved-model-01-0.06.hdf5', '../input/deepfake-20/saved-model-02-0.05.hdf5', '../input/model-epoch-3/saved-model-03-0.06.hdf5','../input/model-02/saved-model-01-0.06.hdf5']*2

sub_file = ['submission_'+str(i)+'.csv' for i in range(1,9)]

video = ['./videos/']*4+['./videos_2/']*4

for xxxxx in range(8):
    start = time.time()
    model.load_weights(weights[xxxxx])

    def get_birghtness(img):
        return img/img.max()
    # %% [code]
    def process_img(img,flip=[False]*10):
        imgs=[]
        for x in range(10):
            if flip[x]:
                imgs.append(get_birghtness(cv2.flip(img[:,x*240:(x+1)*240,:],1)))
            else:
                imgs.append(get_birghtness(img[:,x*240:(x+1)*240,:]))
        return np.array(imgs)

    sample_submission = pd.read_csv("../input/deepfake-detection-challenge/sample_submission.csv")
    test_files=glob.glob(video[xxxxx]+'*.jpg')
    submission=pd.DataFrame()
    submission['filename']=os.listdir(('../input/deepfake-detection-challenge/test_videos/'))
    submission['label']=0.5
    filenames=[]

    batch=[]
    batch1=[]
    batch2=[]
    batch3=[]

    preds=[]

    for x in test_files:
        img=process_img(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB))
        if img is None:
            continue
        batch.append(img)
        batch1.append(process_img(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB),[True]*10))
        batch2.append(process_img(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB),[True,False]*5))
        batch3.append(process_img(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB),[False,True]*5))

        filenames.append(x.replace(video[xxxxx],'').replace('.jpg','.mp4'))
        if len(batch)==16:
            preds+=(((0.25*model.predict(np.array(batch))))+((0.25*model.predict(np.array(batch1))))+((0.25*model.predict(np.array(batch2))))+((0.25*model.predict(np.array(batch3))))).tolist()
            batch=[]
            batch1=[]
            batch2=[]
            batch3=[]
    if len(batch)!=0:
        preds+=(((0.25*model.predict(np.array(batch))))+((0.25*model.predict(np.array(batch1))))+((0.25*model.predict(np.array(batch2))))+((0.25*model.predict(np.array(batch3))))).tolist()

    print(time.time()-start)

    new_preds=[]
    for x in preds:
        new_preds.append(x[0])
    print(sum(new_preds)/len(new_preds))

    for x,y in zip(new_preds,filenames):
        submission.loc[submission['filename']==y,'label']=min([max([0.05,x]),0.95])

    submission.to_csv(sub_file[xxxxx], index=False)


# In[ ]:


get_ipython().system('rm -r videos')
get_ipython().system('rm -r videos_2')


# In[ ]:


df1 = pd.read_csv('submission_1.csv').set_index('filename').transpose().to_dict()
df2 = pd.read_csv('submission_2.csv').set_index('filename').transpose().to_dict()
df3 = pd.read_csv('submission_3.csv').set_index('filename').transpose().to_dict()
df4 = pd.read_csv('submission_4.csv').set_index('filename').transpose().to_dict()
df5 = pd.read_csv('submission_5.csv').set_index('filename').transpose().to_dict()
df6 = pd.read_csv('submission_6.csv').set_index('filename').transpose().to_dict()
df7 = pd.read_csv('submission_7.csv').set_index('filename').transpose().to_dict()
df8 = pd.read_csv('submission_8.csv').set_index('filename').transpose().to_dict()
filename = []
label = []
for i in df1.keys():
    filename.append(i)
    a = []
    if df1[i]['label']!=0.5:
        a.append(df1[i]['label'])
    if df2[i]['label']!=0.5:
        a.append(df2[i]['label'])
    if df3[i]['label']!=0.5:
        a.append(df3[i]['label'])
    if df4[i]['label']!=0.5:
        a.append(df4[i]['label'])
    if df5[i]['label']!=0.5:
        a.append(df5[i]['label'])
    if df6[i]['label']!=0.5:
        a.append(df6[i]['label'])
    if df7[i]['label']!=0.5:
        a.append(df7[i]['label'])
    if df8[i]['label']!=0.5:
        a.append(df8[i]['label'])
    if len(a)==0:
        label.append(0.5)
    else:
        label.append(min([max([0.05,sum(a)/len(a)]),0.95]))
df = pd.DataFrame()
df['filename'] = filename
df['label'] = label
print(np.array(df['label']).mean())
df.to_csv('submission.csv', index=False)


# In[ ]:


get_ipython().system('rm submission_1.csv')
get_ipython().system('rm submission_2.csv')
get_ipython().system('rm submission_3.csv')
get_ipython().system('rm submission_4.csv')
get_ipython().system('rm submission_5.csv')
get_ipython().system('rm submission_6.csv')
get_ipython().system('rm submission_7.csv')
get_ipython().system('rm submission_8.csv')


# In[ ]:


plt.hist(df['label'])


# In[ ]:


print(time.time()-super_start)

