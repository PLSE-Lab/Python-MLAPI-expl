#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ../input/kaggle-efficientnet-repo/efficientnet-1.0.0-py3-none-any.whl')


# In[ ]:


import pandas as pd
import tensorflow as tf
import cv2
import glob
from tqdm.notebook import tqdm
import numpy as np
import os
from keras.layers import *
from keras import Model
import matplotlib.pyplot as plt
import time
from keras.applications.xception import Xception
import efficientnet.keras as efn


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())


# In[ ]:


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu


# In[ ]:


test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
len(test_videos)


# In[ ]:


import sys
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
sys.path.insert(0, "/kaggle/input/deepfakes-inference-demo")


# In[ ]:


from blazeface import BlazeFace
facedet = BlazeFace().to(gpu)
facedet.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")
_ = facedet.train(False)


# In[ ]:


input_size = 224


# In[ ]:


from torchvision.transforms import Normalize

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)


# In[ ]:


from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor

frames_per_video = 10

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)


# In[ ]:


def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


# In[ ]:


import torch.nn as nn
import torchvision.models as models

class HisResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(HisResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)
        self.fc = nn.Linear(2048, 1)


# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('../input/mobilenet-face/frozen_inference_graph_face.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# In[ ]:




checkpoint = torch.load("/kaggle/input/deepfakes-inference-demo/resnext.pth", map_location=gpu)

model = HisResNeXt().to(gpu)
model.load_state_dict(checkpoint)
_ = model.eval()

del checkpoint


# In[ ]:


def predict_on_video(video_path, batch_size):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)
        
        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.                    
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = make_square_image(resized_face)

                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
                    
                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    #x[n] = cv2.flip(resized_face, 1)
                    #n += 1

            if n > 0:
                x = torch.tensor(x, device=gpu).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                # Make a prediction, then take the average.
                with torch.no_grad():
                    y_pred = model(x)
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    return y_pred[:n].mean().item()

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5


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


if not os.path.isdir(f'./videos/'):
    os.makedirs(f'./videos/')


# In[ ]:


res_predictions =[]


# In[ ]:


for x in tqdm(glob.glob('../input/deepfake-detection-challenge/test_videos/*.mp4')):
    try:
        filename=x.replace('../input/deepfake-detection-challenge/test_videos/','').replace('.mp4','.jpg')
        a=detect_video(x)
        
        y_pred = predict_on_video(x, batch_size=frames_per_video)
        res_predictions.append(y_pred)
        if a is None:
            continue
        cv2.imwrite('./videos/'+filename,a)
    except Exception as err:
        print(err)


# In[ ]:


cm.__exit__(None,Exception,'exit')
sess.close()


# In[ ]:


bottleneck_EfficientNetB1 = efn.EfficientNetB1(weights=None,include_top=False,pooling='avg')
inp=Input((10,240,240,3))
x=TimeDistributed(bottleneck_EfficientNetB1)(inp)
x = LSTM(128)(x)
x = Dense(64, activation='elu')(x)
x = Dense(1,activation='sigmoid')(x)
model_EfficientNetB1=Model(inp,x)

bottleneck_Xception = Xception(weights=None,
                 include_top=False,pooling='avg')
y=TimeDistributed(bottleneck_Xception)(inp)
y = LSTM(128)(y)
y = Dense(64, activation='elu')(y)
y = Dense(1,activation='sigmoid')(y)
model_Xception=Model(inp,y)


# In[ ]:


model_EfficientNetB1.load_weights('../input/efficientnetb1dfdc/EfficientNetB1-e_2_b_4_f_30-10.h5')
model_Xception.load_weights('../input/xceptiondfdc/Xception-e_2_b_4_f_30-10.h5')


# In[ ]:


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
    #print(x)
    img=process_img(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB))
    #print(img)
    if img is None:
        continue
    batch.append(img)
    batch1.append(process_img(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB),True))
    filenames.append(x.replace('./videos/','').replace('.jpg','.mp4'))
    if len(batch)==16:
        preds+=(((0.2*model_EfficientNetB1.predict(np.array(batch))))+((0.2*model_EfficientNetB1.predict(np.array(batch1))))
               +((0.2*model_Xception.predict(np.array(batch))))+((0.2*model_Xception.predict(np.array(batch1))))).tolist()

        #preds+=(((0.5*model_EfficientNetB1.predict(np.array(batch))))+((0.5*model_Xception.predict(np.array(batch))))).tolist()

        batch=[]
        batch1=[]

if len(batch)!=0:
    #preds+=(((0.5*model_EfficientNetB1.predict(np.array(batch))))+((0.5*model_Xception.predict(np.array(batch))))).tolist()
    preds+=(((0.2*model_EfficientNetB1.predict(np.array(batch))))+((0.2*model_EfficientNetB1.predict(np.array(batch1))))
       +((0.2*model_Xception.predict(np.array(batch))))+((0.2*model_Xception.predict(np.array(batch1))))).tolist()


# In[ ]:


#print(preds)


# In[ ]:


#print(res_predictions)


# In[ ]:


new_preds=[]
for x,y in zip(preds,res_predictions):
    new_preds.append(x[0]+(0.2*y))
print(sum(new_preds)/len(new_preds))


# In[ ]:


for x,y in zip(new_preds,filenames):
    #submission.loc[submission['filename']==y,'label']=min([max([0.1,x]),0.9])
    submission.loc[submission['filename']==y,'label']=x


# In[ ]:


plt.hist(submission['label'])


# In[ ]:


submission.head(10)


# In[ ]:


np.array(submission['label']).mean()


# In[ ]:


submission.to_csv('submission.csv', index=False)
get_ipython().system('rm -r videos')


# In[ ]:




