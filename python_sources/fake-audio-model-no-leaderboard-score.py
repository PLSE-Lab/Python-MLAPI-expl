#!/usr/bin/env python
# coding: utf-8

# # Audio Model Notes

# **Notebook**
# 
# We created a fake-audio detection model to detect videos with 'fake audio' and 'real faces'. This model should stop other notebooks from predicting 'real' when they should be predicting 'fake'. This notebook implements this 'fake audio' model with a resnext model and if the fake audio model predicts > .7, the final prediction is .95 fake. If it is lower than .7, it feeds the video to the resnext model and uses that prediction.
# 
# We are releasing this model because of covid-19 (time constraints and lack of access to previous resources). We can get this notebook to commit, but we cannot get it to 'submit'. If any of you can get it to commit, please share to the public.
# 
# **Dataset**
# 
# We created a fake-audio detection model from fake and real audio that we isolated from the dataset. Each 'fake' video has a corresponding 'real' video - we labeled any audio from a 'fake' video that deviated from the corresponding audio from the 'real' video as 'fake audio'. Audio from 'real' video was labeled 'real audio'. The fake audio files can be found here: https://www.kaggle.com/sherkt1/fakeaudio
# 
# **Model**
# 
# We used librosa to extract features from the audio and trained a neural network with three hidden layers. Test accuracy of just fake-audio and real-audio was ~ 93%. Fake-Audio detection accuracy with a threshold above .7 for fake-audio, real, and fake-video was ~ 92%.
# 
# Notebook to train model can be found here: https://www.kaggle.com/sherkt1/train-audio-model?scriptVersionId=30962766
# 
# AudoModel can be found here: https://www.kaggle.com/sherkt1/audiomodel2
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import librosa
import os, sys, time
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "/kaggle/input/helpers/")

from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor
from torchvision.transforms import Normalize
import torch.nn as nn
import torchvision.models as models
from concurrent.futures import ThreadPoolExecutor


# In[ ]:


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu


# In[ ]:


get_ipython().system('ls ../input/deepfake-xception-trained-model')
get_ipython().system('ls ../input/deepfake-kernel-data')
get_ipython().system('pip install ../input/deepfake-xception-trained-model/pytorchcv-0.0.55-py2.py3-none-any.whl --quiet')


# In[ ]:


get_ipython().system('pip install /kaggle/input/video-tools/video-tools/wheelhouse/*.whl')
get_ipython().system('dpkg -i --force-confdef /kaggle/input/video-tools/video-tools/dpkgs/*.deb')


# In[ ]:


import moviepy.editor as mp
import keras
import pickle
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

scalerfile = 'scaler.sav'
scaler = pickle.load(open('/kaggle/input/audiomodel2/' + scalerfile, 'rb'))
audiomodel = keras.models.load_model('/kaggle/input/audiomodel2/audiomodel2.h5')


# # Def

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


# # Initializing Face Extractor

# In[ ]:


import sys
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
sys.path.insert(0, "/kaggle/input/deepfakes-inference-demo")

from blazeface import BlazeFace
facedet = BlazeFace().to(gpu)
facedet.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")
_ = facedet.train(False)

frames_per_video = 10
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)


# # Prediction Def

# In[ ]:


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)
    
def videoprediction(video_path, batch_size):
    try:
        faces = face_extractor.process_video(video_path)
        face_extractor.keep_only_best_face(faces)
        
        if len(faces) > 0:
            x = np.zeros((batch_size, 224, 224, 3), dtype=np.uint8)
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:     
                    resized_face = isotropically_resize_image(face, 224)
                    resized_face = make_square_image(resized_face)

                    if n < batch_size:
                        x[n] = resized_face
                        n += 1

            if n > 0:
                x = torch.tensor(x, device=gpu).float()
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)
                    
                with torch.no_grad():
                    #Need to return both resnext and xception predictions
                    resnext_pred = resnext_model(x)
                    resnext_pred = torch.sigmoid(resnext_pred.squeeze())
                    torch.cuda.empty_cache()
                    
                    return resnext_pred[:n].mean().item()

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5


# # Initialize Models

# In[ ]:


class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)
        self.fc = nn.Linear(2048, 1)
        
class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()

        self.p1 = nn.AdaptiveAvgPool2d((1,1))
        self.p2 = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x):
        x1 = self.p1(x)
        x2 = self.p2(x)
        return (x1+x2) * 0.5


class Head(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.5)
        self.o = nn.Linear(512, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out

class FCN(torch.nn.Module):
    def __init__(self, base, in_f):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, 1)
  
    def forward(self, x):
        x = self.base(x)
        return self.h1(x)
    
checkpoint = torch.load("/kaggle/input/deepfakes-inference-demo/resnext.pth", map_location=gpu)
model = MyResNeXt().to(gpu)
model.load_state_dict(checkpoint)
_ = model.eval()
del checkpoint

resnext_model = model


# # Print to log

# In[ ]:


import os
import gc
__print__ = print
def print(string):
    os.system(f'echo \"{string}\"')
    __print__(string)


# # Prediction Loop

# In[ ]:


get_ipython().system('mkdir temp')

path = '/kaggle/input/deepfake-detection-challenge/test_videos/'
movies = os.listdir(path)

finalpred = []
finalmovies = []
count = 0
for movie in movies:
    try:
        try:
            gc.collect()
            torch.cuda.empty_cache()
            if movie[-4:] != ".mp4":
                continue

            print('count: ' + str(count))
            count += 1
            #Load audio, save as temporary .mp3, load with librosa
            audioclip = mp.VideoFileClip(path + movie)
            audioclip.audio.write_audiofile('/kaggle/working/temp/temp.wav', verbose=False, logger=None)
            y, sr = librosa.load('/kaggle/working/temp/temp.wav', mono=True, duration=30)

            #Extract Features
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)

            #Concatenate Features, Scale
            audiodata = np.concatenate((np.expand_dims(np.mean(chroma_stft), axis=0),np.expand_dims(np.mean(rmse), axis=0),np.expand_dims(np.mean(spec_cent), axis=0),np.expand_dims(np.mean(spec_bw), axis=0),np.expand_dims(np.mean(rolloff), axis=0),np.expand_dims(np.mean(zcr), axis=0)),axis=0)
            for e in mfcc:
                 audiodata = np.concatenate((audiodata,np.expand_dims(np.mean(e), axis=0)),axis=0)
            audiodata = scaler.transform(np.expand_dims(audiodata,axis=0))

            #Predict, choose tocsv or to cnn
            prediction = audiomodel.predict(audiodata)

            if prediction[0][0] > .7:
                finalpred.append(.95)
                finalmovies.append(movie)
            else:
                prediction1 = videoprediction(path+movie,30)
                if prediction1 >= 1:
                    prediction1 = .99
                if prediction1 <= 0:
                    prediction1 = .01
                finalpred.append(prediction1)
                finalmovies.append(movie)
        except:
            prediction1 = videoprediction(path+movie,30)
            if prediction1 >= 1:
                prediction1 = .99
            if prediction1 <= 0:
                prediction1 = .01
            finalpred.append(prediction1)
            finalmovies.append(movie)
    except:
        finalmovies.append(movie)
        finalpred.append(.5)


# In[ ]:


submission = pd.DataFrame({"filename": finalmovies, "label": finalpred})
submission.to_csv("submission.csv", index=False)

