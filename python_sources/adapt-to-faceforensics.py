#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import torch
import cv2
import time
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
from torchvision.transforms import Normalize
gpu = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')


# In[ ]:


os.mkdir('/kaggle/working/img_data')
os.mkdir('/kaggle/working/img_data/Pristine')
os.mkdir('/kaggle/working/img_data/NeuralTextures')
os.mkdir('/kaggle/working/img_data/Face2Face')
img_PATH='/kaggle/working/img_data'
frames_per_video = 17
input_size = 224
test_val_frac = 0.3


# In[ ]:


print(os.listdir(img_PATH))


# In[ ]:


PATH='/kaggle/input/faceforensics'
face2face = os.listdir(os.path.join(PATH,'manipulated_sequences', 'Face2Face','c23', 'videos'))
neuralTextures = os.listdir(os.path.join(PATH,'manipulated_sequences', 'NeuralTextures','c23', 'videos'))
original = os.listdir(os.path.join(PATH,'original_sequences','youtube','c23', 'videos'))
label = ['Face2Face']*1000+['NeuralTextures']*1000+['Pristine']*1000
video_df = pd.DataFrame()
video_df['filename'] = face2face+neuralTextures+original
video_df['category'] = label
video_df.to_csv('/kaggle/working/img_data/metadata.csv')


# In[ ]:


import sys
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
sys.path.insert(0, "/kaggle/input/deepfakes-inference-demo")
from blazeface import BlazeFace
facedet = BlazeFace().to(gpu)
facedet.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")
_ = facedet.train(False)
from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor

video_reader = VideoReader(verbose=True)
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)


# In[ ]:


img_PATH='/kaggle/working/img_data'
for index, row in video_df.iterrows():
    filename=row['filename']
    category=row['category']
    cap = ''
    if category=='Face2Face':
        cap='manipulated_sequences/Face2Face'
    elif category=='NeuralTextures':
        cap='manipulated_sequences/NeuralTextures'
    else:
        cap='original_sequences/youtube'
    video_PATH = os.path.join(PATH, cap, 'c23', 'videos',filename)
    
    faces = face_extractor.process_video(video_PATH)
    face_extractor.keep_only_best_face(faces)
    if len(faces)>0:
        num=0
        for frame_data in faces:
            for face in frame_data['faces']:
                img_name = filename[:-4]+'_img_'+str(num)+'.jpg'
                cv2.imwrite('img_data/'+category+'/'+img_name, face)
                num+=1


# In[ ]:





# In[ ]:


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)


# In[ ]:


def make_split(df, frac):
    val_df = df.sample(frac=frac, random_state=666)
    train_df=df.loc[df.index.isin(val_df.index)]
    val_df.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    return train_df, val_df


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

def make_square_img(img):
    h,w = img.shape[:2]
    size= max(h, w)
    return cv2.copyMakeBorder(img, 0, size-h, 0, size-w, cv2.BORDER_CONSTANT, value=0)

def load_image(filename, label):
    X = torch.zeros((frames_per_video, 3, input_size, input_size))
    for i in range(frames_per_video):
        p = os.path.join(img_PATH, label, filename[:-4]+'_img_'+str(i)+'.jpg')
        img = cv2.imread(p)
        img = isotropically_resize_image(img, input_size)
        img = make_square_img(img)
        img = torch.tensor(img).float()
        img = img.permute(2,0,1)
        img = normalize_transform(img/255)
        X[i] = img
    y = 0 if label=='Pristine' else 1
    y = torch.tensor([y]*frames_per_video)
    return X,  y


# In[ ]:


from torch.utils.data import Dataset
class ImgDataset(Dataset):
    def __init__(self,df):
        self.df = df
    def __getitem__(self, index):
        filename = self.df['filename'][index]
        label = self.df['label'][index]
        return load_image(filename, label)
    def __len__(self,index):
        return len(self.df)
    
train_df, val_df = make_split(video_df, frac=test_val_frac)
print(train_df)


# In[ ]:


#train_data = ImgDataset(train_df)
#test_data = ImgDataset(test_df)
#train_iter = DataLoader(train_data, batchsize=32, shuffle=True, pin_memory=True)
#test_iter = DataLoader(test_data, batchsize=32, shuffle=True, pin_memory=True)


# In[ ]:


#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

