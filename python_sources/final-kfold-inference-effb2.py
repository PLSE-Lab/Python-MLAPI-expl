#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from glob import glob

for path in glob('../input/*/*'):
    print(path)

# Any results you write to the current directory are saved as output.


# In[ ]:


import random
import re
from copy import deepcopy
from typing import Union, List, Tuple, Optional, Callable
from collections import OrderedDict, defaultdict
import math

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torchvision import transforms, models
from torchvision.transforms import Normalize
from tqdm import tqdm
from sklearn.cluster import DBSCAN


# In[ ]:


TARGET_H, TARGET_W = 224, 224
FRAMES_PER_VIDEO = 30
TEST_VIDEOS_PATH = '../input/deepfake-detection-challenge/test_videos'
NN_MODEL_PATHS = [
    '../input/kdold-deepfake-effb2/fold0-effb2-000epoch.pt',
    '../input/kdold-deepfake-effb2/fold0-effb2-001epoch.pt',
    '../input/kdold-deepfake-effb2/fold0-effb2-002epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold0-flip-effb2-000epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold0-flip-effb2-001epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold0-flip-effb2-002epoch.pt',
    
    '../input/kdold-deepfake-effb2/fold1-effb2-000epoch.pt',
    '../input/kdold-deepfake-effb2/fold1-effb2-001epoch.pt',
    '../input/kdold-deepfake-effb2/fold1-effb2-002epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold1-flip-effb2-000epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold1-flip-effb2-001epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold1-flip-effb2-002epoch.pt',
    
    '../input/kdold-deepfake-effb2/fold2-effb2-000epoch.pt',
    '../input/kdold-deepfake-effb2/fold2-effb2-001epoch.pt',
    '../input/kdold-deepfake-effb2/fold2-effb2-002epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold2-flip-effb2-000epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold2-flip-effb2-001epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold2-flip-effb2-002epoch.pt',

    '../input/kdold-deepfake-effb2/fold3-effb2-000epoch.pt',
    '../input/kdold-deepfake-effb2/fold3-effb2-001epoch.pt',
    '../input/kdold-deepfake-effb2/fold3-effb2-002epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold3-flip-effb2-000epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold3-flip-effb2-001epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold3-flip-effb2-002epoch.pt',

    '../input/kdold-deepfake-effb2/fold4-effb2-000epoch.pt',
    '../input/kdold-deepfake-effb2/fold4-effb2-001epoch.pt',
    '../input/kdold-deepfake-effb2/fold4-effb2-002epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold4-flip-effb2-000epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold4-flip-effb2-001epoch.pt',
    '../input/kfolddeepfakeeffb2-flip/fold4-flip-effb2-002epoch.pt',
]


# In[ ]:


SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# In[ ]:


import sys
sys.path.insert(0, "/kaggle/input/face-detector")

from face_detector import FaceDetector
from face_detector.utils import VideoReader


# In[ ]:


get_ipython().system('pip install ../input/pytorchefficientnet/EfficientNet-PyTorch-master > /dev/null')

from efficientnet_pytorch import EfficientNet

def get_net():
    net = EfficientNet.from_name('efficientnet-b2')
    net._fc = nn.Linear(in_features=net._fc.in_features, out_features=2, bias=True)
    return net


# In[ ]:


class DatasetRetriever(Dataset):

    def __init__(self, df):
        self.video_paths = df['video_path']
        self.filenames = df.index
        self.face_dr = FaceDetector(frames_per_video=FRAMES_PER_VIDEO)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize_transform = Normalize(mean, std)
        
        self.video_reader = VideoReader()
        self.video_read_fn = lambda x: self.video_reader.read_frames(x, num_frames=FRAMES_PER_VIDEO)

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        filename = self.filenames[idx]
        
        my_frames, my_idxs = self.video_read_fn(video_path)
        faces = self.face_dr.get_faces(
            my_frames, my_idxs,
            0.7, 0.7, 0.7, 0.6
        )

        n = len(faces)

        video = torch.zeros((n, 3, TARGET_H, TARGET_W))
        for i, face in enumerate(faces[:n]):
            face = 255 - face
            face = face.astype(np.float32)/255.
            face = torch.tensor(face)
            face = face.permute(2,0,1)
            face = self.normalize_transform(face)
            video[i] = face

        return filename, video


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nvideos = []\nfor video_path in glob(os.path.join(TEST_VIDEOS_PATH, '*.mp4')):\n    videos.append({'filename': video_path.split('/')[-1], 'video_path': video_path})\n    \ndf = pd.DataFrame(videos).set_index('filename')\n\nvideos = None\ndel videos\n\ndf.head()")


# In[ ]:


from skimage import io

for filename, video in DatasetRetriever(df[:1]):
    break
    
io.imshow(1 - video.permute(0,2,3,1).numpy()[5,:,:,:])


# In[ ]:





# In[ ]:


class DeepFakePredictor:

    def __init__(self):
        self.models = [self.prepare_model(get_net(), path) for path in NN_MODEL_PATHS]
        self.models_count = len(self.models)

    def predict(self, dataset):
        result = []
        
        with torch.no_grad():
            for filename, video in dataset:
                video = video.to(self.device, dtype=torch.float32)
                try:
                    label = self.predict_ensemble(video)
                except Exception as e:
                    print(f'Warning! {e}, {type(e)}')
                    label = 0.5

                result.append({
                    'filename': filename,
                    'label': label,
                })

        return pd.DataFrame(result).set_index('filename')

    def prepare_model(self, model, path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(self.device);

        if torch.cuda.is_available():
            model = model.cuda()
            
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f'Model prepared. Device is {self.device}')
        return model
    
    @staticmethod
    def net_forward(net, inputs):
        bs = inputs.size(0)
        # Convolution layers
        x = net.extract_features(inputs)
        # Pooling and final linear layer
        x = net._avg_pooling(x)
        emb = x.view(bs, -1)
        x = net._dropout(emb)
        x = net._fc(x)
        return emb, x
    
    def postprocess(self, embs, predictions):
        clusters = defaultdict(list)
        for prediction, cluster_id in zip(predictions, DBSCAN(eps=1.2, min_samples=1).fit_predict(embs)):
            clusters[cluster_id].append(prediction)
        sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))
        if len(sorted_clusters) < 2:
            return sorted_clusters[0][1]
        if len(sorted_clusters[1][1]) / len(predictions) > 0.25:
            return sorted_clusters[0][1] + sorted_clusters[1][1]
        return sorted_clusters[0][1]
    
    def predict_ensemble(self, video):
        embs, predictions = 0, 0
        for model in self.models:
            emb, prediction = self.net_forward(model, video)
            predictions += prediction / self.models_count
            embs += emb / self.models_count

        predictions = nn.functional.softmax(predictions, dim=1).data.cpu().numpy()[:,1]
        embs = embs.cpu().numpy()
        
        predictions = self.postprocess(embs, predictions)
        return np.mean(predictions)


# In[ ]:


deep_fake_predictor = DeepFakePredictor()


# In[ ]:


from concurrent.futures import ThreadPoolExecutor

def process_dfs(df, num_workers=2):
    def process_df(sub_df):
        dataset = DatasetRetriever(sub_df)
        result = deep_fake_predictor.predict(dataset)
        return result

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = ex.map(process_df, np.split(df, num_workers))

    return results


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nimport time\n\ncount = df.shape[0]\n\n\ntime_start = time.time()\nresults = process_dfs(df[:count])\ndtime = time.time() - time_start\n\nprint(f'[speed]:', round(dtime / count, 2), 'sec/video')\nprint(f'[sum_time]:', f'~{round(dtime / count * 4000 / 60)}', 'min')\n\nresult = pd.concat(list(results))\nresult")


# In[ ]:


result.to_csv('submission.csv')


# In[ ]:




