#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys 
sys.path.append("/kaggle/input/retinafacev0")
sys.path.append("/kaggle/input/deepfakerepov5/DeepFake-master/")
sys.path.append("/kaggle/input/efficient-pytorch/EfficientNet-PyTorch-master/")


# In[ ]:


def get_size(start_path = '../'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            if f.endswith('.mp4'): continue
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

print(get_size()/1024/1024/1024, 'gigabytes')


# In[ ]:


import cv2
import glob
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from shutil import copyfile
from os.path import isfile, join, abspath, exists, isdir, expanduser
from os import listdir, makedirs, getcwd, remove
from PIL import Image
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
from models.face_detection.retinaface import RetinaFaceDetector

# Hack so that RetinaFace does not download Resnet50
cache_dir = expanduser(join('~', '.torch'))
if not exists('/root/.cache/torch/checkpoints'):
    makedirs('/root/.cache/torch/checkpoints')
if not exists(cache_dir):
    makedirs(cache_dir)

models_dir = cache_dir + '/' + 'models/'
if not exists(models_dir):
    makedirs(models_dir)

model_name = 'resnet50-19c8e357.pth'
src = '/kaggle/input/resnet-pretrained/' + model_name;
dest = "/root/.cache/torch/checkpoints/resnet50-19c8e357.pth"
copyfile(src, dest)


# In[ ]:


# File paths
BASE_PATH = '/kaggle/input/deepfake-detection-challenge/'
TEST_VIDEO_PATH = BASE_PATH + 'test_videos/'
SAMP_PATH = BASE_PATH + 'sample_submission.csv'
# File paths
test_img_list = glob.glob(f'{TEST_VIDEO_PATH}*.mp4')


# In[ ]:


class EffnetTest(nn.Module):
    def __init__(self, version):
        super(EffnetTest, self).__init__()
        self.model = EfficientNet.from_name(f"efficientnet-{version}", override_params={"num_classes":1})
        self.model.fc = nn.Linear(512, 1)
        self.model._norm_layer = nn.GroupNorm(num_groups=32, num_channels=3)
        
    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)


# In[ ]:


class SimpleCNNInference:
    def __init__(self, model_path, version='b6', img_size=200):
        self.img_size = img_size
        
        self.transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(size=(self.img_size, self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        self.device = torch.device("cuda")
        self.model = EffnetTest(version=version)
        self.model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
        self.model.to(self.device)
        self.model.eval()
        
    def predict_single_from_array(self, img_frame):
        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB) 
        img_frame = self.transform(img_frame).to(self.device).unsqueeze(0)
        return float(self.model(img_frame))


# In[ ]:


test_filenames = [string.split('/')[-1] for string in test_img_list]

detector = RetinaFaceDetector(weights="/kaggle/input/resnetretinaface/Resnet50_Final.pth")
# EffnetB6 with resolution 200
inference = SimpleCNNInference(model_path="/kaggle/input/effnetb6-200-0741/EffnetB6_pytorch_group_imgface6_200_0.0741.pth", 
                               version='b6', img_size=200)
# EffnetB5 trained on different dataset
inference2 = SimpleCNNInference(model_path="/kaggle/input/testefficientnet/testEfficientNet.pth", 
                               version='b5', img_size=224)
# Normal EffnetB5
inference3 = SimpleCNNInference(model_path="/kaggle/input/effnetb5-imgface6-0074/EffnetB5_imgface6_0.074.pth", 
                               version='b5', img_size=224)
# Finetuned EffnetB6
inference4 = SimpleCNNInference(model_path="/kaggle/input/effnetb6-0071-finetuned/EffnetB6_pytorch_group_imgface6_0.071_finetuned.pth", 
                               version='b6', img_size=224)
# EffnetB4 with data augmentation and label smoothing
inference5 = SimpleCNNInference(model_path="/kaggle/input/effnetb4-0691/EffnetB4_pytorch_group_imgface6_0.0691.pth", 
                               version='b4', img_size=224)
frame_skip = 16
res = 2 # Resolution lowered by this factor
final_predictions = dict()
for i, video in enumerate(tqdm(test_img_list)):
    single_predictions = []
    cap = cv2.VideoCapture(video)   
    for j in range(300):
        _ = cap.grab()

        # Skip n frames
        if j % frame_skip == 0:
            pass
        else: 
            continue
            
        # Inference
        try:
            _, frame = cap.retrieve()
            result = detector.detect_faces(frame[::res, ::res])
            # empty frame
            if result == []:
                continue

            # Result is an array with all the bounding boxes detected.
            bounding_box = result[0]

            x1, y1 = bounding_box[0]*res, bounding_box[1]*res
            x2, y2 = bounding_box[2]*res, bounding_box[3]*res
            frame_face = frame[y1:y2, x1:x2]
            
            pred = inference.predict_single_from_array(frame_face)
            single_predictions.append(1-pred) 
            pred2 = inference2.predict_single_from_array(frame_face)
            single_predictions.append(1-pred2)
            pred3 = inference3.predict_single_from_array(frame_face)
            single_predictions.append(1-pred3)
            pred4 = inference4.predict_single_from_array(frame_face)
            single_predictions.append(1-pred4)
            pred5 = inference5.predict_single_from_array(frame_face)
            single_predictions.append(1-pred5)
        except:
            print("E")
    final_predictions[test_filenames[i]] = np.mean(single_predictions)
    
cv2.destroyAllWindows()


# In[ ]:


sub = pd.read_csv(SAMP_PATH)
# Add predictions and fill nans to be safe
sub['label'] = sub['filename'].map(final_predictions).clip(0.01, 0.99).fillna(0.5)


# In[ ]:


# Save as csv
sub.to_csv('submission.csv', index=False)

