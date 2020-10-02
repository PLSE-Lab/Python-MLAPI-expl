#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
sys.path.insert(0, "/kaggle/input/deepfakes-inference-demo")
sys.path.insert(0, "/kaggle/input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4")
sys.path.insert(0, "/kaggle/input/facenetpytorch/facenet-pytorch-1.0.1/")

from models.mtcnn import MTCNN#, InceptionResnetV1, extract_face
import os
from pretrainedmodels import se_resnext50_32x4d

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
import os
import glob
import time
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from torchvision import transforms

import datetime

# See github.com/timesler/facenet-pytorch:
from concurrent.futures import ThreadPoolExecutor
from albumentations.augmentations.transforms import ShiftScaleRotate, VerticalFlip, RandomBrightnessContrast,     GaussianBlur, CoarseDropout, Normalize
from albumentations.pytorch import ToTensor
from albumentations import Compose


# In[ ]:


test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
len(test_videos)


# In[ ]:


#from torchvision.transforms import Normalize
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu


# In[ ]:


input_size = 224
batch_video = 2
num_frame = 3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = Compose([  
    #RandomBrightnessContrast(p=0.5, brightness_limit=0.5, contrast_limit=0.5),
    Normalize(mean=mean, std=std, p=1),
    ToTensor()
])
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'


# # NextVLAD

# In[ ]:


#!pip install pytorchcv --quiet
device = gpu
#checkpoint = torch.load('se_resnext50.pth',map_location=device)
net = se_resnext50_32x4d(num_classes=1, pretrained=None).to(device)
model = net.to(device)

model = nn.Sequential(*list(model.children())[:-1])
model[-1] = nn.Sequential(nn.AdaptiveAvgPool2d(1)).to(device)
base_model = model


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name
#freeze_until(base_model, "#4.2.conv1.weight") #freezen all layer


class SE_ContextGating(nn.Module):
    def __init__(self, vlad_dim, hidden_size, drop_rate=0.5,gating_reduction=8):

        super(SE_ContextGating, self).__init__()
        
        self.fc1 = nn.Linear(vlad_dim,hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        self.bn1 = nn.BatchNorm1d(hidden_size)      
        self.gate = torch.nn.Sequential(    
            nn.Linear(hidden_size,hidden_size//gating_reduction),
            nn.BatchNorm1d(hidden_size//gating_reduction),
            nn.ReLU(),
            
            nn.Linear(hidden_size//gating_reduction,hidden_size),
            nn.Sigmoid()
        )
               
    def forward(self,x):
        x = self.bn1(self.dropout(self.fc1(x)))
        gate = self.gate(x)
        activation = x * gate
        return activation

class NextVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64,expansion=2,group=8, dim=2048,num_class=1):

        super(NextVLAD, self).__init__()
        
        self.num_clusters = num_clusters
        self.expansion = expansion
        self.group = group
        self.dim = dim
        self.bn1 = nn.BatchNorm1d(group*num_clusters)
        self.bn2 = nn.BatchNorm1d(num_clusters*expansion*dim//group)
        self.centroids1 = nn.Parameter(torch.rand(expansion*dim, group*num_clusters))
        self.centroids2 = nn.Parameter(torch.rand(1,expansion*dim//group,num_clusters ))
        self.fc1 = nn.Linear(dim,expansion*dim) 
        self.fc2 = nn.Linear(dim*expansion,group)
        
        self.cg = SE_ContextGating(num_clusters*expansion*dim//group,dim)
        self.fc3 = nn.Linear(dim,num_class)

    def forward(self, x): #2,4,2048,1,1
        
        max_frames = x.size(1)
        x = x.view(x.size()[:3]) #2,4,2048
        
        x_3d = F.normalize(x, p=2, dim=2)  # across descriptor dim, torch.Size([2,4, 2048, 1, 1])
        
        vlads = []
        for t in range(x_3d.size(0)):
            x = x_3d[t, :, :] #4,2048
            x = self.fc1(x) #expand, 4,2*2048
            
            #attention
            attention = torch.sigmoid(self.fc2(x)) # 4,8
            attention = attention.view(-1, max_frames*self.group, 1) 
            
            feature_size = self.expansion * self.dim // self.group
            #reshaped_input = tf.reshape(input, [-1, self.expansion * self.feature_size])
            reshaped_input = x.view(-1,self.expansion *self.dim) # 4,2*2048
            #activation = tf.matmul(reshaped_input, cluster_weights)
            activation = torch.mm(reshaped_input, self.centroids1) # 4,8*32
            activation = self.bn1(activation)
            #activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.cluster_size])
            activation = activation.view(-1,max_frames*self.group,self.num_clusters) # 1,32,32
            #activation = tf.nn.softmax(activation, axis=-1)
            activation = F.softmax(activation, dim=-1)  # 1,32,32
            #activation = tf.multiply(activation, attention)
            activation = activation * attention # 1,32,32
            #a_sum = tf.sum(activation, -2, keep_dims=True)
            a_sum = activation.sum(dim=-2, keepdim=True) #1,32,1
            
            #a = tf.multiply(a_sum, cluster_weights2)
            a = a_sum * self.centroids2 # 1,512,32 (512=dim*expansion//group,32=clusters)
            #activation = tf.transpose(activation, perm=[0, 2, 1])
            activation = activation.permute(0, 2, 1) #1,32,1
            #reshaped_input = tf.reshape(input, [-1, self.max_frames * self.groups, feature_size])
            reshaped_input = x.view(-1,max_frames*self.group,feature_size) # 1,32,512
            vlad = torch.bmm(activation, reshaped_input) # 1,32,512
            #vlad = tf.transpose(vlad, perm=[0, 2, 1])
            vlad = vlad.permute(0,2,1)
            #vlad = tf.subtract(vlad, a)
            vlad = vlad - a # 1,512,32
            #vlad = tf.nn.l2_normalize(vlad, 1)
            vlad = F.normalize(vlad,p=2,dim=1)
            #vlad = tf.reshape(vlad, [-1, self.cluster_size * feature_size])
            vlad = vlad.view(self.num_clusters*feature_size) #[1, 16384]
            vlads.append(vlad)
        vlads = torch.stack(vlads, dim=0)
        vlads = self.bn2(vlads) #[2, 16384]
        
        x = self.cg(vlads) #SE Context Gating
        x = self.fc3(x) 

        return x
        
    
            
    
class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        frame_feature = []
        for t in range(x.size(1)):
            feature = self.base_model(x[:, t, :, :, :])
            frame_feature.append(feature)
        frame_feature = torch.stack(frame_feature, dim=0).transpose_(0, 1)
        embedded_x = self.net_vlad(frame_feature)
        
        return embedded_x
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


next_vlad = NextVLAD(num_clusters=32,expansion=2,group=8, dim=2048)
model = EmbedNet(model, next_vlad).to(device)
checkpoint = torch.load('/kaggle/input/se50nextv1/3.24_0.pth',map_location=device)
model.load_state_dict(checkpoint)
_ = model.eval()
del checkpoint


# # MTCNN

# In[ ]:


class Detector:
    def __init__(self, net, image_size = 224):
        self.net = net
        self.image_size = image_size
    def __call__(self, frames):
        return self.net.detect(frames)
    
    def face_resize(self, face):
        h, w = face.shape[:2]
        if w != h:
            size = max(w, h)
            offset_w, offset_h = size - w, size - h
            face = cv2.copyMakeBorder(face, offset_h//2, offset_h - offset_h//2, offset_w//2, offset_w - offset_w//2, cv2.BORDER_CONSTANT)
        return cv2.resize(face, (self.image_size, self.image_size), cv2.INTER_LINEAR)
        
    def recover_in_original_frame(self, frames, boxes, resize):
        boxes =  [[(box/resize).astype(int) for box in i] if i is not None else i for i in boxes]
        return [[self.face_resize(frame[box[1]:box[3], box[0]:box[2]]) for box in boxes_perframe] if boxes_perframe is not None else None for frame, boxes_perframe in zip(frames, boxes)]
    
    def make_square(self, boxes, frame_size):
        for i in range(len(boxes)):
            boxes_perframe = []
            if boxes[i] is not None:
                for j in range(len(boxes[i])):
                    box = boxes[i][j]
                    w, h = (box[2] - box[0], box[3] - box[1])
                    size = max(w, h)
                    centre = int(box[0] + w/2), int(box[1] + h/2)

                    xmin = np.clip(centre[0] - size//2, 0, frame_size[1])
                    xmax = np.clip(centre[0] + size//2, 0, frame_size[1])
                    ymin = np.clip(centre[1] - size//2, 0, frame_size[0])
                    ymax = np.clip(centre[1] + size//2, 0, frame_size[0])

                    boxes_perframe.append([xmin, ymin, xmax, ymax])
                boxes_perframe = np.array(boxes_perframe)

            else:
                boxes_perframe = None
                
            boxes[i] = boxes_perframe
        return boxes
    
    
    
    def add_margin(self, boxes, frame_size, margin = 0.2):
        '''
        margin is a number bewteen [0. 1] or a numpy array which contains 4 elements
        [xmin, ymin, xmax, ymax]
        '''
        if isinstance(margin, float):
            margin = np.array([margin] *4)
        for i in range(len(boxes)):
            boxes_perframe = []
            if boxes[i] is not None:
                for j in range(len(boxes[i])):
                    box = boxes[i][j]
                    w, h = (box[2] - box[0], box[3] - box[1])

                    xmin = np.clip(box[0] - w*margin[0], 0, frame_size[1])
                    xmax = np.clip(box[2] + w*margin[2], 0, frame_size[1])
                    ymin = np.clip(box[1] - h*margin[1], 0, frame_size[0])
                    ymax = np.clip(box[3] + h*margin[3], 0, frame_size[0])

                    boxes_perframe.append([xmin, ymin, xmax, ymax])
                boxes_perframe = np.array(boxes_perframe)
            else:
                boxes_perframe = None
            boxes[i] = boxes_perframe
        return boxes



class PathPipeline:
    def __init__(self, root = './'):
        self.root = root
    def __call__(self, filename):
        return f'{self.root}/{filename}'

class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""
    
    def __init__(self, detector, n_frames=None, batch_size=60, resize=None, num = 10):
        """Constructor for DetectionPipeline class.
        
        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
            num: num is the number of faces returned
        """
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
        self.num = num
    
    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        
        
        
        
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.arange(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        faces = []
        probs = []
        frames = []
        frames_small = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_small = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame_small = frame_small.resize([int(d * self.resize) for d in frame_small.size])
                frames_small.append(frame_small)
                
                frames.append(frame)

                # When batch is full, detect faces and reset frame list
                if len(frames_small) % self.batch_size == 0 or j == sample[-1]:
                    
                    result = [list(zip(*sorted(zip(*i), key = lambda x:x[1], reverse = True)) if i[0] is not None else i)  for i in zip(*self.detector(frames_small))]
                    boxes = [i[0] for i in result]
                    W, H = frames_small[0].size
                    boxes = self.detector.add_margin(boxes, (H, W), margin = [0.2, 0.4, 0.2, 0.2])
                    boxes = self.detector.make_square(boxes, (H, W))
                    
                    faces = self.detector.recover_in_original_frame(frames, boxes, self.resize)
                    faces = list(filter(None,faces))
                    faces = [train_transform(image=np.array(face[0]))['image'] for face in faces]

                    frames_small = []
                    frames = []

        v_cap.release()
        return faces


# In[ ]:


mtcnn = MTCNN(image_size = 224, margin=40, post_process=False, keep_all=False, factor=0.5, device=device).eval()
detector = DetectionPipeline(detector=Detector(mtcnn), n_frames=6, batch_size=54, resize=0.25)


#  ## Prediction loop

# In[ ]:


def FaceExtractor(video_path,detector):
    faces = detector(os.path.join(test_dir,video_path))
    faces = faces[:((len(faces)//num_frame)//batch_video)*num_frame*batch_video]
    faces = torch.stack(faces,dim=0).view(-1,num_frame,3,224,224)
    return faces
def predict_on_video(video_path):
    try:
        predicts = []
        faces = FaceExtractor(video_path,detector) #torch.Size([16, 3, 3, 224, 224])
        if len(faces)>0:
            for i in range(faces.size(0)//batch_video):         
                batch = faces[batch_video*i:batch_video*i+batch_video].to(device)
                predict = torch.sigmoid(model(batch)).cpu().detach().numpy()
                if (predict>0.5).sum() == batch_video:
                    return predict.mean()
                predicts.extend(predict)
            return np.mean(predicts)
    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5


# In[ ]:


from concurrent.futures import ThreadPoolExecutor

def predict_on_video_set(videos, num_workers):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(os.path.join(test_dir, filename))
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))

    return list(predictions)


# ## Speed test
# 
# The leaderboard submission must finish within 9 hours. With 4000 test videos, that is `9*60*60/4000 = 8.1` seconds per video. So if the average time per video is greater than ~8 seconds, the kernel will be too slow!

# In[ ]:


speed_test = True  # you have to enable this manually

if speed_test:
    start_time = time.time()
    speedtest_videos = test_videos[:1]
    predictions = predict_on_video_set(speedtest_videos, num_workers=4)
    elapsed = time.time() - start_time
    print("Elapsed %f sec. Average per video: %f sec." % (elapsed, elapsed / len(speedtest_videos)))


# ## Make the submission

# In[ ]:


predictions = predict_on_video_set(test_videos, num_workers=4)


# In[ ]:


submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
#submission_df['label'] = submission_df['label'].clip(0.9,0.1)
submission_df.to_csv("submission.csv", index=False)

