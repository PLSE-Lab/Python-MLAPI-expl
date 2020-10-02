#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install "/kaggle/input/kornia020/kornia-0.2.0e110f3b-py2.py3-none-any.whl"


# In[ ]:


import torch
import torch.nn as nn

import kornia

import cv2
import numpy as np


# In[ ]:


from matplotlib import pyplot as plt

def imshow(input: torch.Tensor):
    #out: torch.Tensor = torchvision.utils.make_grid(input, nrow=4, padding=0)
    out = input[:1]
    out_np: np.array = kornia.tensor_to_image(out)
    plt.imshow(out_np); plt.axis('off');


# In[ ]:


# load video and preprocess
def load_video(video_path: str, num_frames: int) -> torch.Tensor:
    print(video_path)
    # TODO: need to install torchvision 0.5
    # vframes, aframe, info = torchvision.io.read_video('data/deep_fake.mp4')

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Error opening video stream or file")

    # Read until video is completed
    frames_list: List[np.ndarray] = []
    
    count: int = 0
    while(cap.isOpened() and count < num_frames):
        # Capture frame-by-frame
        ret, frame_tmp = cap.read()
        frames_list.append(frame_tmp)
        count += 1

    # create numpy array BxHxWxC
    frames_np: np.ndarray = np.array(frames_list)

    # we need the tensor to be in the shape BxCxHxW
    return kornia.image_to_tensor(frames_np)


# In[ ]:


# define data augmentation pipeline
# Please, check new data augmentation API:
# https://kornia.readthedocs.io/en/latest/augmentation.html
transforms = nn.Sequential(
    kornia.geometry.Resize(256),
    kornia.color.Normalize(mean=0., std=255.),
    kornia.augmentation.RandomHorizontalFlip(),
    kornia.augmentation.RandomGrayscale(),
)


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input/deepfake-detection-challenge/'):
    for filename in filenames:
        if filename[-3:] == "csv": continue
        file_path: str = os.path.join(dirname, filename)

        # load a video
        frames: torch.Tensor = load_video(file_path, num_frames=30)
        print(frames.shape)
        
        imshow(frames)
        
        # send data to CUDA device
        if torch.cuda.is_available():
            frames = frames.cuda()
        
        # perform data augmentation
        frames_out: torch.Tensor = transforms(frames)
        print(frames_out.shape)

# Any results you write to the current directory are saved as output.

