#!/usr/bin/env python
# coding: utf-8

# # Baseline submission using Facenet
# 
# This notebook demonstrates how to use the `facenet-pytorch` package to build a rudimentary deepfake detector without training any models.
# The following steps are performed:
# 
# 1. Create pretrained facial detection (MTCNN) and recognition (Inception Resnet) models.
# 1. For each test video, calculate face feature vectors for N faces evenly spaced through each video.
# 1. Calculate the distance from each face to the centroid for its video.
# 1. Use these distances as your means of discrimination.
# 
# For (much) better results, finetune the resnet to the fake/real binary classification task instead - this is just a baseline. Alternatively, I'm sure there is much more interesting things that can be done with the feature vectors.

# ## Install dependencies

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# Install facenet-pytorch\n!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-1.0.1-py3-none-any.whl\n\n# Copy model checkpoints to torch cache so they are loaded automatically by the package\n!mkdir -p /tmp/.cache/torch/checkpoints/\n!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth /tmp/.cache/torch/checkpoints/vggface2_DG3kwML46X.pt\n!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth /tmp/.cache/torch/checkpoints/vggface2_G5aNV2VSMn.pt')


# ## Imports

# In[ ]:


import os
import glob
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')


# ## Create MTCNN and Inception Resnet models
# 
# Both models are pretrained. The Inception Resnet weights will be downloaded the first time it is instantiated; after that, they will be loaded from the torch cache.

# In[ ]:


# Load face detector
mtcnn = MTCNN(device=device).eval()

# Load facial recognition model
resnet = InceptionResnetV1(pretrained='vggface2', num_classes=2, device=device).eval()


# ## Process test videos
# 
# Loop through all videos and pass N frames from each through the face detector followed by facenet. Calculate the distance from the centroid to the extracted feature for each face.

# In[ ]:


filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')
filenames


# In[ ]:


# Get all test videos
filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')

# Number of frames to sample (evenly spaced) from each video
n_frames = 10

X = []
with torch.no_grad():
    for i, filename in enumerate(filenames):
        print(f'Processing {i+1:5n} of {len(filenames):5n} videos\r', end='')
        
        try:
            # Create video reader and find length
            v_cap = cv2.VideoCapture(filename)
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Pick 'n_frames' evenly spaced frames to sample
            sample = np.linspace(0, v_len - 1, n_frames).round().astype(int)
            imgs = []
            for j in range(v_len):
                success, vframe = v_cap.read()
                vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
                if j in sample:
                    imgs.append(Image.fromarray(vframe))
            v_cap.release()
            
            # Pass image batch to MTCNN as a list of PIL images
            faces = mtcnn(imgs)
            
            # Filter out frames without faces
            faces = [f for f in faces if f is not None]
            faces = torch.stack(faces).to(device)
            
            # Generate facial feature vectors using a pretrained model
            embeddings = resnet(faces)
            
            # Calculate centroid for video and distance of each face's feature vector from centroid
            centroid = embeddings.mean(dim=0)
            X.append((embeddings - centroid).norm(dim=1).cpu().numpy())
        except KeyboardInterrupt:
            raise Exception("Stopped.")
        except:
            X.append(None)


# ## Predict classes
# 
# The below weights were selected by following the same process as above for the train sample videos and then using a logistic regression model to fit to the labels. Note that, intuitively, this is not a very good approach as it does nothing to take into account the progression of feature vectors throughout a video, just combines them together using the weights below. This step is provided as a placeholder only; it should be replaced with a more thoughtful mapping from a sequence of feature vectors to a single prediction.

# In[ ]:


#bias = -0.2942
#weight = 0.068235746

submission = []
for filename, x_i in zip(filenames, X):
    prob = np.prod(x_i)
    #if x_i is not None and len(x_i) == 10:
        #prob = 1 / (1 + np.exp(-(bias + (weight * x_i).sum())))
    #else:
        #prob = 0.5
    submission.append([os.path.basename(filename), prob])


# ## Build submission

# In[ ]:


submission = pd.DataFrame(submission, columns=['filename', 'label'])
submission.sort_values('filename').to_csv('submission.csv', index=False)


# In[ ]:


plt.hist(submission.label, 20)
plt.show()
submission


# In[ ]:




