#!/usr/bin/env python
# coding: utf-8

# # ABCNet
# 
# [ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network](https://arxiv.org/abs/2002.10200)
# 
# 
# > Scene text detection and recognition has received increasing research attention. Existing methods can be roughly categorized into two groups: character-based and segmentation-based. These methods either are costly for character annotation or need to maintain a complex pipeline, which is often not suitable for real-time applications. Here we address the problem by proposing the Adaptive Bezier-Curve Network (ABCNet). Our contributions are three-fold: 1) For the first time, we adaptively fit arbitrarily-shaped text by a parameterized Bezier curve. 2) We design a novel BezierAlign layer for extracting accurate convolution features of a text instance with arbitrary shapes, significantly improving the precision compared with previous methods. 3) Compared with standard bounding box detection, our Bezier curve detection introduces negligible computation overhead, resulting in superiority of our method in both efficiency and accuracy. Experiments on arbitrarily-shaped benchmark datasets, namely Total-Text and CTW1500, demonstrate that ABCNet achieves state-of-the-art accuracy, meanwhile significantly improving the speed. In particular, on Total-Text, our realtime version is over 10 times faster than recent state-of-the-art methods with a competitive recognition accuracy. 
# 
# 
# # Quick Inference on Total-Text Data

# In[ ]:


get_ipython().system('pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html ')
get_ipython().system('pip install cython pyyaml==5.1')
get_ipython().system("pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")
get_ipython().system('gcc --version')

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())


# In[ ]:


get_ipython().system('pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html')


# In[ ]:


import os, sys

get_ipython().system('git clone https://github.com/aim-uofa/AdelaiDet.git')
os.chdir('AdelaiDet')


# In[ ]:


DATA_DIR = '/kaggle/input'
ROOT_DIR = '/kaggle/working'

sys.path.append(os.path.join(ROOT_DIR, 'AdelaiDet')) 


# In[ ]:


get_ipython().system('python setup.py build develop')


# In[ ]:


get_ipython().system('wget -O tt_attn_R_50.pth https://cloudstor.aarnet.edu.au/plus/s/t2EFYGxNpKPUqhc/download')
get_ipython().system('ls -lh tt_attn_R_50.pth')


# In[ ]:


import cv2
import glob
import matplotlib.pyplot as plt

def process(filename):
    plt.figure(figsize=(25,15))
    plt.imshow(filename)


# In[ ]:


images = [cv2.imread(file) for file in glob.glob('/kaggle/input/totaltextstr/Total-Text/Train/*.jpg')]
print(len(images))
    
i = 0
for file in images:
    process(file)
    i += 1
    if i > 10: break


# In[ ]:


get_ipython().system("mkdir 'output'")
get_ipython().system('python demo/demo.py     --config-file configs/BAText/TotalText/attn_R_50.yaml     --input /kaggle/input/totaltextstr/Total-Text/Train/*     --output /kaggle/working/AdelaiDet/output/     --opts MODEL.WEIGHTS tt_attn_R_50.pth')


# In[ ]:


images = [cv2.imread(file) for file in glob.glob('//kaggle/working/AdelaiDet/output/*.jpg')]

print(len(images))
    
i = 0
for file in images:
    process(file)
    i += 1
    if i > 20: break


# In[ ]:


get_ipython().system('rm -rf /kaggle/working/AdelaiDet')

