#!/usr/bin/env python
# coding: utf-8

# ## EfficientNet
# 
# Since AlexNet won the 2012 ImageNet competition, CNNs (short for Convolutional Neural Networks) have become the de facto algorithms for a wide variety of tasks in deep learning, especially for computer vision. From 2012 to date, researchers have been experimenting and trying to come up with better and better architectures to improve models accuracy on different tasks. Today, we will take a deep dive into the latest research paper, EfficientNet, which not only focuses on improving the accuracy, but also the efficiency of models.
# 
# ![Screenshot_2020-01-09%20EfficientNet%20Rethinking%20Model%20Scaling%20for%20Convolutional%20Neural%20Networks.png](attachment:Screenshot_2020-01-09%20EfficientNet%20Rethinking%20Model%20Scaling%20for%20Convolutional%20Neural%20Networks.png)
# 
# To read more about EfficientNet, check out:
# - [paper](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)
# - [summary](https://medium.com/@nainaakash012/efficientnet-rethinking-model-scaling-for-convolutional-neural-networks-92941c5bfb95)

# In[ ]:


import json
from PIL import Image
 
import torch
from torchvision import transforms


# In[ ]:


model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)


# In[ ]:


get_ipython().system(' wget https://github.com/qubvel/efficientnet/raw/master/misc/panda.jpg')


# In[ ]:


img = Image.open('panda.jpg')


# In[ ]:


img


# In[ ]:


get_ipython().system(' wget https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/master/examples/simple/labels_map.txt')


# In[ ]:


# Load class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]


# In[ ]:


# Preprocess image
tfms = transforms.Compose([transforms.Resize(1), transforms.CenterCrop(2), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(img).unsqueeze(0)
 
# Perform model inference
model.eval()
with torch.no_grad():
    logits = model(img)


# In[ ]:


preds = torch.topk(logits, k=3).indices.squeeze(0).tolist()
 
for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    print('{:<75} ({:.2f}%)'.format(label, prob*100))


# ## Keep on following trends
