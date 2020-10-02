#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip3 install pretrainedmodels torch==0.4.1')
import glob
import cv2
import numpy as np
import pretrainedmodels
import torch
import time
from tqdm import tqdm

def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def get_image_generator(image_paths, batch_size):
    batch = []
    for path in image_paths:
        batch.append(read_image(path))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0: yield batch

def get_images_in(path):
    return np.sort(glob.glob(f'{path}/*.jpg'))

class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = pretrainedmodels.se_resnet50(pretrained='imagenet')
        self.resnet.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.resnet.last_linear = torch.nn.Linear(self.resnet.last_linear.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

model = Model(1).cuda()
model.eval()

batch = torch.randn(16, 3, 224, 224).cuda()

# PyTorch is asynchronous,
# but when there's no I/O bottleneck, it's forced to wait for previous batch to compute.
results = []
total_time = 0
for i in tqdm(range(125)):
    start_time = time.time()
    results.extend((model(batch) > 0)[:, 0])
    total_time += (time.time() - start_time)
print('Inference Time, automatic synchronization: %0.2f Minutes' % ((total_time)/60))

# When there is an I/O bottleneck, PyTorch can compute batches asynchronously.
# Since this computation intersects with image reading, it's partially ignored by the timing code.
results = []
total_time = 0
for _ in tqdm(get_image_generator(get_images_in('/kaggle/input/test_v2')[:2000], 16), total=125):
    start_time = time.time()
    results.extend((model(batch) > 0)[:, 0])
    total_time += (time.time() - start_time)

print('Inference Time, no synchronization output: %0.2f Minutes' % ((total_time)/60))

# Also, sending the tensor to .extend() doesn't force synchronization
# But converting it to the numpy array does
results = []
total_time = 0
for _ in tqdm(get_image_generator(get_images_in('/kaggle/input/test_v2')[:2000], 16), total=125):
    start_time = time.time()
    results.extend(np.array((model(batch) > 0)[:, 0]))
    total_time += (time.time() - start_time)

print('Inference Time, np.array output: %0.2f Minutes' % ((total_time)/60))

# It's also possible to force synchronization with torch.cuda.synchronize()
results = []
total_time = 0
for _ in tqdm(get_image_generator(get_images_in('/kaggle/input/test_v2')[:2000], 16), total=125):
    start_time = time.time()
    results.extend((model(batch) > 0)[:, 0])
    torch.cuda.synchronize()
    total_time += (time.time() - start_time)

print('Inference Time, torch.cuda.synchronize output: %0.2f Minutes' % ((total_time)/60))


# In[ ]:




