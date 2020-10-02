#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torchvision.models as models
import glob
from PIL import Image
import torch
from sklearn.manifold import TSNE
import random
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)
torch.backends.cudnn.deterministic = True

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:


test_folder = '../input/plates/plates/test'
test_pictures = []
for filename in sorted(glob.glob(f'{test_folder}/*')):
    im=Image.open(filename)
    try:
        assert im.mode == 'RGB', 'Image channel order is not RGB.'
    except:
        continue
    np_im = np.array(im)
    if np_im.shape[0] > np_im.shape[1]:
        np_im = np.swapaxes(np_im, 0, 1)
    test_pictures.append(np.array(im))


# In[ ]:


model = models.resnet50(pretrained=True).to(device)


# In[ ]:


model.eval();


# In[ ]:


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

result = []

for i in test_pictures:
    i_tensor = torch.from_numpy((i / 255. - mean) / std).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
    
    result.append(model(i_tensor).detach().cpu().numpy())


# In[ ]:


result = np.array(result).squeeze()


# In[ ]:


result_embedded = TSNE(n_components=2, random_state=3, perplexity=500).fit_transform(result)


# In[ ]:


plt.figure(figsize=(20, 10))
x = result_embedded[:,0]
y = result_embedded[:,1]
plt.scatter(x, y)

xl = np.linspace(-2, 1, 100)
yl = -xl + 1
plt.plot(xl, yl, '-r')

yl = np.linspace(-3, 0, 100)
xl = yl-yl+1
plt.plot(xl, yl, '-r')

plt.show()


# In[ ]:


result_embedded[(x <= 1) & (x+y <=1)].shape


# In[ ]:


df = pd.read_csv('../input/sample_submission.csv')
df.head()


# In[ ]:


df['label'] = ((x <= 1) & (x+y <=1))


# In[ ]:


df['label'] = df['label'].map(lambda x: 'cleaned' if x else 'dirty')


# In[ ]:


df.to_csv('out.csv', index=False)


# In[ ]:




