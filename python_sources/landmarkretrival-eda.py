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

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('../input/landmark-retrieval-2020/train.csv')


# In[ ]:


data.info()


# In[ ]:


print(f"Columns that we are using are :- {data.columns[0]}, {data.columns[1]}")


# In[ ]:


print(f"Total Numbers of the id or images {data.shape[0]}, with unique Landmarks {len(set(data['landmark_id']))}")


# Here we are having nearly 1.58 Million images and 81.3k LandMark ids

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


landmarks = data['landmark_id']
ids = data['id']


# In[ ]:


#ids, landmarks


# In[ ]:


data['count'] = data["landmark_id"].value_counts()


# In[ ]:


data_count=data['count'].dropna()


# In[ ]:


xval = data_count.values


# In[ ]:


yval = np.array(list(set(landmarks)))


# In[ ]:


len(xval[:20]), len(yval[:20])


# In[ ]:


type(xval), type(yval)


# In[ ]:


import matplotlib.pyplot as plt

ax = plt.figure(figsize=(30, 10))
plt.hist(xval, bins='auto')
ax.show()


# In[ ]:


fig = plt.figure(figsize=(30, 10))
plt.plot(yval, xval, c = 'r', lw=2)
fig.show()


# **See Images How it is**
# 
# This list will Exaplain about the normal Looping or Manual Inputs
# Here you can see images of the 0 class

# In[ ]:


import os

images = ['../input/landmark-retrieval-2020/index/0/0/0/'+i for i in os.listdir('../input/landmark-retrieval-2020/index/0/0/0') if i.endswith('.jpg')]


# In[ ]:


images


# In[ ]:


import cv2
import matplotlib.pyplot as plt
import numpy as np

w=10
h=10
fig=plt.figure(figsize=(10, 8))
columns = 4
rows = 5
for i in range(1, len(images)):
    img = cv2.imread(images[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()


# **Automated Images Looping**
# 
# 
# Here learn about the How to loop automatically through the images 
# 
# Just go through the Nested list

# In[ ]:


lis = [f'../input/landmark-retrieval-2020/index/{l}/{j}/{k}/'+ i for l in range(2) for j in range(2) for k in range(1) for i in os.listdir(f'../input/landmark-retrieval-2020/index/{l}/{j}/{k}') if i.endswith(".jpg")]


# In[ ]:


len(lis)


# In[ ]:


import cv2
import matplotlib.pyplot as plt
import numpy as np

w=10
h=10
fig=plt.figure(figsize=(15, 15))
columns = 10
rows = 8
for i in range(1, len(lis)):
    img = cv2.imread(lis[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.title(lis[i])
plt.show()


# **Printing images with the Title**
# 
# Learn how to print images with the title 
# 
# Here i gave image name as the title

# In[ ]:


import cv2
import matplotlib.pyplot as plt
import numpy as np

w=10
h=10
fig=plt.figure(figsize=(15, 15))
columns = 4
rows = 4
for i in range(1, len(lis[:9])):
    img = cv2.imread(lis[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.title(lis[i][-20:])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




