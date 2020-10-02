#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


sys.path.append("/kaggle/input/mtcnngithub/mtcnn-master/")


# In[ ]:


from mtcnn import MTCNN
#from mtcnn.mtcnn import MTCNN also works


# In[ ]:


img = cv2.cvtColor(cv2.imread("/kaggle/input/mtcnngithub/mtcnn-master/ivan.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()


# In[ ]:


detector.detect_faces(img)


# In[ ]:




