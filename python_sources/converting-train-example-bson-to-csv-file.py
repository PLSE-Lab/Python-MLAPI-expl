#!/usr/bin/env python
# coding: utf-8

# This notebook works towards converting the weird sorted file of tarin_example.bson into diffulet csv dataset where the first column is the category id and the rest  column from 1 till column (180x180x3) "the number of pixel in each image" is setted as the pixel value of the affilated image. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bson
import io
from skimage.data import imread

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))

df=pd.DataFrame({"category":[]})
df=pd.concat([df,pd.DataFrame(columns=[i for i in range(180*180*3)])])
df['category'].astype('object')
index=0
pd.options.display.precision=11

for i,j in enumerate(data):
    for k in range(len(j['imgs'])):
        image=np.reshape((imread(io.BytesIO(j['imgs'][k]['picture']))),-1)
        image=image.tolist()
        image.insert(0,j["category_id"])
        df.loc[index]=image
        index=index+1    
            
        
df.to_csv("train_example.csv",index=False)
# Any results you write to the current directory are saved as output.


# Now you have a typical dataset where the first columns is the category id and the reset columns are the affilated featuers as shown in the following output.

# In[ ]:


print(df.head(10))

