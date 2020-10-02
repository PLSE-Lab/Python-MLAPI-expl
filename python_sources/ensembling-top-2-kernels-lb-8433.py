#!/usr/bin/env python
# coding: utf-8

# #### This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 
# 
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# 
# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# 
# import os
# 
# # Any results you write to the current directory are saved as output.
# print(os.listdir("../input/ensemble"))
# 

# **kernel 1 : https://www.kaggle.com/meaninglesslives/unet-plus-plus-with-efficientnet-encoder**
# 
# **kernel 2 : https://www.kaggle.com/raddar/sample-submission-leak**
# 
# ref : https://www.kaggle.com/vaishvik25/ensemble-v2

# In[ ]:




import pandas as pd
import numpy as np

xf2 = pd.read_csv("../input/ensemble/raddar.csv")

xf1 = pd.read_csv("../input/ensemble/sid.csv")

xf1.columns = ['ImageId', 'enc1']
xf2.columns = ['ImageId', 'enc2']

xf3 = pd.merge(left = xf1, right = xf2, on = 'ImageId', how = 'inner')
print(xf1.shape, xf2.shape, xf3.shape)



# identify the positions where xf1 has empty predictions but xf2 does not
xf3[xf3['enc1'] != xf3['enc2']]
id1 = np.where(xf3['enc1'] == '-1')[0]
id2 = np.where(xf3['enc2'] != '-1')[0]
idx = np.intersect1d(id1,id2)

# map non-empty xf2 slots to empty ones in xf1
xf3['EncodedPixels'] = xf3['enc1']
xf3['EncodedPixels'][idx] = xf3['enc2'][idx]

xf3[['ImageId','EncodedPixels']].to_csv('hybrid_1_2.csv', index = False)

