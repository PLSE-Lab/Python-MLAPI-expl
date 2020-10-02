#!/usr/bin/env python
# coding: utf-8

# # Use imgdupes for the detection of duplicates with exact neighbor searching using faiss
# 1.  In the training set
# 2. In the test set 
# 3. Between the above sets

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system(' conda install -y faiss-gpu cudatoolkit=10.0 -c pytorch')


# In[ ]:


get_ipython().system('  pip install imgdupes')


# In[ ]:


get_ipython().system(' ls -la ../input/')


# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().system(' imgdupes --recursive  --faiss-flat "../input/test_images" phash 1')


# # Let us see if these are really duplicates
# 
# ../input/test_images/01c31b10ab99.png
# ../input/test_images/b29bd35acaf6.png
# 
# ../input/test_images/13e28ec4534a.png
# ../input/test_images/6543b4168f98.png
# 
# ../input/test_images/1466c7e5936e.png
# ../input/test_images/23c5eba92749.png
# 
# ../input/test_images/d00312c50737.png
# ../input/test_images/1822b6c60784.png
# 
# ../input/test_images/2fb539602f57.png
# ../input/test_images/80aa9b30d2f9.png
# 
# ../input/test_images/417d3908ee21.png
# ../input/test_images/9d9de8c9afb5.png
# 
# ../input/test_images/4247b91698fc.png
# ../input/test_images/a2319c2af727.png
# 
# ../input/test_images/aa381cb2abd2.png
# ../input/test_images/9bd683e16325.png

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')

# figure size in inches optional
rcParams['figure.figsize'] = 15 ,15
def plotTwo(img_A,img_B):
    # read images    
    # display images
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img_A);
    ax[1].imshow(img_B);


# In[ ]:


img_A = mpimg.imread('../input/test_images/01c31b10ab99.png')
img_B = mpimg.imread('../input/test_images/b29bd35acaf6.png')
plotTwo (img_A,img_B)


# In[ ]:


img_A = mpimg.imread('../input/test_images/417d3908ee21.png')
img_B = mpimg.imread('../input/test_images/9d9de8c9afb5.png')
plotTwo (img_A,img_B)


# # So you get the idea here. 

# In[ ]:




