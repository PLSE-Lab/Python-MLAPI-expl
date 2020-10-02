#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os # accessing directory structure
import random
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import matplotlib.image as mpimg


# In[ ]:


listImageFilePaths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        listImageFilePaths.append( os.path.join(dirname, filename))
        
print('Total images ', len(listImageFilePaths))


# In[ ]:


def displayImage(imagePath):
    image = mpimg.imread(imagePath)
    plt.imshow(image)
    print('Image path', imagePath)
#     print('Image size -', image.shape)
    plt.show()


# In[ ]:


def displayRandomImages(listImageFilePaths, nImageToDisplay = 5):
    listImageIndex = random.sample(range(0, len(listImageFilePaths)), nImageToDisplay)
    
    for imageIndex in listImageIndex:
        displayImage(listImageFilePaths[imageIndex])


# In[ ]:


displayRandomImages(listImageFilePaths, nImageToDisplay = 5)


# In[ ]:




