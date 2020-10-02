#!/usr/bin/env python
# coding: utf-8

# Combining faces of top players in fifa
# --------------------------------------
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# **Using the weighted addition function opencv** (150 images)

# In[ ]:


import numpy as np
import cv2
import os

base = '../input/Pictures/'
images = os.listdir(base)
images = images[:150]
output = cv2.imread(base+images[0])
image1 = cv2.imread(base+images[1])
cv2.addWeighted(image1, 1.0/len(images), output, 1.0/len(images), 0, output)

for i in range(2,len(images)):

	# load the image
	image1 = cv2.imread(base+images[i])
	cv2.addWeighted(image1, 1.0/len(images), output, 1, 0, output)
cv2.imwrite("Output1.jpg", output)


# In[ ]:


from matplotlib.pyplot import imshow
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')
pil_im = Image.open('Output1.jpg', 'r')
imshow(np.asarray(pil_im))


# **Using the weighted addition function opencv** (200 images)

# In[ ]:


images = os.listdir(base)
images = images[:200]
output = cv2.imread(base+images[0])
image1 = cv2.imread(base+images[1])
cv2.addWeighted(image1, 1.0/len(images), output, 1.0/len(images), 0, output)

for i in range(2,len(images)):

	# load the image
	image1 = cv2.imread(base+images[i])
	cv2.addWeighted(image1, 1.0/len(images), output, 1, 0, output)
cv2.imwrite("Output2.jpg", output)


# In[ ]:


from matplotlib.pyplot import imshow
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')
pil_im = Image.open('Output2.jpg', 'r')
imshow(np.asarray(pil_im))


# **Using the weighted addition function opencv** (200 images constant weighted)

# In[ ]:


images = os.listdir(base)
images = images[:200]
output = cv2.imread(base+images[0])
image1 = cv2.imread(base+images[1])
cv2.addWeighted(image1, 0.01, output, 0.01, 0, output)

for i in range(2,len(images)):

	# load the image
	image1 = cv2.imread(base+images[i])
	cv2.addWeighted(image1, 0.01, output, 1, 0, output)
cv2.imwrite("Output3.jpg", output)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pil_im = Image.open('Output3.jpg', 'r')
imshow(np.asarray(pil_im))


# **Using the weighted addition function opencv** (300 images)

# In[ ]:


images = os.listdir(base)
images = images[:300]
output = cv2.imread(base+images[0])
image1 = cv2.imread(base+images[1])
cv2.addWeighted(image1, 0.0033333, output, 0.0033333, 0, output)

for i in range(2,len(images)):

	# load the image
	image1 = cv2.imread(base+images[i])
	cv2.addWeighted(image1, 0.0033333, output, 1,0, output)
cv2.imwrite("Output4.jpg", output)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pil_im = Image.open('Output4.jpg', 'r')
imshow(np.asarray(pil_im))


# **Using the weighted addition function opencv** (all images)

# In[ ]:


images = os.listdir(base)
output = cv2.imread(base+images[0])
image1 = cv2.imread(base+images[1])
cv2.addWeighted(image1, 0.0035, output, 0.0035, 0, output)

for i in range(2,len(images)):

	# load the image
	image1 = cv2.imread(base+images[i])
	cv2.addWeighted(image1, 0.0035, output, 1, 0, output)
cv2.imwrite("Output5.jpg", output)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pil_im = Image.open('Output5.jpg', 'r')
imshow(np.asarray(pil_im))


# Combining faces of top female players
# -------------------------------------

# In[ ]:


base = '../input/Pictures_f/'
images = os.listdir(base)
output = cv2.imread(base+images[0])
image1 = cv2.imread(base+images[1])
cv2.addWeighted(image1, 0.01, output, 0.01, 0, output)

for i in range(2,len(images)):

	# load the image
	image1 = cv2.imread(base+images[i])
	cv2.addWeighted(image1, 0.01, output, 1, 0, output)
cv2.imwrite("Output6.jpg", output)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pil_im = Image.open('Output6.jpg', 'r')
imshow(np.asarray(pil_im))


# Combining faces of top female players and male players
# -------------------------------------

# In[ ]:


base = '../input/Pictures_f/'
base1 = '../input/Pictures/'
images = os.listdir(base)
images2 = os.listdir(base1)
output = cv2.imread(base+images[0])
image1 = cv2.imread(base+images[1])
div = 0.0035
cv2.addWeighted(image1, div, output, div, 0, output)

for i in range(2,len(images)):

	# load the image
	image1 = cv2.imread(base+images[i])
	cv2.addWeighted(image1, div, output, 1, 0, output)
 
for i in range(len(images2)):
    # load the image
	image1 = cv2.imread(base1+images2[i])
	cv2.addWeighted(image1, div, output, 1, 0, output)
    
cv2.imwrite("Output7.jpg", output)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pil_im = Image.open('Output7.jpg', 'r')
imshow(np.asarray(pil_im))

