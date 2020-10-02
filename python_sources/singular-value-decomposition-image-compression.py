#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from random import randrange
import os
from PIL import Image
import matplotlib.pyplot as plt


# # Load Data

# In[ ]:


filenames = os.listdir('../input/butterfly-dataset/leedsbutterfly/images')


# Load random image in dataset and convert to grayscale

# In[ ]:


img = Image.open('../input/butterfly-dataset/leedsbutterfly/images/' + filenames[randrange(len(filenames))]).convert('L').resize((256,256))


# In[ ]:


plt.imshow(img, cmap='gray')


# # Compute Singular Value Decomposition

# In[ ]:


U,S,V = np.linalg.svd(np.array(img), full_matrices=False)
S = np.diag(S)


# The plot below represents the energy each singular vector adds to the overall image

# In[ ]:


plt.plot(np.cumsum(S)/np.sum(S))
plt.title('Cumulative Sum of Sigma Matrix')


# # Image Reconstruction

# The energy in the first *I* columns can be caluculated using the following formula

# ![energy](https://i.imgur.com/VyUtygs.png)

# # First 5 Columns

# Only keeping the first 5 columns of the SVD matrices, truncating the rest and reconstructing the original image

# In[ ]:


r = 5
reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]


# In[ ]:


energy = 0
for i in range(r):
    energy = energy + S[i][i]*S[i][i]
energy = energy / np.sum(np.square(S))
print('The first ' + str(r) + ' columns contained ' + str(energy * 100) + '% of the original energy of the image')


# In[ ]:


plt.imshow(reconstruction,cmap='gray')


# # First 10 Columns

# In[ ]:


r = 10
reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]


# In[ ]:


energy = 0
for i in range(r):
    energy = energy + S[i][i]*S[i][i]
energy = energy / np.sum(np.square(S))
print('The first ' + str(r) + ' columns contained ' + str(energy * 100) + '% of the original energy of the image')


# In[ ]:


plt.imshow(reconstruction,cmap='gray')


# # First 25 Columns

# In[ ]:


r = 25
reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]


# In[ ]:


energy = 0
for i in range(r):
    energy = energy + S[i][i]*S[i][i]
energy = energy / np.sum(np.square(S))
print('The first ' + str(r) + ' columns contained ' + str(energy * 100) + '% of the original energy of the image')


# In[ ]:


plt.imshow(reconstruction,cmap='gray')


# # First 50 Columns

# In[ ]:


r = 50
reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]


# In[ ]:


energy = 0
for i in range(r):
    energy = energy + S[i][i]*S[i][i]
energy = energy / np.sum(np.square(S))
print('The first ' + str(r) + ' columns contained ' + str(energy * 100) + '% of the original energy of the image')


# In[ ]:


plt.imshow(reconstruction,cmap='gray')


# # Other Images

# In[ ]:


img = Image.open('../input/butterfly-dataset/leedsbutterfly/images/' + filenames[randrange(len(filenames))]).convert('L').resize((256,256))
print('Original')
plt.imshow(img, cmap='gray')


# Using first 50 columns

# In[ ]:


U,S,V = np.linalg.svd(np.array(img), full_matrices=False)
S = np.diag(S)
r = 50
reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]
print('Reconstruction')
plt.imshow(reconstruction,cmap='gray')


# In[ ]:


img = Image.open('../input/butterfly-dataset/leedsbutterfly/images/' + filenames[randrange(len(filenames))]).convert('L').resize((256,256))
print('Original')
plt.imshow(img, cmap='gray')


# In[ ]:


U,S,V = np.linalg.svd(np.array(img), full_matrices=False)
S = np.diag(S)
r = 50
reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]
print('Reconstruction')
plt.imshow(reconstruction,cmap='gray')

