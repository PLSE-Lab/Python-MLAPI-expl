#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('mode.chained_assignment', None)


# Let us convert the problem into a standard Neural Network Image problem whereby we take an input image and produce an output image - have a look at the many kaggle comptetions that use RLE or run length encoding!
# 
# The idea here is to create "images" for every molecule
# 
# We will then create a target output "image" for each molecule and you should then train your favourite NN to reproduce the output target images
# 
# Once you have done that, you can grab the actual scalar coupling constants by just looking up there values from the 2d matrix output from your neural network
# 
# (As these matrices are symmetric you should average over (x,y) and (y,x) indices)
# 
# Note I am producing the smallest possible matrices to save memory - it will be up to you to create a generator to expand them to 29 by 29.
# 
# Good Luck

# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


structures = pd.read_csv('../input/structures.csv')


# In[ ]:


train.head()


# In[ ]:


structures.head()


# In[ ]:


rawimagedata = {}
sizesofmatrices = {}

for k,groupdf in tqdm((structures.groupby('molecule_name'))):
    # I am just mapping the atom types to numerics as an example feel free to one hot encode them
    groupdf.atom =  groupdf.atom.map({'H': 1, 'C': 2, 'N':3,'O':4,'F':5})
    inputimage = groupdf.merge(groupdf,on=['molecule_name'],how='outer')
    #Fermi Contact seems to love r^-3!
    inputimage['recipdistancecubed'] = 1/np.sqrt((inputimage.x_x-inputimage.x_y)**2+
                                                 (inputimage.y_x-inputimage.y_y)**2+
                                                 (inputimage.z_x-inputimage.z_y)**2)**3
    inputimage.recipdistancecubed = inputimage.recipdistancecubed.replace(np.inf,0)
    sizesofmatrices[k] = int(math.sqrt(inputimage.shape[0]))
    rawimagedata[k] = inputimage[['atom_x','atom_y','recipdistancecubed']].values.reshape(sizesofmatrices[k],sizesofmatrices[k],3)
    break


# In[ ]:


targetimages = {}
for k,groupdf in tqdm((train.groupby('molecule_name'))):

    outputimage = pd.DataFrame({'molecule_name':k,'atom_index':np.arange(sizesofmatrices[k])})
    outputimage = outputimage.merge(outputimage,on=['molecule_name'],how='outer')
    outputimage = outputimage.merge(groupdf,
                                    left_on=['molecule_name','atom_index_x','atom_index_y'],
                                    right_on=['molecule_name','atom_index_0','atom_index_1'],how='left')
    outputimage = outputimage.merge(groupdf,
                                    left_on=['molecule_name','atom_index_x','atom_index_y'],
                                    right_on=['molecule_name','atom_index_1','atom_index_0'],how='left')
    outputimage['sc'] = outputimage.scalar_coupling_constant_x.fillna(0)+outputimage.scalar_coupling_constant_y.fillna(0)
    targetimages[k] = outputimage[['sc']].values.reshape(sizesofmatrices[k],sizesofmatrices[k])
    break


# In[ ]:


rawimagedata['dsgdb9nsd_000001']


# In[ ]:


targetimages['dsgdb9nsd_000001']


# Note the output target matrix is symmetric so you will get better results averaging (x,y) with (y,x)

# In[ ]:




