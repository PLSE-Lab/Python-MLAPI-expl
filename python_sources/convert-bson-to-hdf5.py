#!/usr/bin/env python
# coding: utf-8

# ## This Notebook is under construction
# 

# In[ ]:


import pandas as pd
import numpy as np
import bson
import h5py
import os
import tables ##enables hdf tables
from multiprocessing import pool
import concurrent.futures as threading
import cv2 #opencv helpful for storing image as array
import itertools #helps in parallel processing
from matplotlib.colors import ListedColormap
import scipy as scp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time
import xarray as xr


# In[ ]:


path = '../input/'
# path='./'
get_ipython().system('ls "$path"')


# ###### Get specific chunk of data

# In[ ]:


def read_data(bfile,itr_number,CHUNK_SIZE,demo=False):
    """
    INPUTS:
    -------------
    bfile: bson file
    itr_number: the location of the data to be read
    CHUNK_SIZE: size of data to be read
    demo: boolean when True :get dataframe before applying ravel to imgs
    
    OUTPUT  df:
    -----------------
    when demo = True 
    df is a pandas DataFrme with  multi-index ['category_id','_id','img-num','imgs']
    and dtypes = {'category_id': int64,
                    '_id': int64,
                    'img-num': int64,
                    'imgs': ndarray of dtype np.uint8 and dimension(180,180,3)}
    when demo =False
    df is a Data with multindex ['category_id','_id','img-num'] and 
    columns [0, 180*180*3-1] the result of applying ravel_img
    
    Example use:
    -----------------
    read_data ('{}train.bson'.format(path),0,100)
    
    """
    t0 = time.time()
    print('Thread {} starting .....\n'.format(itr_number))

    with open(bfile,'rb') as b:
        iterator = bson.decode_file_iter(b)
        df = pd.DataFrame(list(itertools.islice(iterator,
                                                itr_number*CHUNK_SIZE,
                                                (itr_number+1)*CHUNK_SIZE)))
    #flatten all pictures
    df = df.set_index(['category_id','_id'])['imgs'].apply(
    pd.Series).stack().apply(
        lambda dct:dct['picture']).reset_index().rename(
        columns={0:'imgs','level_2':'img-num'})
    if demo:
        df['imgs'] = df['imgs'].apply(
            lambda d: cv2.imdecode(np.fromstring(d, dtype='uint8'),cv2.IMREAD_COLOR))
    else:
        df['imgs'] = df['imgs'].apply(
            lambda d: ravel_img(cv2.imdecode(np.fromstring(d, dtype='uint8'),cv2.IMREAD_COLOR)))
        df = df.set_index(['category_id','_id','img-num'])['imgs'].apply(pd.Series)
    print('Thread {} read and processed Time: {:.2f}\n'.format(itr_number,time.time()-t0))
    
    return df
    


# ## Flatten Image
# ##### image converted to 1d for storage using the function `ravel_image`
# #### the process is reveresed using `unravel_image`

# In[ ]:


def ravel_img(A):
    """
    #input array is 3d 
    #output array 1d
    """
    return scp.hstack([A[:,:,i].flatten() for i in range(3)])
def unravel_img(A):
    """
    reverse ravel_img on array
    input is 1d array
    output is 3d array
    """
    return scp.dstack([ch.reshape(180,-1)for ch in scp.hsplit(A,3)])


# #### Test the functions

# In[ ]:


fig,axs = plt.subplots(1,2)
axs= axs.flatten()
df = read_data('{}train.bson'.format(path),944560,1,demo=True)
a = df.iloc[0,-1]
b = ravel_img(a)
axs[0].imshow(a)
axs[0].set_title('Original Image')
axs[1].imshow(unravel_img(b))
axs[1].set_title('Retrieved Image')
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()


# #### Count the number of images in train.bson

# In[ ]:


# path = ''
# counter = 0 
# imgs = 0
# with open('{}train.bson'.format(path),'rb') as b:
#         iterator = bson.decode_file_iter(b)
#         for d in iterator:
            
#             if 'category_id' not in d.keys():
#                 break
#             imgs += len(d['imgs'])
#             counter += 1

##output imgs = 12371293
##counter = 7069896


# In[ ]:



def process_th (bfile,itr_number,CHUNK_SIZE,file,key):
    """
   
    INPUTS:
    -------------
    bfile: bson file
    itr_number: the location of the data to be read
    CHUNK_SIZE: size of data to be read
    file: output hdf file 
    key: name of dataset in output hdf file
    OUTPUT  df:
    -----------------
    tuple (itr_number, success)
    itr_number is the input itr_number this can be used to find failed chunks 
    success: True when writing the data is successful
    False otherwise
    Example use:
    -----------------
    process_th ('{}train.bson'.format(path),0,100,'train.h5','train')
    """
    t0 = time.time()
    print('Thread {} writing .....\n'.format(itr_number))
    
    df = read_data(bfile,itr_number,CHUNK_SIZE)
    try:
        
        hdf = h5py.File(file)
        if key not in hdf.keys():
            train = hdf.create_dataset(key, (12371293, 180*180*3),dtype='i1')
        hdf.close()
        hdf = pd.HDFStore(file)
        hdf[key] = df
        hdf.close()
        print('Thread {} finished writing Time: {:.2f}\n'.format(itr_number,time.time()-t0))
        return (itr_number,True)
    except Exception as e:
        print('Thread {} Writing Failed.'.format(itr_number))
        print('\n error: {} \n Time: {:.2f}\n'.format(e,time.time()-t0))
        return (itr_number,False)

# process_th (bfile,itr_number,CHUNK_SIZE,'train.h5','train')


# In[ ]:


n=1#adjust based on number of CPU cores
CHUNK_SIZE=1000 #adjust value according to your resources
REC_SIZE = 7069896
#number of chuncks is the ceiling of record size dvided by chunck size
NUM_CHUNK = 1 +(REC_SIZE//CHUNK_SIZE) 
print('total chunks: {}\n\n'.format(NUM_CHUNK))
bfile = '{}{}'.format(path,'train.bson')
file = 'train.h5'
key = 'train'
t0 = time.time()
with threading.ThreadPoolExecutor(max_workers = n) as exec:
    #all chuncks

    futures = {exec.submit(process_th, bfile,i,CHUNK_SIZE,file,key) for i in range(0,5)}
    
res = [f.result() for f in threading.as_completed(futures)]
print('total time: {:.2f}'.format(time.time()-t0))


# ## read data

# In[ ]:


hdf = pd.HDFStore(file)
df = hdf[key]
hdf.close()


# In[ ]:


df.info()


# In[ ]:


df['imgs'] = list(map(unravel_img,df.values))
df = df['imgs']


# In[ ]:


df.head()


# In[ ]:


plt.imshow(df.iloc[-1])
plt.show()


# ### On disk processing

# In[ ]:


#Toddo


# #### Quering the data in hdf file

# In[ ]:


##todo

