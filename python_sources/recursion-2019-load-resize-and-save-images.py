#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# * This kernel lets you efficiently convert all images from their tensor format into RGB images, then save them as 400x400 JPEGs inside two zip files (`train` and `test`).
# * Feel free to customize this kernel as you wish. You can change the shape and extension of the final output image by changing the input arguments to `convert_to_rgb` and `build_new_df`.
# 
# ### Notes
# 
# * In a previous version (V11) of the kernel, I claimed that the `rxrx.io.load_site_as_rgb` function was inefficient, and tried to provide a faster solution. It turns out I did not input the correct argument, so it was instead fetching the images directly from Google Storage; with the correct argument, the speed was comparable. **My sincere apologies for misleading everyone.**
# 
# 
# ### Updates
# 
# * V13: Changed output image size to 400 px instead of 224.
# 
# ### Sources
# 
# * Found out about the loading functions from this kernel: https://www.kaggle.com/jesucristo/quick-visualization-eda

# In[ ]:


import os
import sys
import zipfile

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


# # Preliminary

# We need to also import rxrx in order to convert the tensors into images.

# In[ ]:


get_ipython().system('git clone https://github.com/recursionpharma/rxrx1-utils')
sys.path.append('rxrx1-utils')
import rxrx.io as rio


# Will need those folders later for storing our jpegs.

# In[ ]:


for folder in ['train', 'test']:
    os.makedirs(folder)

get_ipython().system('ls')


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()


# In[ ]:


train_df.tail()


# # Saving as JPEG

# In[ ]:


def convert_to_rgb(df, split, resize=True, new_size=400, extension='jpeg'):
    N = df.shape[0]

    for i in tqdm(range(N)):
        code = df['id_code'][i]
        experiment = df['experiment'][i]
        plate = df['plate'][i]
        well = df['well'][i]

        for site in [1, 2]:
            save_path = f'{split}/{code}_s{site}.{extension}'

            im = rio.load_site_as_rgb(
                split, experiment, plate, well, site, 
                base_path='../input/'
            )
            im = im.astype(np.uint8)
            im = Image.fromarray(im)
            
            if resize:
                im = im.resize((new_size, new_size), resample=Image.BILINEAR)
            
            im.save(save_path)


# In[ ]:


convert_to_rgb(train_df, 'train')
convert_to_rgb(test_df, 'test')


# # Zip everything

# In[ ]:


def zip_and_remove(path):
    ziph = zipfile.ZipFile(f'{path}.zip', 'w', zipfile.ZIP_DEFLATED)
    
    for root, dirs, files in os.walk(path):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            ziph.write(file_path)
            os.remove(file_path)
    
    ziph.close()


# In[ ]:


zip_and_remove('train')
zip_and_remove('test')


# # Create new labels
# 
# Since our data is now "duplicated" (as in, we have separated the sites), we have to also duplicate our labels.

# In[ ]:


def build_new_df(df, extension='jpeg'):
    new_df = pd.concat([df, df])
    new_df['filename'] = pd.concat([
        df['id_code'].apply(lambda string: string + f'_s1.{extension}'),
        df['id_code'].apply(lambda string: string + f'_s2.{extension}')
    ])
    
    return new_df


new_train = build_new_df(train_df)
new_test = build_new_df(test_df)

new_train.to_csv('new_train.csv', index=False)
new_test.to_csv('new_test.csv', index=False)


# # Remove the rxrx1 utils
# 
# Need to remove those, otherwise we will have an error when saving.

# In[ ]:


get_ipython().system('rm -r rxrx1-utils')

