#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.notebook import tqdm


# In[ ]:


data_path = Path('../input/prostate-cancer-grade-assessment/')
os.listdir(data_path)


# In[ ]:


mask_path = Path('../input/panda-train-mask/')
os.listdir(mask_path)


# In[ ]:


get_ipython().system('cd ../input/prostate-cancer-grade-assessment/; du -h')


# In[ ]:


os.listdir(data_path/'train_label_masks')


# In[ ]:


import pandas as pd
train_df = pd.read_csv(mask_path/'train_mask.csv')
train_df.head(10)


# In[ ]:


print('Number of whole-slide images in training set: ', len(train_df))


# In[ ]:


sample_image = train_df.iloc[np.random.choice(len(train_df))].image_id
print(sample_image)


# In[ ]:


import openslide


# In[ ]:


openslide_image = openslide.OpenSlide(str(data_path/'train_label_masks'/(sample_image+'.tiff')))


# In[ ]:


openslide_image.properties


# In[ ]:


img = openslide_image.read_region(location=(0,0),level=2,size=(openslide_image.level_dimensions[2][0],openslide_image.level_dimensions[2][1]))
img


# In[ ]:


Image.fromarray(np.array(img.resize((512,512)))[:,:,:3])


# In[ ]:


get_ipython().run_line_magic('pinfo', 'Image.save')


# In[ ]:


for i in tqdm(train_df['image_id'],total=len(train_df)):
    openslide_image
    img
    Image.fromarray(np.array(img.resize((256,256)))[:,:,:3]).save(i+'.jpeg')
    

