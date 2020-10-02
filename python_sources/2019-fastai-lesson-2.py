#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *
from fastai.basic_data import *


# # Download pictures

# I'm going to train a classifier to distinguish photos of the four UK party leaders (Boris Johnson, Jeremy Corbyn, Jo Swinson and Nicola Sturgon).

# In[ ]:


folder='BoJo'; file='../input/BoJo.csv'

path = Path('../pics')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)

download_images('../input/BoJo.csv', dest, max_pics=120, max_workers=0)
verify_images(path, delete=True, max_size=120)


# In[ ]:


folder='JeremyC'; file='../input/JeremyC.csv'

path = Path('../pics')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)

download_images('../input/JeremyC.csv', dest, max_pics=120, max_workers=0)

verify_images(path, delete=True, max_size=120)


# In[ ]:


folder='JoSwinson'; file='../input/JoSwinson.csv'

path = Path('../pics')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)

download_images('../input/JoSwinson.csv', dest, max_pics=120, max_workers=0)

verify_images(path, delete=True, max_size=120)


# In[ ]:


folder='NicolaSturgeon'; file='../input/NicolaSturgeon.csv'

path = Path('NicolaSturgeon_pics')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)

download_images('../input/NicolaSturgeon.csv', dest, max_pics=120, max_workers=0)

verify_images(path, delete=True, max_size=500)


# In[ ]:


classes = ['BoJo', 'JeremyC', 'JoSwinson', 'NicolaSturgeon']


# In[ ]:


cnt=0
dirname = '../pics'
for filename in dirname:
    try:
        img=Image.open(dirname+"/"+filename)
    except OSError:
        print("FILE: ", filename, "is corrupt!")
        cnt+=1
        print(dirname+"/"+filename)
        #os.remove(dirname+"/"+filename)
print("Successfully Completed Operation! Files Courrupted are ", cnt)


# # View data

# In[ ]:


import os
for dirname, _, filenames in os.walk('../pics'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


np.random.seed(42)

src = ImageDataBunch.from_folder('../pics',train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(7,8))


# In[ ]:





# In[ ]:




