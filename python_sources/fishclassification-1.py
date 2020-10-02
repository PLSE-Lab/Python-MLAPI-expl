#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os,sys
print(os.listdir("/kaggle/"))

# Any results you write to th5,/45,,,,,,e current directory are saved as output.


# In[ ]:


os.listdir('/kaggle/working')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from fastai import *
from fastai.vision import *
from fastai.widgets import *
import shutil


# Copy the images to the new path, as we can't make changes in input directory

# In[ ]:


src = "../input/fishdataset/FishDataset/"
path = "/kaggle/fishset/FishDataset/"
shutil.copytree(src, path)


# In[ ]:


os.listdir(path)


# Verify the images and delete the corrupted images 

# In[ ]:


classes = ['salmon',
 'SardineFish',
 'TilapiaFish',
 'bassfish',
 'swordfish',
 'catfish',
 'ArcticCharFish',
 'haddockfish',
 'tuna',
 'Red-snapper']
pth = Path(path)
for c in classes:
    print(c)
    verify_images(pth/c, delete=True, max_size=500)


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(pth, train=".",test=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=128, num_workers=4).normalize(imagenet_stats)
data


# In[ ]:


# testData = "/kaggle/working/test/"


# In[ ]:


# data.test_ds[0]


# In[ ]:


# import PIL


# In[ ]:


# index = 0
# for img in data.test_ds[:10]:
#     x = image2np(img[0].data*255).astype(np.uint8)
#     PIL.Image.fromarray(x).convert("RGB").save(f"{index}.png")
#     print(f"{index}.png")
#     index += 1 


# In[ ]:


# os.listdir("/kaggle/working")


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 2e-04
learn.fit_one_cycle(8,slice(lr))


# In[ ]:


learn.save('/kaggle/working/stage-1')


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8, max_lr=slice(2e-6,2e-4))


# In[ ]:


learn.save('/kaggle/working/stage-2')


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(pth, train=".",test=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=256, num_workers=4).normalize(imagenet_stats)
data


# In[ ]:



learn.data = data


# In[ ]:


learn.freeze()
learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:



learn.fit_one_cycle(7, slice(1e-05,2e-03))


# In[ ]:


learn.save('/kaggle/working/stage2')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(7, slice(1e-05,2e-03))


# In[ ]:


learn.save('/kaggle/working/stage-2-93')


# In[ ]:



db = (ImageList.from_folder(pth)
                   .split_none()
                   .label_from_folder()
                   .transform(get_transforms(), size=224)
                   .databunch()
     )


# In[ ]:


learn_cln = cnn_learner(db, models.resnet50, metrics=error_rate)
learn_cln.load('/kaggle/working/stage-2-93')


# In[ ]:


ds, idxs = DatasetFormatter().from_toplosses(learn_cln, n_imgs = 200)


# In[ ]:


ImageCleaner(ds, idxs, pth)


# In[ ]:


df = pd.read_csv(pth/'cleaned.csv', header='infer')
df.head()


# In[ ]:


np.random.seed(42)
db = (ImageList.from_df(df, pth)
                   .split_by_rand_pct(0.2)
                   .label_from_df()
                   .transform(size=224)
                   .databunch(bs=64)
                   .normalize(imagenet_stats))


# In[ ]:


learn = cnn_learner(db, models.resnet50, metrics=error_rate)
learn = learn.load('/kaggle/working/stage2-93')


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(6, slice(1e-4,3e-3))


# In[ ]:


learn.save('/kaggle/working/stage-2cleaned99')


# In[ ]:


os.listdir('/kaggle/working/')


# In[ ]:



learn.export()


# In[ ]:


img = open_image(pth/'catfish'/'images26.jpg')
img


# In[ ]:


learn = load_learner(pth)


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# In[ ]:


#ds, idxs = DatasetFormatter().from_similars(learn_cln, n_imgs = 200)


# In[ ]:


#ImageCleaner(ds, idxs,"../fishset/", duplicates=True)


# In[ ]:


#learn_cln.fit_one_cycle(6, max_lr=slice(1e-6,5e-4))


# In[ ]:


# datapath = "../fishset/Fishdata/" 
# np.random.seed(42)
# data = ImageDataBunch.from_folder(datapath, train=".", valid_pct=0.2,
#         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

