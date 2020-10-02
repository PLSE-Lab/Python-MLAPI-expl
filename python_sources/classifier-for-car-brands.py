#!/usr/bin/env python
# coding: utf-8

# # Classify cars based on their brand name

# In[ ]:


# Use Fastai and torch version as per deployment app
# !pip install --upgrade fastai==1.0.52 torch==1.1.0


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


from fastai import *
from fastai.vision import *


# ### Create classification classes

# In[ ]:


classes = ['audi', 'cadillac', 'ferrari', 'jaguar', 'lamborghini', 'maruti', 'mercedes', 'mustang', 'porsche', 'tata']


# ### Download images listed in text file urls to path dest, at most max_pics

# In[ ]:


folder = 'cadillac'
file = 'urls_cadillac.txt'


# In[ ]:


path = Path('data/cars')
dest = path/folder
dest.mkdir(parents=True)


# In[ ]:


get_ipython().system('cp ../input/car-brands/*.* {path}/')


# In[ ]:


download_images(path/file, dest, max_pics=300)


# In[ ]:


folder = 'ferrari'
file = 'urls_ferrari.txt'


# In[ ]:


path = Path('data/cars')

dest = path/folder
dest.mkdir(parents=True)
download_images(path/file, dest, max_pics=300)


# In[ ]:


folder = 'jaguar'
file = 'urls_jaguar.txt'


# In[ ]:


path = Path('data/cars')

dest = path/folder
dest.mkdir(parents=True)
download_images(path/file, dest, max_pics=300)


# In[ ]:


folder = 'lamborghini'
file = 'urls_lamborghini.txt'


# In[ ]:


path = Path('data/cars')

dest = path/folder
dest.mkdir(parents=True)
download_images(path/file, dest, max_pics=300)


# In[ ]:


folder = 'maruti'
file = 'urls_maruti.txt'


# In[ ]:


path = Path('data/cars')

dest = path/folder
dest.mkdir(parents=True)
download_images(path/file, dest, max_pics=300)


# In[ ]:


folder = 'mercedes'
file = 'urls_mercedes.txt'


# In[ ]:


path = Path('data/cars')

dest = path/folder
dest.mkdir(parents=True)
download_images(path/file, dest, max_pics=300)


# In[ ]:


folder = 'mustang'
file = 'urls_mustang.txt'


# In[ ]:


path = Path('data/cars')

dest = path/folder
dest.mkdir(parents=True)
download_images(path/file, dest, max_pics=300)


# In[ ]:


folder = 'porsche'
file = 'urls_porsche.txt'


# In[ ]:


path = Path('data/cars')

dest = path/folder
dest.mkdir(parents=True)
download_images(path/file, dest, max_pics=300)


# In[ ]:


folder = 'tata'
file = 'urls_tata.txt'


# In[ ]:


path = Path('data/cars')

dest = path/folder
dest.mkdir(parents=True)
download_images(path/file, dest, max_pics=300)


# In[ ]:


folder = 'audi'
file = 'urls_audi.txt'


# In[ ]:


path = Path('data/cars')

dest = path/folder
dest.mkdir(parents=True)
download_images(path/file, dest, max_pics=300)


# ### Verify the downloaded images can be opened if not then delete it 

# In[ ]:


for c in classes:
  print(c)
  verify_images(path/c, delete=True, max_workers=8)


# In[ ]:


Path(path/'mustang').ls()


# In[ ]:


def get_ex(): return open_image(path/'mustang/00000132.jpg')

def plots_f(rows, cols, width, height, **kwargs):
    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(
        rows,cols,figsize=(width,height))[1].flatten())]


# ### Use data augmentation to create images using transformation methods

# In[ ]:


tfms=get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.3, max_warp=0.4, p_affine=1., p_lighting=1.)


# In[ ]:


plots_f(3,3,8,8)


# ### Create databunch from verified images
# ### First time we will use small size image to train the model

# In[ ]:


np.random.seed(2)
src = (ImageList.from_folder(path)
       .split_by_rand_pct(0.2)); src


# In[ ]:


def get_databunch(size, bs, padding_mode='reflection'):
    return (src.label_from_folder()
           .transform(tfms, size=size, padding_mode=padding_mode)
           .databunch(bs=bs).normalize(imagenet_stats))


# In[ ]:


data = get_databunch(128, 32);data


# ### Tries resnet34 but resnet50 model gives better accuracy

# In[ ]:


learn=cnn_learner(data, models.resnet50, metrics=accuracy)
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(50, slice(1e-3, 1e-2), pct_start=0.8)


# In[ ]:


learn.model_dir = "/kaggle/working"
learn.save('car_brand_stage-1-128')


# In[ ]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-5, 1e-4))


# In[ ]:


learn.save('car_brand_stage-2-128')


# ### Now that model is almost 70% accurate with size = 128 so try to train it with size = 256 to get more accuracy

# In[ ]:


data= (src
       .transform(get_transforms(), size=256)
       .databunch(bs=32).normalize(imagenet_stats))
learn.data=data
data.train_ds[0][0].shape


# ## Train last few layers

# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10, slice(2e-3, 1e-2))


# In[ ]:


learn.save('car_brand_stage-1-256')


# ## Interpretation

# In[ ]:


learn.load('car_brand_stage-1-256')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# ## Cleaning up

# In[ ]:


# from fastai.widgets import *


# In[ ]:


# ds, idx = DatasetFormatter().from_toplosses(learn, n_imgs=100)


# In[ ]:


# ImageCleaner(ds, idx, path)


# In[ ]:


# ds, idxs = DatasetFormatter().from_similars(learn, n_imgs=100)
# ImageCleaner(ds, idx, path, duplicates=True)


# In[ ]:


# df = pd.read_csv(path/'cleaned.csv', header='infer')
# print(df.head)


# In[ ]:


# db = ImageDataBunch.from_df(path, df, valid_pct=0.2,
#         ds_tfms=get_transforms(), size=128, num_workers=0).normalize(imagenet_stats)


# In[ ]:


# print(data.classes, data.c, len(data.train_ds), len(data.valid_ds))
# print(db.classes, db.c, len(db.train_ds), len(db.valid_ds))


# In[ ]:


# learn.load('car_brand_stage-2')


# In[ ]:


# learn.data=db
# learn.freeze()


# In[ ]:


# learn.lr_find()


# In[ ]:


# learn.recorder.plot()


# In[ ]:


# learn.fit_one_cycle(5)


# In[ ]:


# learn.unfreeze()
# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


# learn.fit_one_cycle(2, max_lr=slice(3e-5, 1e-4))


# In[ ]:


# learn.save('car_brand_stage-3')


# In[ ]:


# learn.unfreeze()


# In[ ]:


# learn.fit_one_cycle(5)


# In[ ]:


# learn.save('car_brand_stage-4')


# ## Get the model for production

# In[ ]:


# import fastai
# # defaults.device = torch.device('cpu')
# test_data = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=256).normalize(imagenet_stats)
# learn = cnn_learner(test_data, models.resnet34).load('car_brand_stage-1-256')
# learn.export()


# In[ ]:


# # https://medium.freecodecamp.org/how-to-transfer-large-files-to-google-colab-and-remote-jupyter-notebooks-26ca252892fa
# import os
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# # from google.colab import auth
# from oauth2client.client import GoogleCredentials


# In[ ]:


# !cp ../input/oauth-20-client-id/* .


# In[ ]:


# gauth = GoogleAuth()
# gauth.CommandLineAuth()
# drive = GoogleDrive(gauth)


# In[ ]:


# path; path.ls()


# In[ ]:


# upload = drive.CreateFile({'title': 'export.pkl'})
# upload.SetContentFile('data/cars/export.pkl')
# upload.Upload()


# In[ ]:


# learn.load('car_brand_stage-1-256')


# In[ ]:


# img = open_image(path/'audi'/'00000011.jpg')
# img


# In[ ]:


# pred_class,pred_idx,outputs = learn.predict(img)
# pred_class

