#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This file contains all the main external libs we'll use
from fastai.imports import * 
import fastai

print(fastai.__version__)


# In[ ]:


from fastai.utils import *
from fastai.vision import *
from fastai.callbacks import *
from pathlib import Path
import PIL
from torch.utils import model_zoo


# In[ ]:


PATH = "../input/emergency-vehicles-identification/Emergency_Vehicles/train"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=224
train_df = pd.read_csv('../input/emergency-vehicles-identification/Emergency_Vehicles/train.csv')
test_df = pd.read_csv('../input/emergency-vehicles-identification/Emergency_Vehicles/test.csv')


# In[ ]:


torch.cuda.is_available()


# In[ ]:


torch.backends.cudnn.enabled


# In[ ]:


os.listdir(PATH)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot('emergency_or_not', data=train_df)
plt.title('Classes', fontsize=15)
plt.show()


# In[ ]:


target_count = train_df.emergency_or_not.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')


# In[ ]:


# Class count
count_class_0, count_class_1 = train_df.emergency_or_not.value_counts()
# Divide by class
df_class_0 = train_df[train_df['emergency_or_not'] == 0]
df_class_1 = train_df[train_df['emergency_or_not'] == 1]


# In[ ]:


df_class_0.head()


# In[ ]:


train_df.info()


# In[ ]:


X = train_df[:1]
y = train_df['emergency_or_not']


# In[ ]:


PATH


# In[ ]:


fnames = np.array([f'{f}' for f in sorted(os.listdir(f'{PATH}'))])
#labels = np.array([(0 if 'cat' in fname else 1) for fname in fnames])
fnames[0]


# In[ ]:


img = plt.imread(f'{PATH}/{fnames[10]}')
plt.imshow(img);


# In[ ]:


cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# In[ ]:


get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth')


# In[ ]:


arch = "../input/resnet34/"


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=20, max_lighting=0.3, max_warp=0.2, max_zoom=1.2)


# In[ ]:


test_images = ImageList.from_df(test_df, path='../input/emergency-vehicles-identification/Emergency_Vehicles', folder='test')
src = (ImageList.from_df(train_df, path='../input/emergency-vehicles-identification/Emergency_Vehicles', folder='train')
       .split_by_rand_pct(0.2)
       .label_from_df()
       .add_test(test_images))


# In[ ]:


data = (src.transform(tfms, 
                     size=32,
                     resize_method=ResizeMethod.PAD, 
                     padding_mode='reflection')
        .databunch(bs=sz)
        .normalize(imagenet_stats))


# In[ ]:


data.classes, data.c


# In[ ]:


data.show_batch(rows=4, figsize=(9,9))


# In[ ]:


Path('models').mkdir(exist_ok=True)
get_ipython().system("cp '../input/resnet34/resnet34.pth' 'models/'")
def load_url(*args, **kwargs):
    model_dir = Path('models')
    filename  = 'resnet34.pth'
    if not (model_dir/filename).is_file(): raise FileNotFoundError
    return torch.load(model_dir/filename)
model_zoo.load_url = load_url


# In[ ]:


from google.cloud import bigquery
client = bigquery.Client()


# In[ ]:


learn = cnn_learner(data,
                    models.resnet34, 
                    metrics=[accuracy, AUROC()], 
                    path = '.')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 1e-2
learn.fit_one_cycle(10, lr)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('Model-1')


# In[ ]:


learn.recorder.plot_lr(show_moms=True)


# In[ ]:


learn = cnn_learner(data,
                    models.resnet34, 
                    metrics=[accuracy, AUROC()], 
                    callback_fns=[partial(SaveModelCallback)],
                    path = '.')
learn = learn.load('Model-1')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 1e-6
learn.fit_one_cycle(10, lr)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(2,2))


# In[ ]:


interp.plot_top_losses(4, figsize=(6,6), heatmap=False)


# In[ ]:


probability, classification = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


probability


# In[ ]:


test_df.head()


# In[ ]:


test_df['emergency_or_not'] = probability.numpy()[:, 0]


# In[ ]:


test_df.columns


# In[ ]:


test_df['emergency_or_not'] = test_df['emergency_or_not'].apply(lambda x: 1 if x > 0.75 else 0)
test_df.to_csv("submission_fastai.csv", index=False)


# In[ ]:




