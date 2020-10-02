#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from skimage.util import montage
import pandas as pd
from torch import optim
import re

from utils import *


# In[ ]:


get_ipython().system('git clone https://github.com/radekosmulski/whale /kaggle/working/whale')


# In[ ]:


import sys
 # Add directory holding utility functions to path to allow importing utility funcitons
#sys.path.insert(0, '/kaggle/working/protein-atlas-fastai')
sys.path.append('/kaggle/working/whale')


# In[ ]:


from whale.utils import map5


# I take a curriculum approach to training here. I first expose the model to as many different images of whales as quickly as possible (no oversampling) and train on images resized to 224x224.
# 
# I would like the conv layers to start picking up on features useful for identifying whales. For that, I want to show the model as rich of a dataset as possible.
# 
# I then train on images resized to 448x448.
# 
# Finally, I train on oversampled data. Here, the model will see some images more often than others but I am hoping that this will help alleviate the class imbalance in the training data.

# In[ ]:


import fastai
from fastprogress import force_console_behavior
import fastprogress
fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


ls ../input


# In[ ]:


path = Path('../input/humpback-whale-identification/')
path_test = Path('../input/humpback-whale-identification/test')
path_train = Path('../input/humpback-whale-identification/train')


# In[ ]:


df = pd.read_csv(path/'train.csv')#.sample(frac=0.05)
df.head()
val_fns = {'69823499d.jpg'}


# In[ ]:


fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}
path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)


# In[ ]:


name = f'res50-full-train'


# In[ ]:


SZ = 224
BS = 64
NUM_WORKERS = 0
SEED=0


# In[ ]:


data = (
    ImageItemList
        .from_df(df[df.Id != 'new_whale'], '../input/humpback-whale-identification/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('../input/humpback-whale-identification/test'))
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='../input/humpback-whale-identification')
        .normalize(imagenet_stats)
)


# In[ ]:


MODEL_PATH = "/tmp/model/"


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlearn = create_cnn(data, models.resnet50, lin_ftrs=[2048], model_dir=MODEL_PATH)\nlearn.clip_grad();')


# In[ ]:


learn.fit_one_cycle(1, 1e-2)
learn.save(f'{name}-stage-1')


# In[ ]:


learn.unfreeze()

max_lr = 1e-3
lrs = [max_lr/100, max_lr/10, max_lr]

learn.fit_one_cycle(1, lrs)
learn.save(f'{name}-stage-2')


# In[ ]:


SZ = 224 * 2
BS = 64 // 4
NUM_WORKERS = 0
SEED=0


# In[ ]:


data = (
    ImageItemList
        .from_df(df[df.Id != 'new_whale'], '../input/humpback-whale-identification/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('../input/humpback-whale-identification/test'))
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='../input/humpback-whale-identification')
        .normalize(imagenet_stats)
)


# In[ ]:


get_ipython().run_cell_magic('time', '', "learn = create_cnn(data, models.resnet50, lin_ftrs=[2048], model_dir=MODEL_PATH)\nlearn.clip_grad();\nlearn.load(f'{name}-stage-2')\nlearn.freeze_to(-1)\n\nlearn.fit_one_cycle(1, 1e-2 / 4)\nlearn.save(f'{name}-stage-3')\n\nlearn.unfreeze()\n\nmax_lr = 1e-3 / 4\nlrs = [max_lr/100, max_lr/10, max_lr]\n\nlearn.fit_one_cycle(1, lrs)\nlearn.save(f'{name}-stage-4')")


# In[ ]:


# with oversampling
df = pd.read_csv('../input/radek-whale-oversample/oversampled_train_and_val.csv')


# In[ ]:


data = (
    ImageItemList
        .from_df(df, '../input/humpback-whale-identification/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('data/test'))
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        .normalize(imagenet_stats)
)


# In[ ]:


get_ipython().run_cell_magic('time', '', "learn = create_cnn(data, models.resnet50, lin_ftrs=[2048], model_dir=MODEL_PATH)\nlearn.clip_grad();\nlearn.load(f'{name}-stage-4')\nlearn.freeze_to(-1)\n\nlearn.fit_one_cycle(1, 1e-2 / 4)\nlearn.save(f'{name}-stage-5')\n\nlearn.unfreeze()\n\nmax_lr = 1e-3 / 4\nlrs = [max_lr/100, max_lr/10, max_lr]\n\nlearn.fit_one_cycle(1, lrs)\nlearn.save(f'{name}-stage-6')")


# ## Predict

# In[ ]:


preds, _ = learn.get_preds(DatasetType.Test)


# In[ ]:


preds = torch.cat((preds, torch.ones_like(preds[:, :1])), 1)


# In[ ]:


preds[:, 5004] = 0.06


# In[ ]:


classes = learn.data.classes + ['new_whale']


# In[ ]:


from whale.utils import *


# In[ ]:


def create_submission(preds, data, name, classes=None):
    if not classes: classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_5_pred_labels(preds, classes)
    sub.to_csv(f'{name}.csv', index=False)


# In[ ]:


create_submission(preds, learn.data, name, classes)


# In[ ]:


pd.read_csv(f'{name}.csv').head()


# In[ ]:


pd.read_csv(f'{name}.csv').Id.str.split().apply(lambda x: x[0] == 'new_whale').mean()


# In[ ]:


#!kaggle competitions submit -c humpback-whale-identification -f subs/{name}.csv.gz -m "{name}"


# In[ ]:


get_ipython().system('rm -rf /kaggle/working/whale')


# In[ ]:





# In[ ]:




