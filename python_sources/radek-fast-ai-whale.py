#!/usr/bin/env python
# coding: utf-8

# https://github.com/radekosmulski/whale

# In[ ]:


from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd

from fastai import *
from fastai.vision import *


# ## A look at the data

# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


path = Path('../input/')
path_test = Path('../input/test')
path_train = Path('../input/train')


# In[ ]:


df = pd.read_csv(path/'train.csv')#.sample(frac=0.05)
df.head()


# In[ ]:


df.Id.value_counts().head()


# In[ ]:


(df.Id == 'new_whale').mean()


# In[ ]:


(df.Id.value_counts() == 1).mean()


# 41% of all whales have only a single image associated with them.
# 
# 38% of all images contain a new whale - a whale that has not been identified as one of the known whales.
# 
# There is a superb writeup on what a solution to this problem might look like [here](https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563/notebook). In general, the conversation in the Kaggle [forum](https://www.kaggle.com/c/humpback-whale-identification/discussion) also seems to have some very informative threads.
# 
# Either way, starting with a simple model that can be hacked together in a couple of lines of code is a recommended approach. It is good to have a baseline to build on - going for a complex model from start is a way for dying a thousand deaths by subtle bugs.

# In[ ]:


df.Id.nunique()


# In[ ]:


df.shape


# In[ ]:


fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}


# In[ ]:


SZ = 224
BS = 64
NUM_WORKERS = 0
SEED=0


# In[ ]:


#tfms = get_transforms(do_flip=False)


# In[ ]:


#data = ImageDataBunch.from_df(path_train, df, ds_tfms=tfms, size=150,num_workers=0)


# In[ ]:


data = (
    ImageItemList
        .from_folder('../input/train')
        .random_split_by_pct(seed=SEED)
        .label_from_func(lambda path: fn2label[path.name])
        .add_test(ImageItemList.from_folder('../input/test'))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='../input/train')
)


# In[ ]:


data.show_batch(rows=3)


# ## Train

# In[ ]:


name = f'res50-{SZ}'


# In[ ]:


get_ipython().system('git clone https://github.com/radekosmulski/whale')


# In[ ]:


import sys
 # Add directory holding utility functions to path to allow importing utility funcitons
#sys.path.insert(0, '/kaggle/working/protein-atlas-fastai')
sys.path.append('/kaggle/working/whale')


# In[ ]:


from whale.utils import map5


# In[ ]:


MODEL_PATH = "/tmp/model/"


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=[accuracy, map5], model_dir=MODEL_PATH)


# In[ ]:


learn.fit_one_cycle(2)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save(f'{name}-stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


max_lr = 1e-4
lrs = [max_lr/100, max_lr/10, max_lr]


# In[ ]:


learn.fit_one_cycle(5, lrs)


# In[ ]:


learn.save(f'{name}-stage-2')


# In[ ]:


learn.recorder.plot_losses()


# This is not a loss plot you would normally expect to see. Why does it look like this? Let's consider what images appear in the validation set:
#  * images of whales that do not appear in the train set (whales where all their images were randomly assigned to the validation set) - there is nothing our model can learn about these!
#  * images of whales with multiple images in the dataset where some subset of those got assigned to the validation set
#  * `new_whale` images
#  
# Intuitively, a model such as the above does not seem to frame the problem in a way that would be easy for a neural network to solve. Nonetheless, it is interesting to think how we could improve on the construction of the validation set? What tweaks could be made to the model to improve its performance?

# ## Predict

# In[ ]:


preds, _ = learn.get_preds(DatasetType.Test)


# In[ ]:


from whale.utils import *


# In[ ]:


def create_submission(preds, data, name, classes=None):
    if not classes: classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_5_pred_labels(preds, classes)
    sub.to_csv(f'{name}.csv', index=False) # compression='gzip'


# In[ ]:


create_submission(preds, learn.data, name)


# In[ ]:


pd.read_csv(f'{name}.csv').head()


# In[ ]:


get_ipython().system('rm -rf /kaggle/working/whale')


# In[ ]:


#!kaggle competitions submit -c humpback-whale-identification -f {name}.csv.gz -m "{name}"


# In[ ]:


#!kaggle competitions submit -c humpback-whale-identification -f {name}.csv -m "{name}"


# In[ ]:




