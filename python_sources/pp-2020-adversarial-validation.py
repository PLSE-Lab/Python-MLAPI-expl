#!/usr/bin/env python
# coding: utf-8

# # Plant Pathology 2020
# 
# ## Adversial Validation
# 
# Since it appears quite common to find leaderboard scores that are significantly lower than validation, this notebook explores the test set, trying to understand what makes it different from the training set.
# 
# I use fastai2 for the model training again, forking the notebook from my [initial fastai2 experiment](https://www.kaggle.com/lextoumbourou/plant-pathology-2020-eda-training-fastai2).

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastcore > /dev/null')
get_ipython().system('pip install git+https://github.com/fastai/fastai2 > /dev/null')
get_ipython().system('pip install iterative-stratification > /dev/null')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import sys
from collections import Counter
from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm.notebook import tqdm
from torchvision.models import densenet121
from sklearn.metrics import roc_auc_score
from torch.utils.data.sampler import WeightedRandomSampler

from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ## Params

# In[ ]:


DATA_PATH = Path('../input/plant-pathology-2020-fgvc7')
IMG_PATH = DATA_PATH / 'images'
LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']

IMG_SIZE = 448
SEED = 420
N_FOLDS = 5
BS = 16
N_FOLDS = 5

ARCH = densenet121


# In[ ]:


seed_everything(SEED)


# In[ ]:


train_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')


# ## Explore the test set

# Firstly let's understand how the test set distribution compares to the train distribution. I'll use the predictions from the current [highest scoring kernel](https://www.kaggle.com/seefun/ensemble-top-kernels).

# In[ ]:


train_df['label'] = train_df[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax(1)


# In[ ]:


submission_csv = pd.read_csv('../input/ensemble-top-kernels/submission.csv')
submission_csv['label'] = submission_csv[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax(1)


# In[ ]:


_, axes = plt.subplots(1, 2, figsize=(12, 4))
axes = axes.flatten()
train_df.label.value_counts().plot.bar(ax=axes[0], title='Train')
submission_csv.label.value_counts().plot.bar(ax=axes[1], title='Test')
plt.show()


# So the major difference seems to be a higher frequency of `healthy` occurances in the train set.

# ## Adversarial Validation

# I'm going to train a model to classify whether the example came from the train or test set. If the distribution in the train and test set is exactly the same, we expect an ROC of about 0.5. Any higher than that suggests that there is something quite different about the test set.

# In[ ]:


train_df['is_test'] = False
test_df['is_test'] = True

all_df = pd.concat([
    train_df[['image_id', 'is_test']], test_df[['image_id', 'is_test']]]
).reset_index(drop=True).sample(frac=1., random_state=SEED)


# In[ ]:


all_df.is_test.value_counts()


# In[ ]:


datablock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    getters=[
        ColReader('image_id', pref=IMG_PATH, suff='.jpg'), ColReader('is_test')
    ],
    splitter=RandomSplitter(seed=SEED),
    item_tfms=Resize(IMG_SIZE),
    batch_tfms=aug_transforms(size=IMG_SIZE, max_rotate=30., min_scale=0.75, flip_vert=True, do_flip=True)
)


# In[ ]:


dls = datablock.dataloaders(source=all_df, bs=BS)


# In[ ]:


dls.show_batch()


# ## Training

# In[ ]:


def get_learner(dls, lr=1e-3):
    opt_func = partial(Adam, lr=lr, wd=0.01, eps=1e-8)

    learn = cnn_learner(
        dls, ARCH, opt_func=opt_func,
        metrics=[RocAuc()]).to_fp16()

    return learn


# In[ ]:


learn = get_learner(dls)
learn.fit_one_cycle(1)
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-4, 1e-3))


# In[ ]:


loss, metric = learn.validate()


# ## Final score

# In[ ]:


metric


# The AUC is pretty average - that's a good sign. It indicates the train and test set are pretty similar. However, it's not 0.5, so there does appear to be some distribution difference between the 2 sets.
