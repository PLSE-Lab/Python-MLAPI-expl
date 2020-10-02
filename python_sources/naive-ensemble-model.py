#!/usr/bin/env python
# coding: utf-8

# # Reference
# 
# 1. https://www.kaggle.com/daisukelab/cnn-2d-basic-solution-powered-by-fast-ai.
# 2. https://www.kaggle.com/daisukelab/fat2019_prep_mels1
# 3. https://www.kaggle.com/c/freesound-audio-tagging-2019/overview/timeline
# 4. https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb
# 
# # New
# I tried training with more models and 2 models: vgg_16 , vgg_19 with the best results.
# Also, I changed the bz from 64 to 128 and also gave better results.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import IPython
import IPython.display
import PIL
import pickle
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## File/folder definitions
# 
# - `df` will handle training data.
# - `test_df` will handle test data.

# In[ ]:


DATA = Path('../input/freesound-audio-tagging-2019')
PREPROCESSED = Path('../input/fat2019_prep_mels1')
WORK = Path('work')
Path(WORK).mkdir(exist_ok=True, parents=True)


# In[ ]:


CSV_TRN_CURATED = DATA/'train_curated.csv'
CSV_TRN_NOISY = PREPROCESSED/'trn_noisy_best50s.csv'
CSV_SUBMISSION = DATA/'sample_submission.csv'

MELS_TRN_CURATED = PREPROCESSED/'mels_train_curated.pkl'
MELS_TRN_NOISY = '../input/fat2019_prep_mels1/mels_trn_noisy_best50s.pkl'
MELS_TEST = PREPROCESSED/'mels_test.pkl'

trn_curated_df = pd.read_csv(CSV_TRN_CURATED)
trn_noisy_df = pd.read_csv(CSV_TRN_NOISY)
test_df = pd.read_csv(CSV_SUBMISSION)

df = pd.concat([trn_curated_df, trn_noisy_df], ignore_index=True) # not enough memory
# df = pd.concat([trn_curated_df], ignore_index=True)
test_df = pd.read_csv(CSV_SUBMISSION)

X_train = pickle.load(open(MELS_TRN_CURATED, 'rb'))  # Create an empty dictionary
with open("../input/fat2019_prep_mels1/mels_trn_noisy_best50s.pkl", 'rb') as f:
    X_train.append(pickle.load(f))   # Update contents of file2 to the dictionary

X_test = pickle.load(open(MELS_TEST, 'rb'))


# ## Custom `open_image` for fast.ai library to load data from memory
# 
# - Important note: Random cropping 1 sec, this is working like augmentation.

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.vision.data import *
import random

CUR_X_FILES, CUR_X = list(df.fname.values), X_train

def open_fat2019_image(fn, convert_mode, after_open)->Image:
    # open
    idx = CUR_X_FILES.index(fn.split('/')[-1])
    x = PIL.Image.fromarray(CUR_X[idx])
    # crop
    time_dim, base_dim = x.size
    crop_x = random.randint(0, time_dim - base_dim)
    x = x.crop([crop_x, 0, crop_x+base_dim, base_dim])    
    # standardize
    return Image(pil2tensor(x, np.float32).div_(255))

vision.data.open_image = open_fat2019_image


# ## Follow multi-label classification
# 
# - Almost following fast.ai course: https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb
# - But `pretrained=False`

# In[ ]:


tfms = get_transforms(do_flip=True, max_rotate=0, max_lighting=0.1, max_zoom=0, max_warp=0.)
src = (ImageList.from_csv(WORK, Path('..')/CSV_TRN_CURATED, folder='trn_curated')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=',')
)
data = (src.transform(tfms, size=128)
        .databunch(bs=128).normalize(imagenet_stats)
)


# In[ ]:


list_model = [ models.vgg16_bn, models.vgg19_bn]
predicts = None
i = 0


# In[ ]:


len(list_model)


# In[ ]:


print(list_model[i])
CUR_X_FILES, CUR_X = list(df.fname.values), X_train
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, list_model[i], pretrained=False, metrics=[f_score])
learn.unfreeze()
learn.lr_find()

learn.fit_one_cycle(5, slice(1e-6, 1e-1))

learn.lr_find()

learn.fit_one_cycle(100, slice(1e-6, 1e-2))

learn.export()

CUR_X_FILES, CUR_X = list(test_df.fname.values), X_test

test = ImageList.from_csv(WORK, Path('..')/CSV_SUBMISSION, folder='test')
learn = load_learner(WORK, test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
if predicts is None:
    predicts = preds
else:
    predicts += preds
i+=1


# In[ ]:


print(list_model[i])
CUR_X_FILES, CUR_X = list(df.fname.values), X_train
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, list_model[i], pretrained=False, metrics=[f_score])
learn.unfreeze()
learn.lr_find()

learn.fit_one_cycle(5, slice(1e-6, 1e-1))

learn.lr_find()

learn.fit_one_cycle(100, slice(1e-6, 1e-2))

learn.export()

CUR_X_FILES, CUR_X = list(test_df.fname.values), X_test

test = ImageList.from_csv(WORK, Path('..')/CSV_SUBMISSION, folder='test')
learn = load_learner(WORK, test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
if predicts is None:
    predicts = preds
else:
    predicts += preds
i+=1


# In[ ]:


print(i)


# In[ ]:


predicts = predicts/i


# In[ ]:


test_df[learn.data.classes] = predicts
test_df.to_csv('submission.csv', index=False)
test_df.head()

