#!/usr/bin/env python
# coding: utf-8

# # CNN 2D Basic Solution #2.5
# 
# This is 2nd version of https://www.kaggle.com/daisukelab/cnn-2d-basic-solution-powered-by-fast-ai.
# 
# - Based on fast.ai library, solving issue as image multi-label classification.
# - Using mel-spectrogram preprocessing.
# - Using preprocessed dataset additionally. https://www.kaggle.com/daisukelab/fat2019_prep_mels1
# 
# ## _What's new in 14-Apr version_
# 
# - _Smaller subset of noisy training data was added; The best 50s. (I should call random 50 actually...)_
# - _Training uses both curated and this subset of noisy data._
# 
# # Important notice for final submission
# 
# This kernel is not preprocessing __test set__, then this doesn't work for final submission.
# We should be ready for preprocessing new test set used in re-evaluation after deadline:
# 
# _"Note, as this competition is a Kernels-only, two-stage competition, following the final submission deadline for the competition, your kernel code will be re-run on a privately-held test set that is not provided to you. It is your model's score against this private test set that will determine your ranking on the private leaderboard and final standing in the competition. The leaderboard will be updated in the days following the competition's completion, and our team will announce that the re-run has been completed and leaderboard finalized with an announcement made on the competition forums."_
# 
# Link: https://www.kaggle.com/c/freesound-audio-tagging-2019/overview/timeline

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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system(' ls ../input/fat2019_prep_mels1')


# ## File/folder definitions
# 
# - `df` will handle training data.
# - `test_df` will handle test data.

# In[ ]:


DATA = Path('../input/freesound-audio-tagging-2019')
PREPROCESSED = Path('../input/fat2019_prep_mels1')
WORK = Path('work')
Path(WORK).mkdir(exist_ok=True, parents=True)

CSV_TRN_CURATED = DATA/'train_curated.csv'
CSV_TRN_NOISY = DATA/'train_noisy.csv'
CSV_TRN_NOISY_BEST50S = PREPROCESSED/'trn_noisy_best50s.csv'
CSV_SUBMISSION = DATA/'sample_submission.csv'

MELS_TRN_CURATED = PREPROCESSED/'mels_train_curated.pkl'
MELS_TRN_NOISY = PREPROCESSED/'mels_train_noisy.pkl'
MELS_TRN_NOISY_BEST50S = PREPROCESSED/'mels_trn_noisy_best50s.pkl'
MELS_TEST = PREPROCESSED/'mels_test.pkl'

trn_curated_df = pd.read_csv(CSV_TRN_CURATED)
trn_noisy_df = pd.read_csv(CSV_TRN_NOISY)
trn_noisy50s_df = pd.read_csv(CSV_TRN_NOISY_BEST50S)
test_df = pd.read_csv(CSV_SUBMISSION)

#df = pd.concat([trn_curated_df, trn_noisy_df], ignore_index=True) # not enough memory
df = pd.concat([trn_curated_df, trn_noisy50s_df], ignore_index=True, sort=True)
test_df = pd.read_csv(CSV_SUBMISSION)

X_train = pickle.load(open(MELS_TRN_CURATED, 'rb')) + pickle.load(open(MELS_TRN_NOISY_BEST50S, 'rb'))


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
        .databunch(bs=64).normalize(imagenet_stats)
)


# In[ ]:


data.show_batch(3)


# In[ ]:


f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[f_score])
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-6, 1e-1))


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(100, slice(1e-6, 1e-2))


# In[ ]:


learn.save('fat2019_fastai_cnn2d_stage-2')
learn.export()


# ## Test prediction and making submission file simple
# - Switch to test data.
# - Overwrite results to sample submission; simple way to prepare submission file.

# In[ ]:


del X_train
X_test = pickle.load(open(MELS_TEST, 'rb'))
CUR_X_FILES, CUR_X = list(test_df.fname.values), X_test

test = ImageList.from_csv(WORK, Path('..')/CSV_SUBMISSION, folder='test')
learn = load_learner(WORK, test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test_df[learn.data.classes] = preds
test_df.to_csv('submission.csv', index=False)
test_df.head()


# ## Visualizing activations

# In[ ]:


del X_test
X_train = pickle.load(open(MELS_TRN_CURATED, 'rb'))

CUR_X_FILES, CUR_X = list(df.fname.values), X_train
learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[f_score])
learn.load('fat2019_fastai_cnn2d_stage-2');


# In[ ]:


# Thanks to https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb
from fastai.callbacks.hooks import *

def visualize_cnn_by_cam(learn, data_index):
    x, _y = learn.data.valid_ds[data_index]
    y = _y.data
    if not isinstance(y, (list, np.ndarray)): # single label -> one hot encoding
        y = np.eye(learn.data.valid_ds.c)[y]

    m = learn.model.eval()
    xb,_ = learn.data.one_item(x)
    xb_im = Image(learn.data.denorm(xb)[0])
    xb = xb.cuda()

    def hooked_backward(cat):
        with hook_output(m[0]) as hook_a: 
            with hook_output(m[0], grad=True) as hook_g:
                preds = m(xb)
                preds[0,int(cat)].backward()
        return hook_a,hook_g
    def show_heatmap(img, hm, label):
        _,axs = plt.subplots(1, 2)
        axs[0].set_title(label)
        img.show(axs[0])
        axs[1].set_title(f'CAM of {label}')
        img.show(axs[1])
        axs[1].imshow(hm, alpha=0.6, extent=(0,img.shape[0],img.shape[0],0),
                      interpolation='bilinear', cmap='magma');
        plt.show()

    for y_i in np.where(y > 0)[0]:
        hook_a,hook_g = hooked_backward(cat=y_i)
        acts = hook_a.stored[0].cpu()
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        mult = (acts*grad_chan[...,None,None]).mean(0)
        show_heatmap(img=xb_im, hm=mult, label=str(learn.data.valid_ds.y[data_index]))

for idx in range(10):
    visualize_cnn_by_cam(learn, idx)


# In[ ]:


# https://discuss.pytorch.org/t/how-to-visualize-the-actual-convolution-filters-in-cnn/13850
from sklearn.preprocessing import minmax_scale

def visualize_first_layer(learn):
    conv1 = list(learn.model.children())[0][0]
    weights = conv1.weight.data.cpu().numpy()
    weights_shape = weights.shape
    weights = minmax_scale(weights.ravel()).reshape(weights_shape)
    fig, axes = plt.subplots(8, 8, figsize=(8,8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.rollaxis(weights[i], 0, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

visualize_first_layer(learn)


# Fiters are not matured yet...

# In[ ]:




