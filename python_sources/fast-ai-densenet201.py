#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading...
from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
from functools import partial
from sklearn import metrics
from collections import Counter
from fastai.callbacks import *


# In[ ]:


## Making pretrained weights work without needing to find the default filename
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
        os.makedirs('/tmp/.cache/torch/checkpoints/')
get_ipython().system("cp '../input/densenet201/densenet201-4c113574.pth' '/tmp/.cache/torch/checkpoints/densenet201-c1103571.pth'")


# In[ ]:


os.listdir('../input')


# In[ ]:


print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)


# In[ ]:


# Set seed fol all
def seed_everything(seed=1358):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()


# In[ ]:


# Reading train_dataset
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')
train_dir = os.path.join(base_image_dir,'train_images/')
df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
df = df.drop(columns=['id_code'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head()


# In[ ]:


# Set Batch Size and Image size
bs = 32 
sz=224


# In[ ]:


# Data Augmentation and Transformation up to Data Bunch
tfms = get_transforms(do_flip=True,
                      flip_vert=True,
                      max_rotate=360,
                      max_warp=0.,
                      max_zoom=1.05,
                      max_lighting=0.1,
                      p_lighting=0.5
                     )
src = (ImageList.from_df(df=df
                         ,path='./'
                         ,cols='path'
                         #,convert_mode='L'
                        ) 
        .split_by_rand_pct(0.15) 
        .label_from_df(cols='diagnosis',label_cls=FloatList) 
      )
data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
        .databunch(bs=bs,num_workers=4) 
        .normalize(imagenet_stats)      
       )


# In[ ]:


# Definition of Quadratic Kappa
from sklearn.metrics import cohen_kappa_score
def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')


# In[ ]:


# Densenet201 Model definition
# Set Callback for Early Stopping
learn = cnn_learner(data, base_arch=models.densenet201, metrics = [quadratic_kappa],
                    callback_fns=[partial(EarlyStoppingCallback, monitor='quadratic_kappa', min_delta=0.01, patience=3)]
                    )


# In[ ]:


# Fit to Data
learn.fit_one_cycle(4)


# In[ ]:


# Unfreeze and finding best LR
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


# First
learn.fit_one_cycle(20, max_lr=slice(5e-6,5e-5))


# In[ ]:


# Second for fitting again!!!
learn.fit_one_cycle(20, max_lr=slice(5e-5,5e-4))


# In[ ]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[ ]:


# Save the model
learn.export()
learn.save('stage-1')


# In[ ]:


# Again using data with default transformation and different Image Size
bs = 32 
sz=320
tfms = get_transforms()
src = (ImageList.from_df(df=df
                         ,path='./'
                         ,cols='path'
                        ) 
        .split_by_rand_pct(0.20) 
        .label_from_df(cols='diagnosis',label_cls=FloatList) 
      )
data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
        .databunch(bs=bs,num_workers=4) 
        .normalize(imagenet_stats)      
       )


# In[ ]:


# loading the weights and replacing the data
learn.load('stage-1') 
learn.data = data 


# In[ ]:


# Clear cache for GPU
torch.cuda.empty_cache()


# In[ ]:


learn.freeze()
learn.fit_one_cycle(4)


# In[ ]:


# Finding best LR
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


# Fit the model
learn.fit_one_cycle(20, max_lr=slice(1e-6, 8e-5))
learn.save('stage-2')


# In[ ]:


# Classification Interpretation
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()


# In[ ]:


# Predict the valid data set
valid_preds = learn.get_preds(ds_type=DatasetType.Valid)


# In[ ]:


# Optimizer Class for Classification
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[ ]:


# Fiting the optimizer
optR = OptimizedRounder()
optR.fit(valid_preds[0],valid_preds[1])


# In[ ]:


coefficients = optR.coefficients()


# In[ ]:


print(coefficients)


# In[ ]:


# Reading Sample Submission
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()


# In[ ]:


# Reading test data
learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))


# In[ ]:


# Using Fst AI TTA
preds,y = learn.TTA(ds_type=DatasetType.Test)


# In[ ]:


# Predict the Classes
test_predictions = optR.predict(preds, coefficients)


# In[ ]:


sample_df.diagnosis = test_predictions.astype(int)
sample_df.groupby('diagnosis').count()


# In[ ]:


sample_df.to_csv('submission.csv',index=False)

