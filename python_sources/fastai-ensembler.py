#!/usr/bin/env python
# coding: utf-8

# **BEFORE YOU FORK, PLEASE SUPPORT AND UPVOTE**

# Original Kernel taken from https://www.kaggle.com/tanlikesmath/intro-aptos-diabetic-retinopathy-eda-starter
# 
# Changes done:
# 
# 1. Change image size to 256
# 2. Fix TTA
# 3. Add stratified KFold splitting

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


if not os.path.exists('models/'):
        os.makedirs('models')

get_ipython().system("cp '../input/aptosresnet152/resnet50-3.pth' 'models/resnet50.pth'")
get_ipython().system("cp '../input/aptosresnet152/stage-2.pth' 'models/resnet152.pth'")


# In[ ]:


import os
os.listdir('.')


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 999
seed_everything(SEED)


# In[ ]:


base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')
train_dir = os.path.join(base_image_dir,'train_images/')
df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
df = df.drop(columns=['id_code'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)


# In[ ]:


len_df = len(df)
print(f"There are {len_df} images")


# In[ ]:


folds = pd.read_csv('../input/atposfolds/folds.csv')


# The images are actually quite big. We will resize to a much smaller size.

# In[ ]:


fold_num = 1
val_idxs = folds[folds['folds'] == fold_num].index.values


# In[ ]:


bs = 64 #smaller batch size is better for training, but may take longer
sz=256


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=360, max_warp=0,
                      max_zoom=1.1, max_lighting=0.1, p_lighting=0.5)

src = (ImageList.from_df(df=df,path='./',cols='path')
        .split_by_idx(val_idxs)
        .label_from_df(cols='diagnosis', label_cls=FloatList)
      )

data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros')
       .databunch(bs=bs,num_workers=4)
       .normalize(imagenet_stats) #Normalize     
       )


# In[ ]:


from sklearn.metrics import cohen_kappa_score
def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')


# In[ ]:


learn = cnn_learner(data, base_arch=models.resnet50, metrics = [quadratic_kappa], pretrained=False)
learn.load('resnet50')


# In[ ]:


learn_2 = cnn_learner(data, base_arch=models.resnet152, metrics = [quadratic_kappa], pretrained=False)
learn_2.load('resnet152')


# Inference

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


import numpy as np
import pandas as pd
import os
import scipy as sp
from functools import partial
from sklearn import metrics
from collections import Counter
import json


# In[ ]:


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


optR = OptimizedRounder()


# In[ ]:


# coefficients = optR.coefficients()
# coefficients = [0.535809, 1.569422, 2.61038,  3.090442]
coefficients = [0.499944, 1.577832, 2.627495, 3.263393]
coefficients_2 = [0.54458,  1.570697, 2.664879, 2.892028]

print(coefficients)
print(coefficients_2)


# ## Submission
# Let's now create a submission

# In[ ]:


sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()


# ### ResNet50

# In[ ]:


learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))


# In[ ]:


preds,y = learn.get_preds(ds_type=DatasetType.Test)


# ### ResNet152

# In[ ]:


learn_2.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))


# In[ ]:


preds_2, y_2 = learn_2.get_preds(ds_type=DatasetType.Test)


# ### Ensembling

# In[ ]:


preds_avg = (preds * 0.6 + preds_2 * 0.4)
test_predictions = optR.predict(preds_avg, coefficients)


# In[ ]:


sample_df.diagnosis = test_predictions.astype(int)
sample_df.head()


# In[ ]:


sample_df.to_csv('submission.csv',index=False)


# In[ ]:


sample_df.diagnosis.value_counts()


# In[ ]:




