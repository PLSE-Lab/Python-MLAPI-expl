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

import os
print(os.listdir("../input/fastai-pretrained"))
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:




import os
import cv2
import random
import numpy as np
import pandas as pd
import scipy as sp
import torch
from fastai.vision import *
import glob


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 2019
seed_everything(SEED)
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)


# In[ ]:


PATH = "../input/aptos2019-blindness-detection"
get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints/')
get_ipython().system('cp ../input/fastai-pretrained/densenet161-8d451a50.pth /tmp/.cache/torch/checkpoints/densenet161-8d451a50.pth')
get_ipython().system('cp ../input/fastai-pretrained/densenet169-b2777c0a.pth /tmp/.cache/torch/checkpoints/densenet169-b2777c0a.pth')


# In[ ]:


SIZE=300

train_df=pd.read_csv(PATH+'/train.csv')
test_df=pd.read_csv(PATH+'/sample_submission.csv')


# In[ ]:


train = ImageList.from_df(train_df, path=PATH, cols='id_code', folder="train_images", suffix='.png')
test = ImageList.from_df(test_df, path=PATH, cols='id_code', folder="test_images", suffix='.png')


# In[ ]:


def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')


# In[ ]:


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def open_aptos2019_image(fn, convert_mode, after_open)->Image:
    image = cv2.imread(fn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (SIZE, SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image,(0,0),10),-4,12,8)
    return Image(pil2tensor(image, np.float32).div_(255))

vision.data.open_image = open_aptos2019_image


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=360, max_warp=0,
                      max_zoom=1.3, max_lighting=0.1, p_lighting=0.5,p_affine=1.0)
from sklearn.metrics import cohen_kappa_score
data = (
    train.split_by_rand_pct(0.2)
    .label_from_df(cols='diagnosis', label_cls=FloatList)
    .add_test(test)
    .transform(tfms, size=SIZE)
    .databunch(path=Path('.'), bs=32,num_workers=4).normalize(imagenet_stats)
)


# In[ ]:


data.show_batch(rows=4,figsize=(15,15))


# In[ ]:


learn_1 = cnn_learner(data, models.densenet169, metrics=[quadratic_kappa], pretrained=True).mixup()


# In[ ]:


learn_1.fit_one_cycle(10,1e-02)
learn_1.recorder.plot_losses()
learn_1.recorder.plot_metrics()


# In[ ]:


learn_1.save("densenet161phase1")
learn_1.export()


# In[ ]:


learn_1.unfreeze()


# In[ ]:


learn_1.lr_find()


# In[ ]:


learn_1.recorder.plot()


# In[ ]:


learn_1.fit_one_cycle(5, max_lr=slice(7e-05,1e-04))
learn_1.recorder.plot_losses()
learn_1.recorder.plot_metrics()


# In[ ]:


learn_1.save("densenetphase2")
learn_1.export()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


valid_preds, valid_y = learn_1.get_preds(ds_type=DatasetType.Valid)

test_preds, _ = learn_1.get_preds(ds_type=DatasetType.Test)


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

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

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
optR.fit(valid_preds, valid_y)
coefficients = optR.coefficients()

valid_predictions = optR.predict(valid_preds, coefficients)[:,0].astype(int)
test_predictions = optR.predict(test_preds, coefficients)[:,0].astype(int)

valid_score = cohen_kappa_score(valid_y.numpy().astype(int), valid_predictions, weights="quadratic")


# In[ ]:


valid_score = cohen_kappa_score(valid_y.numpy().astype(int), valid_predictions, weights="quadratic")


# In[ ]:





# In[ ]:


print("coefficients:", coefficients)
print("validation score:", valid_score)


# In[ ]:


test_df.diagnosis = test_predictions
test_df.to_csv("submission.csv", index=None)
test_df.head()

