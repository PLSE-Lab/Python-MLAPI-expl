#!/usr/bin/env python
# coding: utf-8

# Initially I forked from this [kernel](https://www.kaggle.com/khursani8/fast-ai-starter-resnet34), changed architecture to ResNet 50, added augmentation and did some initial tuning of parameters like learning rate.

# In later versions I plugged in OptimizedRounder class and Ben's processing functions.

# # Libraries import

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


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

import PIL
import cv2


# In[ ]:


# Set seed for all
def seed_everything(seed=1358):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()


# # Ben's Preprocessing Functions

# These functions are taken from famous kernel https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping. Below I am showing how they can be applied for fast.ai pipeline. 

# In[ ]:


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

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

def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image


# # Data

# In[ ]:


PATH = Path('../input/aptos2019-blindness-detection')


# In[ ]:


df = pd.read_csv(PATH/'train.csv')
df.head()


# In[ ]:


get_ipython().system('ls ../input/resnet50/')


# In[ ]:


# copy pretrained weights for resnet50 to the folder fastai will search by default
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")


# In[ ]:


df.diagnosis.value_counts() 


# So our train set is definitely imbalanced, majority of images are normal (without illness).

# # Model

# In[ ]:


IMG_SIZE = 512

def _load_format(path, convert_mode, after_open)->Image:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0), 10) ,-4 ,128)
                    
    return Image(pil2tensor(image, np.float32).div_(255)) #return fastai Image format

vision.data.open_image = _load_format
    
src = (
    ImageList.from_df(df,PATH,folder='train_images',suffix='.png')
        .split_by_rand_pct(0.2, seed=42)
        .label_from_df(cols='diagnosis',label_cls=FloatList)    
    )
src


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.10, max_zoom=1.3, max_warp=0.0, max_lighting=0.2)


# Let's train with small image size first to get some rough approximation

# In[ ]:


data = (
    src.transform(tfms,size=128)
    .databunch()
    .normalize(imagenet_stats)
)
data


# In[ ]:


# Definition of Quadratic Kappa
from sklearn.metrics import cohen_kappa_score
def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')

learn = cnn_learner(data, base_arch=models.resnet50 ,metrics=[quadratic_kappa],model_dir='/kaggle',pretrained=True)


# In[ ]:


# Find a good learning rate
learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-2
learn.fit_one_cycle(3, lr)


# Now switching to 224x224 size which is usually used for ResNet 50:

# In[ ]:


# progressive resizing
learn.data = data = (
    src.transform(tfms,size=224)
    .databunch()
    .normalize(imagenet_stats)
)
learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-2
learn.fit_one_cycle(20, lr)


# In[ ]:


learn.unfreeze()

learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8, slice(1e-6,1e-3))


# # Metric Optimization

# This part is taken from @abhishek great kernel: https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa

# In[ ]:


valid_preds = learn.get_preds(ds_type=DatasetType.Valid)


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
optR.fit(valid_preds[0],valid_preds[1])


# In[ ]:


coefficients = optR.coefficients()
print(coefficients)


# 
# 
# # Predictions

# In[ ]:


# test_df = pd.read_csv(PATH/'test.csv')
# test_df.head()
sample_df = pd.read_csv(PATH/'sample_submission.csv')
sample_df.head()


# In[ ]:


learn.data.add_test(ImageList.from_df(sample_df,PATH,folder='test_images',suffix='.png'))


# In[ ]:


preds,y = learn.get_preds(DatasetType.Test)


# In[ ]:


test_predictions = optR.predict(preds, coefficients)


# In[ ]:


sample_df.diagnosis = test_predictions.astype(int)
sample_df.head()


# In[ ]:


sample_df.to_csv('submission.csv',index=False)

