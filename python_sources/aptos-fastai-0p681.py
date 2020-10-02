#!/usr/bin/env python
# coding: utf-8

# # APTOS - Fastai
# 
# Simple code for beginners

# This notebook is using fastai library with default parameters. Resnet-50 pretrained model.
# 
# Learning Rate--> max_lr=slice(1e-4,1e-3)
# 
# Epochs=11
# 
# Bens cropping and preprocessing

# In[ ]:


import time
t_start = time.time()


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks import *


# In[ ]:


from functools import partial
from sklearn import metrics
from collections import Counter


# In[ ]:


import os
import PIL
import cv2


# In[ ]:


pwd


# In[ ]:


print(os.listdir("../input/"))


# In[ ]:


print(os.listdir("../input/aptos2019-blindness-detection"))


# In[ ]:


sys.path.insert(0, '../input/aptos2019-blindness-detection')
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")


# In[ ]:


dat_path = Path('../input/aptos2019-blindness-detection')
trn_path = Path('../input/aptos2019-blindness-detection/train_images')
(dat_path,trn_path)


# In[ ]:


train_df = pd.read_csv(dat_path/'train.csv')
train_df.shape


# In[ ]:


train_df.head()


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


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.10, max_zoom=1.3, max_warp=0.0, max_lighting=0.2)


# In[ ]:


bs = 64


# In[ ]:


data = ImageDataBunch.from_df(path=dat_path, folder='train_images', df=train_df, suffix='.png', 
                               fn_col=0, label_col=1,valid_pct=0.2, size=256, ds_tfms=tfms).normalize(imagenet_stats)


# In[ ]:


#data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


#from sklearn.metrics import cohen_kappa_score
#def quadratic_kappa(y_hat, y):
#    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')


# In[ ]:


kappa = KappaScore()
kappa.weights = "quadratic"


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=[error_rate, kappa])


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(9, max_lr=slice(1e-4,1e-3))


# In[ ]:


#learn.unfreeze()
#learn.fit_one_cycle(1, max_lr=slice(1e-4,1e-3))


# In[ ]:


learn.model_dir = Path('/kaggle/working/')


# In[ ]:


learn.save('/kaggle/working/FastAI_APTOS_epoch_11')


# In[ ]:


test_df = pd.read_csv(dat_path/'sample_submission.csv')


# In[ ]:


test_df.head()


# In[ ]:


learn.data.add_test(ImageList.from_df(test_df,dat_path,folder='test_images',suffix='.png'))


# In[ ]:


preds,y = learn.get_preds(DatasetType.Test)


# In[ ]:


test_df.diagnosis = preds.argmax(1)
test_df.head()


# In[ ]:


test_df.to_csv('submission.csv',index=False)


# In[ ]:


t_finish = time.time()
total_time = round((t_finish-t_start) / 3600, 4)
print('Kernel runtime = {} hours ({} minutes)'.format(total_time, 
                                                      int(total_time*60)))

