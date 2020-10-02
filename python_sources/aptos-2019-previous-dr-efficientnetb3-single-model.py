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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install efficientnet_pytorch')


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# In[ ]:


from fastai import *
from fastai.vision import *
from efficientnet_pytorch import *
import pandas as pd
import matplotlib.pyplot as plt


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


print('Make sure cuda is installed:', torch.cuda.is_available())
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)


# In[ ]:


from sklearn.metrics import cohen_kappa_score
def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')


# In[ ]:


bs = 32 #smaller batch size is better for training, but may take longer
sz=300


# In[ ]:


base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')
train_dir = os.path.join(base_image_dir,'train_images/')
df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
df = df.drop(columns=['id_code'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)


# In[ ]:


import cv2
def open_aptos2019_image(fn, convert_mode, after_open,tol=7)->Image:
    img = cv2.imread(fn)
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
            img = np.stack([img3,img2,img1],axis=-1)
    #         print(img.shape)
        return Image(pil2tensor(img, np.float32).div_(255))
    

    

vision.data.open_image = open_aptos2019_image


# In[ ]:


from sklearn.model_selection import KFold
n_fold = 5
seed = 42
folds = KFold(n_splits=n_fold, shuffle=True, random_state=seed)        


# In[ ]:


md_ef = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1)


# In[ ]:


from fastai.callbacks.hooks import *
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df['path'],df['diagnosis'])):
    if fold_ != 0:
        continue
    tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.3,max_lighting=0.1,p_lighting=0.5)
    src = (ImageList.from_df(df=df,path='./',cols='path') #get dataset from dataset
        .split_by_idx(val_idx) #Splitting the dataset
        .label_from_df(cols='diagnosis',label_cls=FloatList) #obtain labels from the level column
      )
    data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') #Data augmentation
        .databunch(bs=bs,num_workers=4) #DataBunch
        .normalize(imagenet_stats) #Normalize     
       )
    learn = Learner(data,md_ef, wd=1e-2, metrics = [quadratic_kappa])
    learn.split([learn.layer_groups[0][:-1],learn.layer_groups[0][-1]])
    #concat_pool = True
    #bn_final = False
    #lin_ftrs = None
    #ps=0.5
   # nf = num_features_model(nn.Sequential(*learn.layer_groups[0].children())) * (2 if concat_pool else 1)
    #nf = 1536
    #head = create_head(nf, data.c, lin_ftrs, ps=ps, concat_pool=concat_pool, bn_final=bn_final)
    prev_dir = learn.model_dir
    learn.model_dir = Path('.')
    learn.load('../input/dr-efficientnet-b3/efficientnet300').to_fp16()
    learn.model_dir = prev_dir
    #src = (ImageList.from_df(df=df,path='./',cols='path',label_cls=FloatList) #get dataset from dataset
    #    .split_by_idx(val_idx) #Splitting the dataset
    #    .label_from_df(cols='diagnosis') #obtain labels from the level column
    #  )
    #learn.data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') #Data augmentation
    #    .databunch(bs=bs,num_workers=4) #DataBunch
    #    .normalize(imagenet_stats) #Normalize     
    #   )
    #learn.loss_func = learn.data.loss_func
    #learn.model[-1][-1]=nn.Linear(in_features=512,out_features=1, bias=True).cuda()


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1,max_lr=1e-3)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


from fastai.callbacks import *
learn.fit_one_cycle(30,max_lr=slice(2e-3),callbacks=[SaveModelCallback(learn, every='improvement',monitor='valid_loss')])
learn.export('APTOS-previous-0')
validate = learn.validate()


# In[ ]:


print(validate[1].cpu().numpy())


# In[ ]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()

