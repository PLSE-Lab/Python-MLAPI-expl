#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from functools import partial
from pathlib import Path
from tqdm.notebook import tqdm
home = Path(".")

dir_q = Path("/kaggle/input/")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
from pathlib import Path
from fastai import *
from fastai.vision import *
from fastai.callbacks import TrackerCallback, SaveModelCallback
import os
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import random
from fastai.callbacks import *
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)


# In[ ]:


dir_q.ls()


# In[ ]:


import shutil
import cv2
import re


# In[ ]:


df = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
df.head()


# In[ ]:


def root(x):
  ans='grapheme_root_'+str(x)
  return ans
def vowel(x):
  ans='vowel_diacritic_'+str(x)
  return ans
def conso(x):
  ans='consonant_diacritic_'+str(x)
  return ans


# In[ ]:


df['grapheme_root']=df.grapheme_root.apply(root,convert_dtype=True)
df['vowel_diacritic']=df.vowel_diacritic.apply(vowel,convert_dtype=True)
df['consonant_diacritic']=df.consonant_diacritic.apply(conso,convert_dtype=True)


# In[ ]:


df['category']=df['grapheme_root']+'/'+df['vowel_diacritic']+'/'+df['consonant_diacritic']


# In[ ]:


df.head()


# In[ ]:


df.to_csv('train_new.csv',index=False)


# In[ ]:


dir_q.ls()


# In[ ]:


sz=128
bs=128


# In[ ]:


stats = ([0.0692], [0.2051])
data = (ImageList.from_df(df, path=dir_q, folder='grapheme-imgs-128x128', suffix='.png', 
        cols='image_id')
        .split_by_rand_pct(0.3)
        .label_from_df(cols=['category'],label_delim='/')
        .transform(get_transforms(do_flip=False,max_warp=0.1), size=sz, padding_mode='zeros')
        .databunch(bs=bs)).normalize(stats)


# In[ ]:


data.show_batch(rows=3,figsize=(14,6))


# In[ ]:


len(data.classes)


# In[ ]:


dir_q.ls()


# In[ ]:


arch = models.resnet34


# In[ ]:


get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints/')
get_ipython().system('cp /kaggle/input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth')


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2,beta=0.2)
try:
    learn = cnn_learner(data, arch, metrics = [acc_02, f_score],model_dir='/kaggle/working').to_fp16()
except:
    get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints/')
    get_ipython().system('cp /kaggle/input/resnet34/resnet34.pth /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth')
    
    learn = cnn_learner(data, arch, metrics = [acc_02, f_score],model_dir='/kaggle/working').to_fp16()
    
    


# In[ ]:


lr=0.01


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# # As above declared lr=0.01

# In[ ]:


learn.fit_one_cycle(1,slice(lr))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('model1')


# In[ ]:


learn.load('model1')
learn.unfreeze()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


learn.export('/kaggle/working/export.pkl')


# In[ ]:


import cv2
from tqdm import tqdm_notebook as tqdm
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


dir_q.ls()


# In[ ]:


HEIGHT = 137
WIDTH = 236
SIZE = 128

TRAIN = ['/kaggle/input/bengaliai-cv19/test_image_data_3.parquet',
        '/kaggle/input/bengaliai-cv19/test_image_data_2.parquet',
        '/kaggle/input/bengaliai-cv19/test_image_data_1.parquet',
        '/kaggle/input/bengaliai-cv19/test_image_data_0.parquet',]

OUT_TRAIN = 'test.zip'


# * # code image processing taken from
# https://www.kaggle.com/iafoss/image-preprocessing-128x128

# In[ ]:


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))


# In[ ]:


x_tot,x2_tot = [],[]
with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:
    for fname in TRAIN:
        df1 = pd.read_parquet(fname)
        #the input is inverted
        data = 255 - df1.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
        for idx in tqdm(range(len(df1))):
            name = df1.iloc[idx,0]
            #normalize each image by its max val
            img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)
            img = crop_resize(img)
        
            x_tot.append((img/255.0).mean())
            x2_tot.append(((img/255.0)**2).mean()) 
            img = cv2.imencode('.png',img)[1]
            img_out.writestr(name + '.png', img)


# In[ ]:


get_ipython().system("unzip '/kaggle/working/test.zip'")


# In[ ]:


defaults.device = torch.device('cuda')


# In[ ]:


learnp = load_learner('/kaggle/input/grapheme-model/').to_fp16()


# In[ ]:


test_set=['/kaggle/working/Test_0.png','/kaggle/working/Test_1.png','/kaggle/working/Test_2.png',
          '/kaggle/working/Test_3.png','/kaggle/working/Test_4.png','/kaggle/working/Test_5.png',
          '/kaggle/working/Test_6.png','/kaggle/working/Test_7.png','/kaggle/working/Test_8.png',
          '/kaggle/working/Test_9.png','/kaggle/working/Test_10.png','/kaggle/working/Test_11.png'
          ]


# In[ ]:


img2 = open_image(test_set[2])
pred_class,pred_idx,outputs = learnp.predict(img2)
print("prediction : "+str(pred_class))
img2


# In[ ]:


sample=pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')


# In[ ]:


sample.head(36)


# In[ ]:


k=0
for i in range(0,36,3):
    img = open_image(test_set[k])
    pred_class,pred_idx,outputs = learn.predict(img)
    ans=str(pred_class).split(";")
    if len(ans)==2:
        ans.insert(1,'grapheme_root_72')
    k=k+1
    for j in range(3):
        value=ans[j].split("_")
        sample.loc[[i+j],'target']=int(value[-1])

        


# In[ ]:


get_ipython().system("rm '/kaggle/working/test.zip'")


# In[ ]:


sample.to_csv('submission.csv',index=False)


# In[ ]:


sample.head(36)

