#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
torch.__version__


# In[ ]:


#!pip3 install fastai==1.0.30


# In[ ]:


import fastai
fastai.__version__


# In[ ]:


#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline


# In[ ]:


get_ipython().system('git clone https://github.com/wdhorton/protein-atlas-fastai')


# In[ ]:


get_ipython().system('ls /kaggle/working/protein-atlas-fastai')


# In[ ]:


import sys
 # Add directory holding utility functions to path to allow importing utility funcitons
#sys.path.insert(0, '/kaggle/working/protein-atlas-fastai')
sys.path.append('/kaggle/working/protein-atlas-fastai')


# In[ ]:


from fastai.vision import * 
from fastai import *


# In[ ]:


import cv2
import numpy as np

from fastai.vision.image import *


# adapted from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
def open_4_channel(fname):
    fname = str(fname)
    # strip extension before adding color
    if fname.endswith('.png'):
        fname = fname[:-4]
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(fname+'_'+color+'.png', flags).astype(np.float32)/255
           for color in colors]
    
    x = np.stack(img, axis=-1)
    return Image(pil2tensor(x, np.float32).float())


# In[ ]:


#import utils


# In[ ]:


import os
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from fastai import *
from fastai.vision import *

#from utils import open_4_channel
from resnet import Resnet4Channel


# In[ ]:


bs = 16


# In[ ]:


path = Path('../input/')


# In[ ]:


df = pd.read_csv(path/'train.csv')
df.head()


# In[ ]:


np.random.seed(42)
src = (ImageItemList.from_csv(path, 'train.csv', folder='train', suffix='.png',num_workers=0)
       .random_split_by_pct(0.2)
       .label_from_df(sep=' ',  classes=[str(i) for i in range(28)]))


# In[ ]:


src.train.x.create_func = open_4_channel
src.train.x.open = open_4_channel


# In[ ]:


src.valid.x.create_func = open_4_channel
src.valid.x.open = open_4_channel


# In[ ]:


test_ids = list(sorted({fname.split('_')[0] for fname in os.listdir(path/'test')}))


# In[ ]:


test_fnames = [path/'test'/test_id for test_id in test_ids]


# In[ ]:


test_fnames[:5]


# In[ ]:


src.add_test(test_fnames, label='0');


# In[ ]:


src.test.x.create_func = open_4_channel
src.test.x.open = open_4_channel


# In[ ]:


protein_stats = ([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])


# In[ ]:


trn_tfms,_ = get_transforms(do_flip=True, flip_vert=True, max_rotate=30., max_zoom=1,
                      max_lighting=0.05, max_warp=0.)


# In[ ]:


data = (src.transform((trn_tfms, _), size=224).databunch(num_workers=0).normalize(protein_stats))


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


def resnet50(pretrained):
    return Resnet4Channel(encoder_depth=50)


# In[ ]:


# copied from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
def _resnet_split(m): return (m[0][6],m[1])


# In[ ]:


f1_score = partial(fbeta, thresh=0.2, beta=1)


# In[ ]:


TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"


# In[ ]:


learn = create_cnn(
    data,
    resnet50,
    cut=-2,
    split_on=_resnet_split,
    loss_func=F.binary_cross_entropy_with_logits,
    path=path,    
    metrics=[f1_score], 
    #temp_dir=TMP_PATH, 
    model_dir=MODEL_PATH
)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 3e-2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


#learn.save('stage-1-rn50-datablocks')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(15, slice(3e-5, lr/5))


# In[ ]:


#learn.save('stage-2-rn50')


# In[ ]:


preds,_ = learn.get_preds(DatasetType.Test)


# In[ ]:


path_output = Path('/kaggle/working')


# In[ ]:


pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>0.2)[0]])) for row in np.array(preds)]
df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
df.to_csv(path_output/'protein_predictions_datablocks.csv', header=True, index=False)


# In[ ]:


get_ipython().system('rm -rf /kaggle/working/protein-atlas-fastai')


# In[ ]:


get_ipython().system('ls /kaggle/working')


# In[ ]:




