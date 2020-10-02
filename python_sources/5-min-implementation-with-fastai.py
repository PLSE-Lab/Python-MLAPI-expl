#!/usr/bin/env python
# coding: utf-8

# ### Hi everyone !! This is a super quick implementation using fastai.
# 
# 1. I am using Resnet50 which has got me an F1 score of about 68% training time per epoch is abt 6min
# 
# 2. Earlier I has tried resnet 34 which had god me an score of 64% training time was 4min
# 
# 
# note : all the above experiments were done for 5 epochs becz it takes long for the model to run.
# 
# I have not applied any transforms here as this requires a lot of experimentation.
# results of the transformers will be updated bu tomorrow.
# 
# 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from fastai import *
from fastai.vision import *
from torchvision.models import *


# In[ ]:


# path = '../input/jovian-pytorch-z2g/Human protein atlas/train.csv'

path = '../input/jovian-pytorch-z2g/Human protein atlas/'


# In[ ]:





# In[ ]:


transforms = get_transforms(do_flip = True,flip_vert = True,
                            max_rotate = 10.0,max_zoom = 1.1,
                            max_lighting = 0.2,max_warp = 0.2,
                            p_lighting = 0.75)


# In[ ]:


# transforms = get_transforms


# In[ ]:


# ImageDataBunch.from_csv??


# In[ ]:


np.random.seed(42) # set random seed so we always get the same validation set
data = ImageDataBunch.from_csv(path,folder = 'train', suffix='.png',csv_labels = 'train.csv',
                              label_delim = ' ',bs = 8)


# In[ ]:


data.x[0].shape


# In[ ]:


len(data.train_ds.x) / 8


# In[ ]:



data.show_batch(rows=3, figsize=(12, 9))


# In[ ]:


def F_score(output, label, threshold=0.2, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)


# In[ ]:



learn = cnn_learner(data, models.resnet50,metrics = [F_score])


# In[ ]:





# In[ ]:


learn.fit_one_cycle(20)


# In[ ]:


# learn.fit(2,lr = 3e-4)


# In[ ]:


# learn.one_cycle_scheduler

preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn,preds,y,losses)


# In[ ]:


interp.plot_multi_top_losses()


# # Test Data

# In[ ]:


def decode_target(target, text_labels=False, threshold=0.5):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)


# In[ ]:


import gc 
gc.collect()


# In[ ]:


# import pandas as pd


# In[ ]:


subs = pd.read_csv('../input/jovian-pytorch-z2g/submission.csv')

test = '../input/jovian-pytorch-z2g/Human protein atlas/test'

from tqdm import tqdm


preds =  []
for i in tqdm(subs['Image']):
    
    img = open_image(test + '/' + str(i) + '.png')
    
    
    preds.append(learn.predict(img)[2])
        


# In[ ]:


import gc
gc.collect()


# In[ ]:


preds[0:2]


# In[ ]:


final_preds = [decode_target(i) for i in preds]


# In[ ]:


final_preds


# In[ ]:


subs.head()


# In[ ]:


subs['Label']  = final_preds


# In[ ]:


subs.to_csv('sub_7.csv',index = False)


# In[ ]:


subs


# In[ ]:


#intepreter

