#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# In[ ]:


path=Path('/kaggle/input/imet-2020-fgvc7')


# In[ ]:


df = pd.read_csv(path/'train.csv')
df.head()


# In[ ]:


tfms = get_transforms(max_lighting=0.1, max_zoom=1.05, max_warp=0.)
#removed the vert flip


# In[ ]:


np.random.seed(42)
src = (ImageList.from_csv(path, 'train.csv', folder='train', suffix='.png')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))


# In[ ]:


data = (src.transform(tfms, size=224)
        .databunch(bs=200).normalize(imagenet_stats))
#swappng to good default for resnet34


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


arch = models.resnet34


# ## Getting pretrained model to start from with no internet
# Thanks to https://www.kaggle.com/aminyakubu/aptos-2019-blindness-detection-fast-ai
# for the guidance on how to load pretrained models into this.  I just watched the error for learn = cnn_learner(....) from two cells down to identify the directory to copy the pth file to and what the postfixed id was.
# 
# This is the original instructions shared:  Later, I use resnet as the base architecture. However, since we can't use the internet for this kernel in this competition I will set these directories which will contain the models. This is because, cnn_learner will check those directories first before attempting to download. When internet for the kernel is turned off and these models don't exist, an error will be raised. To add the models, click on add dataset at the top right corner of this kernel and search for resnet. Make sure to choose resnet for PyTorch
# 
# Thanks Amin!  https://www.kaggle.com/aminyakubu

# In[ ]:


# creating directories and copying the models to those directories
get_ipython().system('mkdir -p /root/.cache/torch/checkpoints')
get_ipython().system('cp ../input/resnet34fastai/resnet34.pth /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth')


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, arch, metrics=[acc_02, f_score])


# In[ ]:


get_ipython().system(' ls ../input')


# In[ ]:


get_ipython().system('cp ../input/resnet34stage1imet2020/stage-1-rn34.pth /kaggle/working')
get_ipython().system('cp ../input/resnet34stage1imet2020/stage-2-rn34-2.pth /kaggle/working')
get_ipython().system('cp ../input/resnet34imet2020stage2/stage-2-rn34\\ \\(1st\\ try\\).pth /kaggle/working/stage-2-rn34-1.pth')


# In[ ]:


learn.model_dir = Path('/kaggle/working')


# In[ ]:


learn.load('stage-1-rn34')
learn.freeze()


# In[ ]:



#learn.lr_find()
#skip lrfind to save us 10 mins execution


# In[ ]:


#learn.recorder.plot()
#nothing to plot if didnt run lr_find


# In[ ]:


lr = 0.01


# In[ ]:


#learn.fit_one_cycle(1, slice(lr))


# In[ ]:


learn.save('stage-1-rn34') #save our stage one
#learn.load('stage-2-rn34-2') #load our pretrained finetuned model
learn.load('stage-2-rn34-1') #load our 1st pretrained finetuned model


# In[ ]:


learn.unfreeze()


# In[ ]:


#learn.lr_find()


# In[ ]:


#learn.recorder.plot()


# In[ ]:


#learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-5)) #did 10 on GCP


# In[ ]:


learn.path = Path('/kaggle/working')
learn.export()


# In[ ]:


test = ImageList.from_folder(path/'test')
len(test)


# In[ ]:


learn = load_learner(Path('/kaggle/working'), test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[ ]:


labelled_preds[:5]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]


# In[ ]:


df = pd.DataFrame({'id':fnames, 'attribute_ids':labelled_preds})


# In[ ]:


outputpath = Path('/kaggle/working')
df.to_csv(outputpath/'submission.csv', index=False)

