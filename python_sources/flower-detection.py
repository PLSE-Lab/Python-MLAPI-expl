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

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/flower-recognition-he/he_challenge_data/data/train.csv')


# In[ ]:


df_psuLabels = pd.read_csv('/kaggle/input/pseudolabel/subm.csv')


# In[ ]:


concat_df = pd.concat([df, df_psuLabels])


# In[ ]:


from sklearn.utils import shuffle
new_df = shuffle(concat_df)


# In[ ]:


get_ipython().system('mkdir /kaggle/working/all_img')


# In[ ]:


get_ipython().system('cp /kaggle/input/flower-recognition-he/he_challenge_data/data/test/*jpg /kaggle/working/all_img')


# In[ ]:


get_ipython().system('cp /kaggle/input/flower-recognition-he/he_challenge_data/data/train/*jpg /kaggle/working/all_img')


# In[ ]:


from fastai.vision import *
from fastai import *


# In[ ]:


from pathlib import Path
path = Path('/kaggle/working')
path


# In[ ]:


sz = 128
bs = 32
tfms = get_transforms(do_flip=True,
                      max_rotate=15,
                      max_warp=0.,
                      max_lighting=0.1,
                      p_lighting=0.3
                     )
src = (ImageList.from_df(df=df
                         ,path=path/'all_img'
                         ,cols='image_id'
                         , suffix = '.jpg'
                         #,convert_mode='L'
                        ) 
        .split_by_rand_pct(0) 
        .label_from_df(cols='category') 
      )
data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='reflection') 
        .databunch(bs=bs,num_workers=4) 
        .normalize(imagenet_stats)      
       )


# In[ ]:


data.show_batch()


# In[ ]:


from fastai.callbacks import *


# In[ ]:


learn = cnn_learner(data, base_arch=models.densenet201, metrics = [accuracy],
                    callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.01, patience=2)],path = '/kaggle/working/', model_dir = '/kaggle/working/'
                    ).mixup()


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(20, max_lr=slice(1e-5,1e-4))


# In[ ]:


learn.save('stg-1')


# In[ ]:


sz = 256
bs = 32
tfms = get_transforms(do_flip=True,
                      max_rotate=15,
                      max_warp=0.,
                      max_lighting=0.1,
                      p_lighting=0.3
                     )
src = (ImageList.from_df(df=df
                         ,path=path/'all_img'
                         ,cols='image_id'
                         , suffix = '.jpg'
                         #,convert_mode='L'
                        ) 
        .split_by_rand_pct(0) 
        .label_from_df(cols='category') 
      )
data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='reflection') 
        .databunch(bs=bs,num_workers=4) 
        .normalize(imagenet_stats)      
       )


# In[ ]:


learn.load('stg-1')


# In[ ]:


learn.data = data 


# In[ ]:


learn.freeze()
learn.fit_one_cycle(4)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(20, max_lr=slice(7e-6, 1e-5), wd = 1e-1)


# In[ ]:


learn.save('stg-2')


# In[ ]:


sz = 320
bs = 32
tfms = get_transforms(do_flip=True,
                      max_rotate=15,
                      max_warp=0.,
                      max_lighting=0.1,
                      p_lighting=0.3
                     )
src = (ImageList.from_df(df=df
                         ,path=path/'all_img'
                         ,cols='image_id'
                         , suffix = '.jpg'
                         #,convert_mode='L'
                        ) 
        .split_by_rand_pct(0) 
        .label_from_df(cols='category') 
      )
data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='reflection') 
        .databunch(bs=bs,num_workers=4) 
        .normalize(imagenet_stats)      
       )


# In[ ]:


learn.load('stg-2')


# In[ ]:


learn.data = data 


# In[ ]:


learn.freeze()
learn.fit_one_cycle(4)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(20, max_lr=slice(5e-5, 1e-5), wd = 1e-1)


# In[ ]:


learn.save('stg-3')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()


# In[ ]:


valid_preds = learn.get_preds(ds_type=DatasetType.Valid)


# In[ ]:


sample_df = pd.read_csv('/kaggle/input/flower-recognition-he/he_challenge_data/data/sample_submission.csv')
sample_df.head()


# In[ ]:


learn.data.add_test(ImageList.from_df(sample_df,'/kaggle/input/flower-recognition-he/he_challenge_data/data/',folder='test',suffix='.jpg'))


# In[ ]:


preds,y = learn.TTA(ds_type=DatasetType.Test)


# In[ ]:


labelled_preds = []


# In[ ]:


for pred in preds:
    labelled_preds.append(int(np.argmax(pred))+1)


# In[ ]:


sample_df.category = labelled_preds
sample_df.groupby('category').count()


# In[ ]:


sample_df.to_csv('submission.csv',index=False)


# In[ ]:


sample_df = sample_df.sort_values(by = ['image_id'], ascending = [True])


# In[ ]:


learn.export('dense161.pkl')


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "subm.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(sample_df)


# In[ ]:



interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:



interp.most_confused(min_val=2)


# In[ ]:




