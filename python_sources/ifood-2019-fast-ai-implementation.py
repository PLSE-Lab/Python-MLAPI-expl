#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system(' pip install pretrainedmodels')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
from matplotlib import style
import seaborn as sns

style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

from sklearn.metrics import confusion_matrix
from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
from sklearn.utils import shuffle


# In[ ]:


from fastai import *
from fastai.vision import *
import pretrainedmodels


# In[ ]:


train_dir = '../input/ifood-2019-fgvc6/train_set/train_set/'
val_dir = '../input/ifood-2019-fgvc6/val_set/val_set/'


# In[ ]:


train_df = pd.read_csv('../input/ifood-2019-fgvc6/train_labels.csv')
train_df['path'] = train_df['img_name'].map(lambda x: os.path.join(train_dir,x))
val_df = pd.read_csv('../input/ifood-2019-fgvc6/val_labels.csv')
val_df['path'] = val_df['img_name'].map(lambda x: os.path.join(val_dir,x))


# In[ ]:


df = pd.concat([train_df, val_df], ignore_index=True)
df.head()


# In[ ]:


val_idx = [i for i in range(len(train_df), len(df))]


# In[ ]:


sz = 256
bs = 32


# In[ ]:


np.random.seed(42)
tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.2,max_lighting=0.5,p_lighting=0.5, p_affine=0.5)
src = (ImageList.from_df(df=df,path='./',cols='path') #get dataset from dataset
       .split_by_idx(val_idx)
        .label_from_df(cols='label') #obtain labels from the level column
      )
data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='border') #Data augmentation
        .databunch(bs=bs,num_workers=4) #DataBunch
        .normalize(imagenet_stats) #Normalize
       )


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


import gc
gc.collect()


# In[ ]:


def top_3_accuracy(preds, targs):
    return top_k_accuracy(preds, targs, 3)


# In[ ]:


model_name = 'se_resnext101_32x4d'
def get_cadene_model(pretrained=True, model_name='se_resnext101_32x4d'):
    if pretrained:
        arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    else:
        arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
    return arch


# In[ ]:


try:
    os.mkdir('/tmp/.cache/')
except:
    pass

try:
    os.mkdir('/tmp/.cache/torch')
except:
    pass

try:
    os.mkdir('/tmp/.cache/torch/checkpoints')
except:
    pass

get_ipython().system(' cp ../input/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth /tmp/.cache/torch/checkpoints/se_resnext101_32x4d-3b2fe3d8.pth')


# In[ ]:


arch = get_cadene_model


# In[ ]:


learn = cnn_learner(data, get_cadene_model, metrics=[top_3_accuracy], model_dir='/tmp/models', loss_func = LabelSmoothingCrossEntropy()).mixup()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


stage = 1
csvlogger = callbacks.CSVLogger(learn=learn, filename='history_stage_'+str(stage)+'_'+model_name, append=True)
saveModel = callbacks.SaveModelCallback(learn, every='epoch',
                                        monitor='top_3_accuracy', mode='max',
                                        name='stage_'+str(stage))
reduceLR = callbacks.ReduceLROnPlateauCallback(learn=learn, monitor = 'top_3_accuracy', mode = 'max', patience = 1, factor = 0.5)


# In[ ]:


lr = 3e-3
learn.fit_one_cycle(4, slice(lr))


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


learn.save('stage-1-SE_Resnext101')


# In[ ]:


# learn.data.batch_size = 48
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


stage = 2
csvlogger = callbacks.CSVLogger(learn=learn, filename='history_stage_'+str(stage)+'_'+model_name, append=True)
saveModel = callbacks.SaveModelCallback(learn, every='epoch',
                                        monitor='top_3_accuracy', mode='max',
                                        name='stage_'+str(stage))
reduceLR = callbacks.ReduceLROnPlateauCallback(learn=learn, monitor = 'top_3_accuracy', mode = 'max', patience = 1, factor = 0.5)


# In[ ]:


learn.fit_one_cycle(10 , slice(1e-5, 1e-3))


# In[ ]:


learn.save('stage-2-SE_Resnext101')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


test = ImageList.from_folder('../input/ifood-2019-fgvc6/test_set')
len(test)


# In[ ]:


learn.export('/tmp/export.pkl')
learn = load_learner('/tmp/', test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


preds.shape


# In[ ]:


fnames = [f.name for f in learn.data.test_ds.items]
fnames[:4]
# labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred)]) for pred in preds]


# In[ ]:


col = ['img_name']
test_df = pd.DataFrame(fnames, columns=col)
test_df['label'] = ''


# In[ ]:


predictions = np.array(preds).reshape(len(preds), 251)
predictions.shape


# In[ ]:


from tqdm import tqdm_notebook as T
for i, pred in T(enumerate(predictions), total=len(predictions)):
    test_df.loc[i, 'label'] = ' '.join(str(int(i)) for i in np.argsort(pred)[::-1][:3])


# In[ ]:


test_df.head(15)


# In[ ]:


test_df.to_csv('submission_SE_Resnext101_fastai_mixup_2.csv', index=False)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# In[ ]:


create_download_link(test_df, filename='submission_SE_Resnext101_fastai_mixup_2.csv')


# In[ ]:




