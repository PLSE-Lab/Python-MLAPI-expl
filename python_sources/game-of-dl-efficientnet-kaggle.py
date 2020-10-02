#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q efficientnet_pytorch')


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


from fastai import *
from fastai.utils import *
from fastai.vision import *
from fastai.callbacks import *
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import PIL
from torch.utils import model_zoo

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from efficientnet_pytorch import EfficientNet


# In[5]:


import warnings
warnings.filterwarnings("ignore")


# In[6]:


import os
print(os.listdir('.'))


# In[7]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# In[8]:


print('Make sure cuda is installed:', torch.cuda.is_available())
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)


# In[9]:


print(os.listdir('../input'))


# In[10]:


hack_path = Path('../input')


# In[11]:


# Load train dataframe
train_df = pd.read_csv(hack_path/'train/train.csv')
test_df = pd.read_csv(hack_path/'test_ApKoW4T.csv')
sample = pd.read_csv(hack_path/'sample_submission_ns2btKE.csv')


# In[12]:


def get_data(bs, size):
    data = ImageDataBunch.from_df(df=train_df, path=hack_path/'train', folder='images',
                                  bs=bs, size=size, valid_pct=0.1, 
                                  resize_method=ResizeMethod.SQUISH, 
                                  ds_tfms=get_transforms(max_lighting=0.4, max_zoom=1.2, 
                                                         max_warp=0.2, max_rotate=20, 
                                                         xtra_tfms=[flip_lr()]))
    test_data = ImageList.from_df(test_df, path=hack_path/'train', folder='images')
    data.add_test(test_data)
    data.normalize(imagenet_stats)
    return data


# In[13]:


data = get_data(bs=48, size=224)


# In[14]:


data.show_batch(rows=3, figsize=(10,8))


# ### Efficientnet-B3

# In[15]:


model_name = 'efficientnet-b3'


# In[16]:


def get_model(pretrained=True, **kwargs):
    model = EfficientNet.from_pretrained(model_name)
    model._fc = nn.Linear(model._fc.in_features, data.c)
    return model


# In[17]:


learn = Learner(data, get_model(), 
                metrics=[FBeta(), accuracy],
                callback_fns=[partial(SaveModelCallback)],
                wd=0.1,
                path = '.').mixup()


# In[18]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[19]:


min_grad_lr = learn.recorder.min_grad_lr
min_grad_lr


# In[20]:


learn.fit_one_cycle(20, min_grad_lr)


# In[21]:


learn.recorder.plot_losses()


# In[22]:


learn.recorder.plot_lr(show_moms=True)


# In[23]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[24]:


min_grad_lr = learn.recorder.min_grad_lr
min_grad_lr


# In[25]:


learn.fit_one_cycle(20, slice(min_grad_lr))


# In[26]:


unfrozen_validation = learn.validate()
print("Final model validation loss: {0}".format(unfrozen_validation[0]))


# In[27]:


learn.save('efficientnet-unfrozen', return_path=True)


# In[28]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[29]:


# interp.plot_top_losses(15, figsize=(15,11))


# In[30]:


interp.plot_confusion_matrix(figsize=(6,6), dpi=60)


# In[31]:


interp.most_confused(min_val=2)


# In[32]:


probability, classification = learn.TTA(ds_type=DatasetType.Test)


# In[33]:


probability.argmax(dim=1)[:10]


# In[34]:


(probability.argmax(dim=1) + 1).unique()


# In[35]:


sample.category = probability.argmax(dim=1) + 1


# In[36]:


sample.category.value_counts()


# In[37]:


sample.head()


# In[38]:


sample.to_csv('submission_efficientnetb3_kaggle.csv', index=False)


# In[39]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# create a link to download the dataframe
create_download_link(sample)

