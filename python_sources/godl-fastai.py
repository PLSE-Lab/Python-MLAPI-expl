#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision import *
from fastai.callbacks import *


# In[2]:


from pathlib import Path


# In[3]:


bs = 64


# # Import data 

# In[4]:


path = Path("../input/gameofdl/")
train_path = path/'train.csv'
test_path = path/'test.csv'
train_path, test_path


# In[5]:


np.random.seed(42)


# In[6]:


data = ImageDataBunch.from_csv(path=path, 
                               folder='train', 
                               csv_labels='train.csv', 
                               test='test', 
                               ds_tfms=get_transforms(), 
                               size=224, 
                               bs=bs).normalize(imagenet_stats)


# In[7]:


data.path = pathlib.Path('.')


# In[8]:


get_transforms()


# In[9]:


#data.show_batch()


# In[10]:


data.show_batch(rows=3, figsize=(7,6))


# In[11]:


print(data.classes)
len(data.classes),data.c


# # Training: resnet34

# In[12]:


learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy])


# In[13]:


learn.fit_one_cycle(4)


# In[14]:


learn.save('stage-1')


# In[15]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[16]:


interp.plot_top_losses(9, figsize=(15,11))


# In[17]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[18]:


interp.most_confused(min_val=10)


# In[19]:


learn.unfreeze()


# In[20]:


learn.fit_one_cycle(1)


# In[21]:


learn.load('stage-1');


# In[22]:


learn.lr_find()


# In[23]:


learn.recorder.plot()


# In[24]:


learn.load('stage-1');
learn.unfreeze()
learn.fit_one_cycle(2)


# # Training: resnet50

# In[25]:


learn50 = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[26]:


learn50.lr_find()
learn50.recorder.plot()


# In[27]:


learn50.fit_one_cycle(8)


# In[28]:


learn50.save('stage-1-50')


# In[29]:


learn50.load('stage-1-50')
learn50.unfreeze()
learn50.fit_one_cycle(5, max_lr=slice(1e-4,1e-2))


# In[33]:


learn50.save('stage-2-50')


# # Increasing Resolutions

# In[ ]:


sizes = [32, 64, 128, 224]


# In[30]:


def get_data(sz, bs):
    data = ImageDataBunch.from_csv(path=path, 
                                   folder='train', 
                                   csv_labels='train.csv', 
                                   test='test', 
                                   ds_tfms=get_transforms(), 
                                   size=sz, 
                                   bs=bs).normalize(imagenet_stats)
    data.path = pathlib.Path('.')
    return data


# ## Using Default LR

# In[ ]:


learn50 = cnn_learner(get_data(8, int(2048/8)), 
                      models.resnet50, 
                      metrics=error_rate)
#learn50.save('res50_0')
learn50.save('res50_8')

learn50 = cnn_learner(get_data(16, int(2048/16)), 
                      models.resnet50, 
                      metrics=error_rate).load('res50_8')
learn50.save('res50_16')

learn50 = cnn_learner(get_data(24, int(2048/24)), 
                      models.resnet50, 
                      metrics=error_rate).load('res50_16')
learn50.save('res50_24')

learn50 = cnn_learner(get_data(32, int(2048/32)), 
                      models.resnet50, 
                      metrics=error_rate).load('res50_24')
learn50.save('res50_32')

learn50 = cnn_learner(get_data(64, int(2048/64)), 
                      models.resnet50, 
                      metrics=error_rate).load('res50_32')
learn50.save('res50_64')

learn50 = cnn_learner(get_data(128, int(2048/128)), 
                      models.resnet50, 
                      metrics=error_rate).load('res50_64')
learn50.save('res50_128')

learn50 = cnn_learner(get_data(224, int(2048/224)), 
                      models.resnet50, 
                      metrics=error_rate).load('res50_128')
learn50.save('res50_224')


# In[31]:


def train_model(sz, i):
    learn50 = cnn_learner(get_data(sz, int(2048/sz)), 
                          models.resnet50, 
                          metrics=[error_rate, accuracy]).load('res50_'+str(sz-8))
    learn50.fit_one_cycle(6*i)
    learn50.lr_find()
    learn50.recorder.plot()
    learn50.unfreeze()
    learn50.fit_one_cycle(2*i)
    learn50.save('res50_'+str(sz))


# In[ ]:


train_model(8, 1)


# In[ ]:


train_model(16, 2)


# In[ ]:


train_model(24, 3)


# In[ ]:


train_model(32, 4)


# In[ ]:


sz = 64; i = 5
learn50 = cnn_learner(get_data(sz, int(2048/sz)), 
                      models.resnet50, 
                      metrics=[error_rate, accuracy]).load('res50_32')
learn50.fit_one_cycle(6*i)
learn50.lr_find()
learn50.recorder.plot()
learn50.unfreeze()
learn50.fit_one_cycle(2*i)
learn50.save('res50_'+str(sz))


# In[ ]:


sz = 128; i = 6
learn50 = cnn_learner(get_data(sz, bs), 
                      models.resnet50, 
                      metrics=[error_rate, accuracy], 
                      callbacks=[SaveModelCallback(learn50, every='improvement', monitor='accuracy', name='best_128')]).load('res50_64')
learn50.fit_one_cycle(6*i)


# In[ ]:


learn50.unfreeze()
learn50.fit_one_cycle(2*i)
learn50.save('res50_'+str(sz))


# In[35]:


sz = 224; i = 7
learn50 = cnn_learner(get_data(sz, bs),  
                      models.resnet50, 
                      metrics=[error_rate, accuracy], 
                      callbacks=[SaveModelCallback(learn50, every='improvement', monitor='accuracy', name='best_224')]).load('stage-2-50')
                      #callbacks=[SaveModelCallback(learn50, every='improvement', monitor='accuracy', name='best_224')]).load('res50_128')  

learn50.fit_one_cycle(6*i)


# In[36]:


sz = 224; i = 7
learn50.load('best_224')
learn50.fit_one_cycle(2*i)
learn50.save('res50_'+str(sz))


# ## Mixup

# In[37]:


learn50 = cnn_learner(get_data(224, 64), 
                      models.resnet50, 
                      metrics=[error_rate, accuracy], 
                      callbacks=[SaveModelCallback(learn50, every='improvement', monitor='accuracy', name='best_224_mixup')]).load('res50_224').mixup()


# In[38]:


learn50.fit(8)


# In[39]:


learn50.save('mixup_8')
learn50.fit(5)
learn50.save('mixup_5')


# In[40]:


#learn50.unfreeze()
#learn50.fit_one_cycle(5)


# ## Submission

# In[41]:


import numpy as np
import pandas as pd


# In[42]:


learn50.load('mixup_5')
log_preds, test_labels = learn50.get_preds(ds_type=DatasetType.Test)


# In[43]:


preds = np.argmax(log_preds, 1)
preds_classes = [data.classes[i] for i in preds]


# In[44]:


a = np.array(preds)
data.test_ds.x[0]


# In[45]:


test_df = pd.DataFrame({ 'image': os.listdir('../input/gameofdl/test/test_images/'), 'category': preds_classes})
test_df.head()


# In[ ]:


#test_df.sort_values(by='image').reset_index(drop=True)


# In[46]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

test_df['category'] = pd.DataFrame(data=preds_classes)

# create a link to download the dataframe
create_download_link(test_df)


# In[ ]:




