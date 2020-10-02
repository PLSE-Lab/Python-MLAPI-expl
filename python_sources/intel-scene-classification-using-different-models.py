#!/usr/bin/env python
# coding: utf-8

# In[1]:


path = "../input/scene_classification/scene_classification/train/"


# In[2]:


from fastai import *
from fastai.vision import *


# In[3]:


bs = 256


# In[4]:


df = pd.read_csv('../input/scene_classification/scene_classification/train.csv')
df.head()


# In[5]:


tfms = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0)
data = (ImageList.from_csv(path, csv_name = '../train.csv') 
        .split_by_rand_pct()              
        .label_from_df()            
        .add_test_folder(test_folder = '../test')              
        .transform(tfms, size=256)
        .databunch(num_workers=0))


# In[6]:


data.show_batch(rows=3, figsize=(8,10))


# In[7]:


print(data.classes)


# In[8]:


learn_34 = cnn_learner(data, models.resnet34, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn_50 = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn_101 = cnn_learner(data, models.resnet101, metrics=[error_rate, accuracy], model_dir="/tmp/model/")


# **ResNet34**

# In[9]:


learn_34.fit_one_cycle(4)


# In[10]:


interp = ClassificationInterpretation.from_learner(learn_34)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[11]:


interp.plot_confusion_matrix(figsize=(7,7), dpi=60)


# In[13]:


learn_34.save('/kaggle/working/resnet34-stage1')


# In[15]:


learn_34.unfreeze()


# In[16]:


learn_34.lr_find()


# In[17]:


learn_34.recorder.plot()


# In[18]:


learn_34.fit_one_cycle(1, max_lr=slice(1e-6, 1e-4))


# In[19]:


learn_34.save('/kaggle/working/resnet34-stage2')


# **ResNet 50**

# In[20]:


learn_50.fit_one_cycle(4)


# In[21]:


interp = ClassificationInterpretation.from_learner(learn_50)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[22]:


interp.plot_confusion_matrix(figsize=(7,7), dpi=60)


# In[23]:


learn_50.save('/kaggle/working/resnet50-stage1')


# In[24]:


learn_50.unfreeze()


# In[25]:


learn_50.lr_find()


# In[26]:


learn_50.recorder.plot()


# In[27]:


learn_50.fit_one_cycle(1, max_lr=slice(1e-6, 1e-4))


# In[28]:


learn_50.save('/kaggle/working/resnet50-stage2')


# **ResNet 101**

# In[29]:


learn_101.fit_one_cycle(4)


# In[30]:


interp = ClassificationInterpretation.from_learner(learn_101)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[31]:


interp.plot_confusion_matrix(figsize=(7,7), dpi=60)


# In[32]:


learn_101.save('/kaggle/working/resnet101-stage1')


# In[33]:


learn_101.unfreeze()


# In[34]:


learn_101.lr_find()


# In[35]:


learn_101.recorder.plot()


# In[36]:


learn_101.fit_one_cycle(1, max_lr=slice(1e-6, 1e-4))


# In[37]:


learn_101.save('/kaggle/working/resnet101-stage2')

