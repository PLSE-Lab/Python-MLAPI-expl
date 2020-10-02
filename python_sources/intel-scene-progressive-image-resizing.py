#!/usr/bin/env python
# coding: utf-8

# In[ ]:


path = "../input/scene_classification/scene_classification/train/"


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


bs = 256


# In[ ]:


df = pd.read_csv('../input/scene_classification/scene_classification/train.csv')
df.head()


# In[ ]:


tfms = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0)


# In[ ]:


data_small = (ImageList.from_csv(path, csv_name = '../train.csv') 
        .split_by_rand_pct()              
        .label_from_df()            
        .add_test_folder(test_folder = '../test')              
        .transform(tfms, size=128)
        .databunch(num_workers=0))


# In[ ]:


data_large = (ImageList.from_csv(path, csv_name = '../train.csv') 
        .split_by_rand_pct()              
        .label_from_df()            
        .add_test_folder(test_folder = '../test')              
        .transform(tfms, size=256)
        .databunch(num_workers=0))


# In[ ]:


data_small.show_batch(rows=3, figsize=(8,10))


# In[ ]:


data_large.show_batch(rows=3, figsize=(8,10))


# In[ ]:


print(data_small.classes)
print(data_large.classes)


# In[ ]:


learn_34 = cnn_learner(data_small, models.resnet34, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn_50 = cnn_learner(data_small, models.resnet50, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn_101 = cnn_learner(data_small, models.resnet101, metrics=[error_rate, accuracy], model_dir="/tmp/model/")


# In[ ]:


learn_34.fit_one_cycle(4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn_34)

losses,idxs = interp.top_losses()

len(data_small.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(7,7), dpi=60)


# In[ ]:


learn_34.save('/kaggle/working/resnet34-size128-stage1')


# In[ ]:


learn_34.data = data_large


# In[ ]:


learn_34.unfreeze()


# In[ ]:


learn_34.lr_find()


# In[ ]:


learn_34.recorder.plot()


# In[ ]:


learn_34.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))


# In[ ]:


learn_34.save('/kaggle/working/resnet34-size256-stage1')


# **ResNet 50**

# In[ ]:


learn_50.fit_one_cycle(4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn_50)

losses,idxs = interp.top_losses()

len(data_small.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(7,7), dpi=60)


# In[ ]:


learn_50.save('/kaggle/working/resnet50-size128-stage1')


# In[ ]:


learn_50.data = data_large


# In[ ]:


learn_50.unfreeze()


# In[ ]:


learn_50.lr_find()


# In[ ]:


learn_50.recorder.plot()


# In[ ]:


learn_50.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))


# In[ ]:


learn_50.save('/kaggle/working/resnet50-size256-stage1')


# **ResNet 101**

# In[ ]:


learn_101.fit_one_cycle(4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn_101)

losses,idxs = interp.top_losses()

len(data_small.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(7,7), dpi=60)


# In[ ]:


learn_101.save('/kaggle/working/resnet101-size128-stage1')


# In[ ]:


learn_101.data = data_large


# In[ ]:


learn_101.unfreeze()


# In[ ]:


learn_101.lr_find()


# In[ ]:


learn_101.recorder.plot()


# In[ ]:


learn_101.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))


# In[ ]:


learn_101.save('/kaggle/working/resnet101-size256-stage1')

