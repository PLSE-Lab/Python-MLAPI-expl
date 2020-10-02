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


df_sub = df[(df['label'] != 2) & (df['label'] != 5)]
df_sub['label'].value_counts()


# In[ ]:


tfms = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0)


# In[ ]:


allClasses = [0,1,2,3,4,5]


# In[ ]:


data_small_sub = (ImageList.from_df(df_sub,path) 
        .split_by_rand_pct()              
        .label_from_df(classes=allClasses)            
        .add_test_folder(test_folder = '../test')              
        .transform(tfms, size=128)
        .databunch(num_workers=0))


# In[ ]:


data_small_sub.show_batch(rows=3, figsize=(8,10))


# In[ ]:


data_large_full = (ImageList.from_df(df,path) 
        .split_by_rand_pct()              
        .label_from_df(classes=allClasses)            
        .add_test_folder(test_folder = '../test')              
        .transform(tfms, size=256)
        .databunch(num_workers=0))


# In[ ]:


data_large_full.show_batch(rows=3, figsize=(8,10))


# In[ ]:


print(data_small_sub.classes)
print(data_large_full.classes)


# In[ ]:


learn_101 = cnn_learner(data_small_sub, models.resnet101, metrics=[error_rate, accuracy], model_dir="/tmp/model/")


# In[ ]:


learn_101.fit_one_cycle(4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn_101)

losses,idxs = interp.top_losses()

len(data_small_sub.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(7,7), dpi=60)


# In[ ]:


learn_101.save('/kaggle/working/resnet101-size128-fewclasses-stage1')


# In[ ]:


learn_101.data = data_large_full


# In[ ]:


learn_101.unfreeze()


# In[ ]:


learn_101.lr_find()


# In[ ]:


learn_101.recorder.plot()


# In[ ]:


learn_101.fit_one_cycle(2, max_lr=slice(1e-5, 1e-4))


# In[ ]:


learn_101.save('/kaggle/working/resnet101-size256-allclasses-stage1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn_101)

losses,idxs = interp.top_losses()

len(data_large_full.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(7,7), dpi=60)

