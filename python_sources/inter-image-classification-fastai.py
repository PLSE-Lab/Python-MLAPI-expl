#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('mkdir ./images/')
get_ipython().system('cp -a ../input/seg_train/seg_train ./images/train')
get_ipython().system('cp -a ../input/seg_test/seg_test ./images/valid')
get_ipython().system('cp -a ../input/seg_pred/seg_pred ./images/test')


# In[34]:


from fastai import *
from fastai.vision import *

import matplotlib.pyplot as plt


# In[ ]:


data = ImageDataBunch.from_folder(path="./images/", train="train", valid="valid", test="test", ds_tfms=get_transforms(), size=224, bs=64)
data.normalize(imagenet_stats)


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=[accuracy])


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10, 1e-02)


# In[ ]:


learn.recorder.plot_losses()


# In[47]:


learn.recorder.plot_metrics()


# In[ ]:


predictions, targets = learn.get_preds(ds_type=DatasetType.Test)


# In[40]:


classes = predictions.argmax(1)
class_dict = dict(enumerate(learn.data.classes))
labels = [class_dict[i] for i in list(classes[:9].tolist())]
test_images = [i.name for i in learn.data.test_ds.items][:9]


# In[46]:


plt.figure(figsize=(10,8))

for i, fn in enumerate(test_images):
    img = plt.imread("./images/test/" + fn, 0)
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(labels[i])
    plt.axis("off")


# In[33]:


get_ipython().system('rm -rf ./images/')


# In[ ]:




