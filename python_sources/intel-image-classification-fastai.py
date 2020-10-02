#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# In[ ]:


path = '/kaggle/input/intel-image-classification'


# In[ ]:


import os
print(os.listdir(path))


# In[ ]:


# Data augmentation
tfms = get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


src = (ImageList.from_folder(path)
                .split_by_folder(train='seg_train', valid='seg_test')
                .label_from_folder()
                .add_test_folder(test_folder = 'seg_pred'))


# In[ ]:


# Can achieve better result by first training with smaller size like 75x75 first before training on full size
data = src.transform(tfms, size=150).databunch().normalize()


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


arch = models.resnet50
metrics = [error_rate, accuracy]


# In[ ]:


learn = cnn_learner(data, arch, metrics=metrics)


# In[ ]:


learn.model_dir = '/kaggle/working/'


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1-rn50')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


# Can train a little bit more for better result
learn.fit_one_cycle(5, slice(5e-5, 5e-4))


# In[ ]:


learn.save('stage-2-rn50')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


# Interpret the result
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(12,9))


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


# Plot the confusion matrix
interp.plot_confusion_matrix()


# In[ ]:


# Make predictions of the test folder 
predictions, targets = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


# Make predictions on the first 9 images
classes = predictions.argmax(1)
class_dict = dict(enumerate(learn.data.classes))
labels = [class_dict[i] for i in list(classes[:9].tolist())]
test_images = [i.name for i in learn.data.test_ds.items][:9]


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

for i, fn in enumerate(test_images):
    img = plt.imread(path + '/seg_pred/seg_pred/' + fn, 0)
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(labels[i])
    plt.axis("off")


# **As we can see, the predictions made were pretty good!!! :))**
