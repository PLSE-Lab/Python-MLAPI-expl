#!/usr/bin/env python
# coding: utf-8

# **Please note that several cells have been commented out because cell 14, sometimes, displays an error. If you fork and run the notebook cell by cell, it will run normally.**

# **Create directory and upload urls file into your server
# **
# 

# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


classes = ['beaver', 'capybara', 'coypu']


# In[ ]:


folder = 'beaver'
file = 'beavers.csv'


# In[ ]:


path = Path('data/rodent')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# In[ ]:


get_ipython().system('cp ../input/* {path}/')


# In[ ]:


download_images(path/file, dest, max_pics=100)


# In[ ]:


folder = 'capybara'
file = 'capybaras.csv'


# In[ ]:


path = Path('data/rodent')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# In[ ]:


download_images(path/file, dest, max_pics=100)


# In[ ]:


folder = 'coypu'
file = 'coypus.csv'


# In[ ]:


path = Path('data/rodent')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# In[ ]:


download_images(path/file, dest, max_pics=100)


# In[ ]:


for c in classes:
     print(c)
     verify_images(path/c, delete=True, max_size=500)


# **View data**
# 

# In[ ]:


#np.random.seed(42)
#data = ImageDataBunch.from_folder(path, train="", valid_pct=0.2,
#         ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)


# In[ ]:


#data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


#data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


#learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[ ]:


#learn.fit_one_cycle(4)


# In[ ]:


#learn.save('stage-1')


# In[ ]:


#learn.unfreeze()


# In[ ]:


#learn.lr_find()


# In[ ]:


#learn.recorder.plot(suggestion=True)


# In[ ]:


#learn.fit_one_cycle(4, max_lr=slice(3e-6,3e-5))


# In[ ]:


#learn.save('stage-2')


# **Interpretation**

# In[ ]:


#learn.load('stage-2');


# In[ ]:


#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_top_losses(9,figsize=(12,12))


# In[ ]:


#interp.plot_confusion_matrix()


# **Clean up**

# In[ ]:


#from fastai.widgets import *


# In[ ]:


#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)


# In[ ]:


#ds, idxs = DatasetFormatter().from_similars(learn)


# In[ ]:


#ImageCleaner(ds, idxs, path, duplicates=True)


# **Make sure to recreate the ImageDataBunch and learn_cln from the cleaned.csv file!**

# In[ ]:


#np.random.seed(42)
#cleaned_data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
#ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)


# In[ ]:


#cleaned_data.classes, cleaned_data.c, len(cleaned_data.train_ds), len(cleaned_data.valid_ds)


# In[ ]:


#cleaned_data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


#learn_cln = cnn_learner(cleaned_data, models.resnet50, metrics=error_rate)


# In[ ]:


#learn.fit_one_cycle(4)


# In[ ]:


#learn.save('stage-3')


# **Production phase**

# In[ ]:


#img = open_image(path/'beaver'/'00000010.jpg')
#img


# In[ ]:


#classes = ['beaver', 'capybara', 'coypu']


# In[ ]:


#data2 = ImageDataBunch.single_from_classes(path, classes,
#                                  ds_tfms=get_transforms(),
#                                  size=224).normalize(imagenet_stats)


# In[ ]:


#learn = cnn_learner(data2, models.resnet50).load('stage-3')


# In[ ]:


#pred_class,pred_idx,outputs = learn.predict(img)
#pred_class

