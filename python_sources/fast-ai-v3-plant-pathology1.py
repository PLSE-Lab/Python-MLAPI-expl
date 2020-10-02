#!/usr/bin/env python
# coding: utf-8

# # Multi-label prediction with Plant Pathology dataset

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *

import os
import pandas as pd
import sys

from collections import Counter
from pathlib import Path
import numpy


# # Importing data

# In[ ]:


path = Path('/kaggle/input/plant-pathology-2020-fgvc7')
path.ls()


# In[ ]:


path2 = Path('/kaggle/input/plant-pathology-2020-fgvc7/images')
# path2.ls()


# In[ ]:


df = pd.read_csv(path/'train.csv')
df.head()


# In[ ]:


test_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageItemList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# # Data prep with API Blocks

# In[ ]:


#transform the data to get a bigger training set
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


# labels for our classes
LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']


# In[ ]:


#dataloading of testset
test = (ImageList.from_df(test_df,path,
                          folder='images',
                          suffix='.jpg',
                          cols='image_id'))


# In[ ]:


test


# In[ ]:


#making training set with API Block
np.random.seed(42)
src=(ImageList.from_csv(path,'train.csv',folder='images',suffix='.jpg')
    .split_by_rand_pct(0.2)
    .label_from_df(cols=LABEL_COLS,label_cls = MultiCategoryList))


# In[ ]:


data = (src.transform(tfms, size=253).add_test(test)
        .databunch(num_workers=0).normalize(imagenet_stats))


# In[ ]:


data.classes


# # Data viewing

# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# # Training model

# In[ ]:


arch = models.resnet50


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = create_cnn(data, arch, metrics=[acc_02, f_score], model_dir='/kaggle/working')


# We use the LR Finder to pick a good learning rate.

# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# Then we can fit the head of our network.

# In[ ]:


lr=1e-03
learn.fit_one_cycle(5,slice(lr))


# In[ ]:


learn.save('stage-1-rn50')


# ...And fine-tune the whole model:

# In[ ]:


learn.unfreeze()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, 1e-3))


# In[ ]:


learn.save('stage-2-rn50')


# Now we load the data again with a better image size, this improve the quality on the images 
# therefor it should be better at predicting.
# We freeze and unfreeze the model again becouse we treat it as if, it was a new model that, just have had some pretraining 

# In[ ]:


tfms = get_transforms()


# In[ ]:


data = (src.transform(tfms, size=399)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


lr=1e-3
learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.unfreeze()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-6, 1e-4))


# In[ ]:


learn.save('stage-2-256-rn50')


# In[ ]:


#plot losses to see how good the model trained
learn.recorder.plot_losses()


# # Submission

# In[ ]:


#Make predictions on test set
preds = learn.get_preds(DatasetType.Test)


# In[ ]:


test = pd.read_csv(path/'test.csv')
test_id = test['image_id'].values


# In[ ]:


submission = pd.DataFrame({'image_id': test_id})
submission = pd.concat([submission, pd.DataFrame(preds[0].numpy(), columns = LABEL_COLS)], axis=1)
submission.to_csv('submission_plant12.csv', index=False)
submission.head(10)
print('Model ready for submission!')

