#!/usr/bin/env python
# coding: utf-8

# Load libraries

# In[ ]:


import numpy as np
import pandas as pd
from fastai.vision import *


# Look at dataset

# In[ ]:


path = Path('../input')


# In[ ]:


path.ls()


# In[ ]:


get_image_files(path/'train-jpg')[:5]


# In[ ]:


df = pd.read_csv(path/'train_v2.csv')
df.head()


# Create datablock

# In[ ]:


np.random.seed(42)
size = 224
bs = 64
num_workers = 0  # set this to 2 to prevent kernel from crashing


# In[ ]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


src = (ImageItemList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
       .random_split_by_pct()
       .label_from_df(sep=' ')
       .add_test_folder('test-jpg-v2'))


# In[ ]:


data = (src.transform(tfms, size=size)
        .databunch(bs=bs, num_workers=num_workers)
        .normalize(imagenet_stats))


# Verify datasets loaded properly.  We should have the following:
# * train: 32,384
# * valid: 8,095
# * test: 61,191

# In[ ]:


print(len(data.train_ds))
print(len(data.valid_ds))
print(len(data.test_ds))


# In[ ]:


data.classes


# Visualize data

# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# 

# Define model architecture and metrics

# In[ ]:


arch = models.resnet50
acc = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)


# Create learner

# In[ ]:


learn = create_cnn(data, arch, metrics=[acc, f_score], model_dir='/tmp/models')


# Use learning rate finder to pick a good learning rate

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-2


# Fit the head of the network

# In[ ]:


learn.fit_one_cycle(4, slice(lr))


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.recorder.plot_losses()


# Unfreeze all layers, run learning rate finder, fit some more

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.recorder.plot_losses()


# Create predictions using test set

# In[ ]:


preds, y = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


preds[:5]


# In[ ]:


thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[ ]:


labelled_preds[:5]


# Generate competition submission using predictions

# In[ ]:


submission = pd.DataFrame({'image_name':os.listdir('../input/test-jpg-v2'), 'tags':labelled_preds})


# In[ ]:


submission['image_name'] = submission['image_name'].map(lambda x: x.split('.')[0])


# In[ ]:


submission = submission.sort_values('image_name')


# In[ ]:


submission[:5]


# In[ ]:


submission.to_csv('submission.csv', index=False)

