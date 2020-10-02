#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# In[ ]:


path = "../input/"
tfms = tfms = get_transforms(do_flip=False)


# In[ ]:


print(os.listdir("../input/train/train")[0:5])


# In[ ]:


help(ImageDataBunch.from_csv)


# In[ ]:


data = ImageDataBunch.from_csv(path, folder = "train/train", csv_labels = "train.csv",
                               test = "../input/test/test", ds_tfms=tfms, size=224)


# In[ ]:


data


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# ### Training : resnet34

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")


# In[ ]:


learn.fit_one_cycle(1)
#learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# ### Results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# ### Predictions on Test Set

# #### Directly using get_preds

# In[ ]:


preds,_ = learn.get_preds(ds_type=DatasetType.Test)
preds[0]


# In[ ]:


test_df = pd.read_csv(path+"/sample_submission.csv")
test_df.head()


# In[ ]:


test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv("submission.csv", index=False)
test_df.head()


# #### Making predictions by iteration (WIP)

# In[ ]:


test_img = ImageList.from_df(test_df, path='../input/test/test')
test_img[0]
learn.predict(test_img[0])


# In[ ]:


test_predictions = []
for test_image in test_data:
    test_predictions.append(learn.predict(test_image)[0])
    
test_predictions[0:5]    

