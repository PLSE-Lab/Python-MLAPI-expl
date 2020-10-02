#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *


# In[ ]:


verbose = 1


# In[ ]:


import os
if verbose >= 1:
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


# In[ ]:


input_path = '../input/planet-understanding-the-amazon-from-space/'
path = Config.data_path()/'planet'
path.mkdir(parents=True, exist_ok=True)
if verbose >= 1:
    print(path)


# In[ ]:


# Files already exist here, so not need to untar
if verbose >= 2:
    get_ipython().system(' ls -al {input_path}/train-jpg')


# In[ ]:


if verbose >= 2:
    get_ipython().system(' ls -al /tmp/.fastai/data/planet/')


# In[ ]:


# Copying Data to a writable path since DataBunch will be created from here and the model will be saved here (hence needs write permisions)
# Will take some time
import time
start_time = time.time()
get_ipython().system('cp -r {input_path}train_v2.csv {path}/.')
get_ipython().system('cp -r {input_path}train-jpg {path}/.')
end_time = time.time()
print("Time Taken: {}".format(end_time - start_time))


# In[ ]:


df_train = pd.read_csv(path/'train_v2.csv')
df_train.head()


# ## Create Databunch

# ### <font color = red> Dataset (from PyTorch) --> DataLoader (from PyTorch) --> DataBunch (from fastai) </font>
# * **Dataset - has `__item__` and `__len__` to index the data**
# * **DataLoader - used to create mini-batches**
# * **DataBunch - Combination of DataLoaders for Train, Validation and Test sets**
# 
# 
# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.
# 
# **The data block API lets you customize the creation of a DataBunch by isolating the underlying parts of that process in separate blocks, mainly:**
# 1. Where are the inputs and how to create them?
# 2. How to split the data into a training and validation sets?
# 3. How to label the inputs?
# 4. What transforms to apply?
# 5. How to add a test set?
# 6. How to wrap in dataloaders and create the DataBunch?
# 
# 

# ### <font color = red>Warp is set to 0. Warp is used to mimic taking images from different angles (maybe you are higher than the 'cat' or cat is at an elevation and you are takikng image from below). For Satellite images, this is not the case so we set warp to 0. </font>

# In[ ]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


if verbose >= 2:
    doc(ImageList.from_csv)


# In[ ]:


np.random.seed(42)
src = (ImageList.from_csv(path=path, csv_name='train_v2.csv', folder='train-jpg', suffix='.jpg')  # Where to find the data? -> in path/'train-jpg' folder
       .split_by_rand_pct(0.2) # How to split in train/valid? -> randomly with the default 20% in valid
       .label_from_df(label_delim=' ')) # How to label? -> use the second column of the csv file and split the tags by ' '


# In[ ]:


data = (src.transform(tfms, size=128) # Data augmentation? -> use tfms with a size of 128
        .databunch().normalize(imagenet_stats)) # Finally -> use the defaults for conversion to databunch


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# ## Create Learner

# To create a `Learner` we use the same function as in lesson 1. Our base architecture is resnet50 again, but the metrics are a little bit differeent: we use `accuracy_thresh` instead of `accuracy`. In lesson 1, we determined the predicition for a given class by picking the final activation that was the biggest, but here, each activation can be 0. or 1. `accuracy_thresh` selects the ones that are above a certain threshold (0.5 by default) and compares them to the ground truth.
# 
# As for Fbeta, it's the metric that was used by Kaggle on this competition. See [here](https://en.wikipedia.org/wiki/F1_score) for more details.

# In[ ]:


arch = models.resnet50


# In[ ]:


# Kaggle comes with internet off. So have to copy over model to the location where fastai would have downloaded it.
# https://forums.fast.ai/t/how-can-i-load-a-pretrained-model-on-kaggle-using-fastai/13941/23

get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints')
get_ipython().system('cp ../input/resnet50/resnet50.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth')


# ### <font color = red>**Changing the metric will not change the way the model is trained. This is just for printing progress.**</font>

# In[ ]:


# We are looking for multiple labels here, so we look for anything with prob > thresh (you decide what the threshold to use) 
# Can be achieved by creating a partial function --> Create something like the other function with some arguments fixed to certain values
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2) 
learn = cnn_learner(data, arch, metrics=[acc_02, f_score])


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# ### <font color = red>Before you unfreeze, you will almost always get this shape. **Find the steepest slope (not the bottom) and use that as your learning rate.** </font>

# In[ ]:


lr = 0.01


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-1-rn50')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# ### <font color = red>RULE OF THUMB:</font>
# * **End value is about 1 order of magnitude lower than the one in stage1 (or divide by 5)**
# * **Start value should come from the learning rate finder and should be well below the value at which the loss starts to become worse. <font color = red>Look for the strongest downward slope that is sticking around for a while</font>. Pick that as the lower end of the learning rate (used for the earlier layers). <font color = red>If there is no downward trend like in the plot above, then pick about 1 order of magnitude before it starts to get worse**</font>

# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))  # lr is 0.01, so lr/5 is 0.002


# In[ ]:


learn.save('stage-2-rn50')


# ## With Larger Image Size
# 
# ### <font color = red> Kaggle Images are 256x256, but we used 128x128 to get a decent model. Now, we can use this model and transfer learning to train a better model with the 256x256 images. </font>
# 
# 

# In[ ]:


data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

# Start with the same learner, just replace the data with the new dataset of 256x256 inputs
learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-2/2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1-256-rn50')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-2-256-rn50')


# In[ ]:


learn.export()


# ## Predictions

# In[ ]:


get_ipython().system(' ls ../input/planet-understanding-the-amazon-from-space/')


# In[ ]:


# Will take some time
import time
start_time = time.time()
get_ipython().system('cp -r {input_path}test-jpg-v2 {path}/.')
end_time = time.time()
print("Time Taken: {}".format(end_time - start_time))


# In[ ]:


test = ImageList.from_folder(path/'test-jpg-v2') #.add(ImageList.from_folder(path/'test-jpg-additional'))
len(test)


# In[ ]:


learn = load_learner(path, test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[ ]:


labelled_preds[:5]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]


# In[ ]:


df = pd.DataFrame({'image_name':fnames,
                   'tags':labelled_preds},
                  columns=['image_name', 'tags'])


# In[ ]:


df.to_csv('submission.csv', index=False)


# In[ ]:


get_ipython().system(' ls ../working')


# In[ ]:




