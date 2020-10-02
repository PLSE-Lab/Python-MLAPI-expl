#!/usr/bin/env python
# coding: utf-8

# # Multiple-label Classification with Planet Amazon Dataset
# 
# `fastai` package will be used for this work.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
from fastai.vision import *


# In[ ]:


path = Path("../input/")
path.ls()


# In[ ]:


# look at the image-label file
df = pd.read_csv(path/"train_v2.csv")
df.head()


# The "tags" column consists of the labels for each training images. Each tags consists of multi-labels which describe the weather (clear, cloudy, partly_cloudy, haze, etc.) and the geographical substances (primary forest, agriculture, road, water, habitation, etc.) that can be found in each of the Amazon's satellite images.

# ## DataBunch Preparation
# 
# "[data block API](https://docs.fast.ai/data_block.html)" procedure will be used to convert the raw images into model-readable DataBunch object.

# In[ ]:


# image transformation
tfms = get_transforms(flip_vert = True, max_lighting = 0.1, max_zoom = 1.05, max_warp = 0.)


# In[ ]:


# create ImageList
np.random.seed(7)
source = (ImageList.from_csv(path, "train_v2.csv", folder = "train-jpg", suffix = ".jpg")
         .split_by_rand_pct(0.2)
         .label_from_df(cols = "tags", label_delim = " "))
# label_delim to separate the words in "tags" column so as to generate multiple labels 


# I will create data with rundown resolution first (128x128) so that I can fit the model faster and see how that model performs with lower quality images, as a baseline benchmark. (default image size provided by Kaggle is 256x256)

# In[ ]:


# data with size 128 (default 256)
data = (source.transform(tfms, size = 128)
       .databunch()
       .normalize(imagenet_stats))


# In[ ]:


# show the data
data.show_batch(rows = 4, figsize = (15,15))


# What an interesting Amazon's satellite images!

# ## Multiclassification
# 
# Pre-trained ResNet50 CNN will be used to fit the data. As for the metrics, `accuracy_thresh` will be used instead of `accuracy` because this is a multiclassification problem, the model should return multiple labels for each images as long as the probabilities of those labels are above a certain threshold. Apart from that, **F2-score** will also be used since it is the metric used by Kaggle in this competition. `fbeta` function from `fastai` will generates this metric.

# In[ ]:


# metrics
acc_thresh = partial(accuracy_thresh, thresh = 0.2) # choose threshold = 0.2
f2_score = partial(fbeta, beta = 2, thresh = 0.2) # fbeta where beta = 2 (F2) and threshold = 0.2


# In[ ]:


# download model
learn = cnn_learner(data, models.resnet50, metrics = [acc_thresh, f2_score], model_dir = "/tmp/models")


# In[ ]:


# find good learning rate
learn.lr_find()
learn.recorder.plot()


# The loss-line drops steepest around 0.01 (1e-02), so this will be used as the learning rate.

# In[ ]:


# baseline model with image size=128
learn.fit_one_cycle(cyc_len = 5, max_lr = slice(1e-2))


# This baseline model achieves an accuracy of 95.6% and F2-score of 0.924 which is pretty good.

# In[ ]:


# save this model
learn.save("baseline-rn50-128")


# ### Model Fine-tuning
# 
# I will now fit the ResNet50 CNN with original image size, expect it to perform better.

# In[ ]:


# create 2nd data with original size
np.random.seed(7)
data2 = (source.transform(tfms, size = 256)
        .databunch()
        .normalize(imagenet_stats))


# In[ ]:


# create another CNN model
learn2 = cnn_learner(data2, models.resnet50, metrics = [acc_thresh, f2_score], model_dir = "/tmp/models")


# In[ ]:


# plot the learning rate of this model
learn2.lr_find()
learn2.recorder.plot()


# Same as above, the loss-line drops steepest around 0.03 (3e-02) and I will use that as learning rate.

# In[ ]:


# baseline model with image size = 256
lr = 3e-2
learn2.fit_one_cycle(cyc_len = 5, max_lr = slice(lr))


# In[ ]:


# save baseline model for size=256
learn2.save("baseline-rn50-256")


# In[ ]:


# plot the learning rate
learn2.unfreeze()
learn2.lr_find()
learn2.recorder.plot()


# 0.0001 (1e-04) seems to be a good turning point before the loss-line shoots up. I will use 0.0001/10 = 0.00001 as one of the learning rate.

# In[ ]:


# model 2 for image size = 256
learn2.fit_one_cycle(cyc_len = 10, max_lr = slice(1e-5, lr/10))


# In[ ]:


# plot the training and validation loss of the model
learn2.recorder.plot_losses()


# In[ ]:


# save the unfreezed model
learn2.save("stage-2-rn50-256")


# The model is now ready for inference, `learn.export` is called to save all the information of the Learner object for inference: the stuff we need in the DataBunch (transforms, classes, normalization...), the model with its weights and all the callbacks the Learner was using.

# In[ ]:


# export the model
learn2.export(file = "../working/export.pkl")


# ## Predictions and Submission
# 
# There are 2 set of images for testing provided by Kaggle, in the folders `test-jpg.tar.7z` and `test-jpg-additional.tar.7z`. All of these test images are compiled in the folder `test-jpg-v2` in the path. 

# In[ ]:


#test = ImageList.from_folder(path/"test-jpg").add(ImageList.from_folder(path/"test-jpg-additional"))
test = ImageList.from_folder(path/"test-jpg-v2")
len(test)


# In[ ]:


load_path = Path("../working/")

learn = load_learner(load_path, test=test)
predicts, _ = learn.get_preds(ds_type = DatasetType.Test)


# In[ ]:


# pick the labels as long as the probability is more than 0.2
labels_pred = [" ".join([learn.data.classes[i] for i,p in enumerate(pred) if p > 0.2]) for pred in predicts]

labels_pred[:5]


# In[ ]:


for img in learn.data.test_ds.items[:10]:
    print(img.name)


# In[ ]:


# pick up the images' names
image_names = [img.name[:-4] for img in learn.data.test_ds.items] # img.name[:-4] because I want to remove '.jpg' from the name


# In[ ]:


# create the dataframe of images' names and their tags (the format we have seen in train_v2.csv)
df2 = pd.DataFrame({"image_name":image_names, "tags":labels_pred})
df2.head()


# In[ ]:


# create the csv file for submission
df2.to_csv("submission.csv", index = False)

