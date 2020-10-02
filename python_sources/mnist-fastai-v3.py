#!/usr/bin/env python
# coding: utf-8

# # CNN for MMIST

# A great thanks to [Sanwal Yousaf](https://www.kaggle.com/sanwal092), of which I got more of the fundamental code like making the directories. And some of the code is commented '#' because it would take longer to submit the notebook if it werent.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# FOR non-FASTAI LIBRARIES
import numpy as np 
import pandas as pd
import os
import random

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from fastai.vision import *
from fastai.metrics import *


# make sure CUDA is available and enabled
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)


# # Importing data

# In[ ]:


path = Path("../input/digit-recognizer")
os.listdir(path)


# In[ ]:


train_df = pd.read_csv(path/"train.csv")
train_df.head()


# In[ ]:


test_df = pd.read_csv(path/"test.csv")
test_df.head()


# In[ ]:


TRAIN = Path("../train")
TEST = Path("../test")


# # Make directories

# In[ ]:


#make directories folders for train
for i in range(10):    
    try:         
        os.makedirs(TRAIN/str(i))       
    except:
        pass

#see directory
print(os.listdir(TRAIN))


# In[ ]:


#make directories folders for test
try:
    os.makedirs(TEST)
except:
    pass

#see directory
print(os.listdir(TEST))


# In[ ]:


# os.listdir(TEST)
if os.path.isdir(TRAIN):
    print('Train directory has been created')
else:
    print('Train directory creation failed.')

if os.path.isdir(TEST):
    print('Test directory has been created')
else:
    print('Test directory creation failed.')


# ### Since FastAI only takes data in as images, not pixel values, we will have to convert this data into images for which we will use the PIL library. 
# 
# We will have to reshape this into 28x28 matrices. To do this, I will use the PIL library in Python 

# In[ ]:


from PIL import Image


# In[ ]:


def pix2img(pix_data, filepath):
    img_mat = pix_data.reshape(28,28)
    img_mat = img_mat.astype(np.uint8())
    
    img_dat = Image.fromarray(img_mat)
    img_dat.save(filepath)


# In[ ]:


# save training images
for idx, data in train_df.iterrows():
    
    label, data = data[0], data[1:]
    folder = TRAIN/str(label)
    
    fname = f"{idx}.jpg"
    filepath = folder/fname
    
    img_data = data.values
    
    pix2img(img_data,filepath)


# In[ ]:


# save test images
for idx, data in test_df.iterrows():
    folder = TEST
    
    fname = f"{idx}.jpg"
    filepath = folder/fname
    
    img_data = data.values
    
    pix2img(img_data,filepath)


# # Prepare train/test set

# In[ ]:


tfms = get_transforms(do_flip = False)


# In[ ]:


print('test : ',TEST)
print('train: ', TRAIN)
print(type(TEST))


# In[ ]:


path = ("../train")
# test = ("../test")


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", test = ("../test"), valid_pct=0.2,
        ds_tfms=get_transforms(), size=28, num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


mnist_stats


# # Train model

# In[ ]:


learn = cnn_learner(data, base_arch = models.resnet34, metrics = accuracy,model_dir="/tmp/models", callback_fns=ShowGraph )


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, 1e-03)


# In[ ]:


learn.save('model1')


# In[ ]:


learn.unfreeze()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5 , 1e-04)


# In[ ]:


learn.fit_one_cycle(5 , slice(1e-05,1e-04))


# In[ ]:


learn.fit_one_cycle(5 , slice(1e-06,1e-05))


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", test = ("../test"), valid_pct=0.2,
        ds_tfms=get_transforms(), size=69, num_workers=0).normalize(imagenet_stats)

learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


lr=1e-03


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-4')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.fit_one_cycle(10, 1e-05, wd=0.5)


# In[ ]:


learn.show_results(3, figsize= (7,7))


# # Prediction

# In[ ]:


class_score , y = learn.get_preds(DatasetType.Test)


# In[ ]:


probabilities = class_score[0].tolist()
[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]


# In[ ]:


class_score = np.argmax(class_score, axis=1)


# In[ ]:


class_score[1].item()


# # Submission

# In[ ]:


import pandas as pd
sample_submission =  pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sample_submission.head()


# In[ ]:


# remove file extension from filename
ImageId = [os.path.splitext(path)[0] for path in os.listdir(TEST)]
# typecast to int so that file can be sorted by ImageId
ImageId = [int(path) for path in ImageId]
# +1 because index starts at 1 in the submission file
ImageId = [ID+1 for ID in ImageId]


# In[ ]:


submission  = pd.DataFrame({
    "ImageId": ImageId,
    "Label": class_score
})

submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head()

