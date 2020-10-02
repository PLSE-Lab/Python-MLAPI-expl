#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd

from fastai import *
from fastai.vision import *

from pathlib import Path
from PIL import Image


# In[ ]:


# !rm -rf /kaggle/output
# !mkdir /kaggle/output


# In[ ]:


input_root_path = Path("/kaggle/input/digit-recognizer")
output_root_path = Path("/kaggle/output")


# In[ ]:


train_df = pd.read_csv(input_root_path/'train.csv')
test_df = pd.read_csv(input_root_path/'test.csv')


# In[ ]:


train_df.head()


# ## Saving a source data to the fast.ai library format

# In[ ]:


def saveDigit(digit, filepath):
    digit = digit.reshape(28, 28)
    digit = digit.astype(np.uint8)

    img = Image.fromarray(digit)
    img.save(filepath)
    
def save_train_images(output_root_path: Path, df: pd.DataFrame, num_classes=10):
    output_data_path = output_root_path/'train'
    os.makedirs(output_data_path)
    for i in range(num_classes):
        os.makedirs(output_data_path/str(i))
    for index, row in df.iterrows():
        label,digit = row[0], row[1:]

        folder = output_data_path/str(label)
        filename = f"{index}.jpg"
        filepath = folder/filename

        digit = digit.values
        saveDigit(digit, filepath)
        
def save_test_images(output_root_path: Path, df: pd.DataFrame):
    output_data_path = output_root_path/'test'
    os.makedirs(output_data_path)
    for index, row in df.iterrows():
        filename = f"{index}.jpg"
        filepath = output_data_path/filename
        saveDigit(row.values, filepath)


# In[ ]:


save_train_images(output_root_path, train_df)
save_test_images(output_root_path, test_df)


# ## Loading train and test data

# In[ ]:


tfms = get_transforms(do_flip=False, max_rotate=20.0, max_warp=0.0)


# In[ ]:


data = ImageDataBunch.from_folder(
    path=output_root_path,
    train='train',
    test='test',
    seed=1234,
    #validation part
    valid_pct=0.2,
    #image size
    size=28,
    #data augumentation
    ds_tfms=tfms
).normalize()


# ## Examples of images 

# In[ ]:


data.show_batch(rows=3, figsize=(5, 5))


# In[ ]:


learn = cnn_learner(data, models.resnet152, metrics=accuracy, callback_fns=ShowGraph).mixup()


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


lrf=learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.save('model')
learn.load('model')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, 1e-03)


# ## Predicting values

# In[ ]:


class_score, _ = learn.get_preds(DatasetType.Test)
class_score = np.argmax(class_score, axis=1)


# In[ ]:


# remove file extension from filename
ImageId = [os.path.splitext(path)[0] for path in os.listdir(output_root_path/'test')]
# typecast to int so that file can be sorted by ImageId
ImageId = [int(path) for path in ImageId]
# +1 because index starts at 1 in the submission file
ImageId = [ID+1 for ID in ImageId]


# ## Saving a submission

# In[ ]:


pd.DataFrame({
    "ImageId": ImageId,
    "Label": class_score
}).to_csv("submission.csv", index=False)

