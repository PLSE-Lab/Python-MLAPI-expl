#!/usr/bin/env python
# coding: utf-8

# # Digit Recognizer with fast.ai v3
# 
# ## Introduction
# This is just the regular MNIST done using fast.ai V3
# 
# Since the library doesn't have an easy way to deal with the input format here (arrays in CSV), we'll have to go through a roundabout route and create the images so that fast.ai will have an easier time reading it through it's Data Block API
# 
# ## Process
# 1. Convert the CSV to vision.image.Image
# 2. Create a Databunch using the custom list, TensorImageList
# 3. Train the model
# 4. Predict the outcome
# 
# ## Note
# Please forgive me if this notebook isn't optimized. This is my first one ever in Kaggle and I'm only starting to get used to the environment.

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import *
from fastai.vision import *

import torchvision

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from time import time
import os
print(os.listdir("../input"))

from IPython.display import HTML
import base64

import PIL

# Any results you write to the current directory are saved as output.


# In[4]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


def imagify(tensor):
    reshaped = tensor.reshape(-1, 28, 28)
    print(reshaped.shape)
    reshaped = np.stack((reshaped,) *3, axis = 1)
    print(reshaped.shape)
    image_arr = []

    for idx, current_image in enumerate(reshaped):
        img = torch.tensor(current_image, dtype=torch.float) / 255.
        img = vision.image.Image(img)
        image_arr.append(img)
    return image_arr

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# Unused. Old implementation
def save_images(pathBase, images, labels=None, is_test=False):
    for idx in range(len(images)):
        fPathBase = f'/tmp/{pathBase}'
        label = None
        base = None
        if not is_test:
            label = labels[idx]
            fPathBase = f'{fPathBase}/{label}'
            base = time() * 1000
        else:
            base = '{0:05d}'.format(idx + 1)
        image = images[idx]
        image = torch.tensor(image)
        image = vision.image.Image(image)
        #image.show()
        if not os.path.exists(fPathBase):
            os.makedirs(fPathBase)
        image.save(f'{fPathBase}/{base}.png')

    Path(fPathBase).ls()

def split_data(data, labels, pct=0.8):
    train_xl = []
    train_yl = []
    valid_xl = []
    valid_yl = []

    for img, label in zip(data, labels):
        if random.random() >= pct:
            valid_xl.append(img)
            valid_yl.append(label)
        else:
            train_xl.append(img)
            train_yl.append(label)
    
    return train_xl, train_yl, valid_xl, valid_yl

def create_label_lists(train_xl, train_yl, valid_xl, valid_yl):
    train_xl = TensorImageList(train_xl)
    train_yl = CategoryList(train_yl, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    valid_xl = TensorImageList(valid_xl)
    valid_yl = CategoryList(valid_yl, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
    train_ll = LabelList(train_xl, train_yl)
    valid_ll = LabelList(valid_xl, valid_yl)
    
    return LabelLists(Path('.'), train_ll, valid_ll)

class TensorImageList(ImageList):
    def get(self, i):
        img = self.items[i]
        self.sizes[i] = img.size
        return img


# In[33]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[34]:


labels = df_train.iloc[:,0].values.flatten()
images = imagify(df_train.iloc[:,1:].values)


# In[35]:


train_xl, train_yl, valid_xl, valid_yl = split_data(images, labels, 0.9)


# In[36]:


test_xl = imagify(df_test.values)
test_xl = TensorImageList(test_xl)


# In[37]:


lls = create_label_lists(train_xl, train_yl, valid_xl, valid_yl)


# In[38]:


#tfms = get_transforms(do_flip=False, max_rotate=20, xtra_tfms=rand_pad(2, 28))


# In[39]:


#tfms = (tfms[0], [])
tfms = (rand_pad(2, 28), [])


# In[40]:


mnist_data = ImageDataBunch.create_from_ll(lls, ds_tfms=tfms)


# In[41]:


mnist_data.add_test(test_xl)


# In[42]:


mnist_data.show_batch()


# In[43]:


arch = models.resnet50 # because why not?
learner = cnn_learner(mnist_data, arch, metrics=[accuracy])


# We're looking for the good `lr` which is where the loss appears to go down the most

# In[44]:


learner.lr_find()
learner.recorder.plot()


# In[45]:


lr = 1e-3


# In[46]:


learner.fit_one_cycle(10, lr)


# In[47]:


# Accuracy Plot:
learner.recorder.plot_metrics()


# In[48]:


# Losses Plot
learner.recorder.plot_losses()


# In[49]:


learner.save('mnist-1') #0.994268


# In[50]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.unfreeze()


# In[73]:


learner.fit_one_cycle(10, slice(1e-6, lr/5))


# In[74]:


learner.save('mnist-2') #0.995119


# Now that we're done training, let's see which of the training data constitutes our biggest losses.

# In[75]:


c_interpret = ClassificationInterpretation.from_learner(learner)
c_interpret.plot_top_losses(12)


# It turns out, even a human would have quite a hard time recognizing some of the numbers above.

# In[76]:


preds = learner.get_preds(ds_type=DatasetType.Test)


# In[77]:


pred_values = preds[0].argmax(1).numpy()


# In[78]:


submission = DataFrame({'ImageId': list(range(1, len(pred_values) + 1)), 'Label': pred_values})
submission.head()


# In[ ]:


#create_download_link(submission)


# In[ ]:




