#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


from os import path
from glob import glob

import pandas as pd
import numpy as np
import cv2

from tqdm.notebook import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# # Read Train Data

# In[ ]:


ROOT_DIR_TRAIN = '/kaggle/input/evohackaton/train/train'


# In[ ]:


def extract_flatten_image_from_image_name(img_name, root=None):
    if root is not None:
        im_path = path.join(root, img_name)
    else:
        im_path = img_name
    im = cv2.imread(im_path, 0)
    im = cv2.resize(im, (32,32))
    return im.flatten()


# In[ ]:


train_df = pd.read_csv('/kaggle/input/evohackaton/train.csv')
train_images = list(map(lambda x: extract_flatten_image_from_image_name(x, ROOT_DIR_TRAIN), train_df['name']))


# In[ ]:


train_images = np.stack(train_images).astype(float)
train_labels = np.array(train_df['category'])


# # Model

# In[ ]:


model = LogisticRegression()


# # Validation

# In[ ]:


val_results = cross_val_score(X=train_images, y=train_labels, estimator=model, cv=5)


# # Read Test Data

# In[ ]:


full_image_pathes = glob('/kaggle/input/evohackaton/test/test/*.jpg')
test_image_names = list(map(path.basename, full_image_pathes))

test_images = list(map(extract_flatten_image_from_image_name, full_image_pathes))


# In[ ]:


test_images = np.stack(test_images).astype(float)


# # Fit Predict

# In[ ]:


model.fit(train_images, train_labels)


# In[ ]:


y_test_hat = model.predict(test_images)


# # Create submission DF

# In[ ]:


submission_df = pd.DataFrame({
    'name':test_image_names,
    'category':y_test_hat
})


# In[ ]:


submission_df.to_csv('submission.csv',index=False)

