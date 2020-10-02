#!/usr/bin/env python
# coding: utf-8

# Notes: 
#     code is written based on kernel https://www.kaggle.com/go1dfish/clear-mask-visualization-and-simple-eda for learning purpose.

# In[ ]:


import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import seaborn as sns


# In[ ]:


ls ../input/severstal-steel-defect-detection/


# In[ ]:


train_df = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")
sample_df = pd.read_csv("../input/severstal-steel-defect-detection/sample_submission.csv")


# In[ ]:


train_df.head()


# In[ ]:


sample_df.head()


# In[ ]:


kind_class_dict = defaultdict(int)
class_dict = defaultdict(int)


# In[ ]:


no_defects_num = 0
defects_num = 0

for col in range(0, len(train_df), 4):
    img_names = [i.split('_')[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError
    labels = train_df.iloc[col:col+4,1]
    if labels.isna().all():
        no_defects_num += 1
    else:
        defects_num += 1
    kind_class_dict[sum(labels.isna() == False)] += 1
    for idx, label in enumerate(labels.isna().values.tolist()):
        if label == False:
            class_dict[idx+1] += 1


# In[ ]:


kind_class_dict ,class_dict


# In[ ]:


fig, ax = plt.subplots()
sns.barplot(x=list(class_dict.keys()), y=list(class_dict.values()))
ax.set_title('the number of images for each class')
ax.set_xlabel('class')


# In[ ]:


fig, ax = plt.subplots()
sns.barplot(x = list(kind_class_dict.keys()), y = list(kind_class_dict.values()))
ax.set_title('Number defects based on image')
ax.set_xlabel("number of class ")


# In[ ]:


train_size_dict = defaultdict(int)
train_path = Path("../input/severstal-steel-defect-detection/train_images/")


# In[ ]:


for img_name in train_path.iterdir():
    img = Image.open(img_name)
    train_size_dict[img.size] += 1


# In[ ]:


train_size_dict


# In[ ]:


test_size_dict = defaultdict(int)
test_path = Path('../input/severstal-steel-defect-detection/test_images/')


# In[ ]:


for img in test_path.iterdir():
    img = Image.open(img_name)
    test_size_dict[img.size] += 1
    


# In[ ]:


test_size_dict


# In[ ]:


train_size_dict


# In[ ]:


palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]


# In[ ]:


def name_and_mask(start_idx):
    col = start_idx
    img_names = [i.split('_')[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError
    
    labels  = train_df.iloc[col:col+4, 1]
    mask  = np.zeros((256,1600,4),dtype=np.uint8)
    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            mask_label = np.zeros(1600*256,dtype=np.uint8)
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                mask_label[pos-1:pos+le-1] = 1
            mask[:, :, idx] = mask_label.reshape(256,1600,order='F')
    return img_names[0], mask


# In[ ]:


def show_mask_image(col):
    name, mask = name_and_mask(col)
    img = cv2.imread(str(train_path/name))
    fig, ax = plt.subplots(figsize=(15,15))
    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    ax.set_title(name)
    ax.imshow(img)
    plt.show()


# In[ ]:


fig, ax = plt.subplots(1,4, figsize=(15,5))
for i in range(4):
    ax[i].axis('off')
    ax[i].imshow(np.ones((50, 50, 3), dtype=np.uint8) * palet[i])
    ax[i].set_title("class color: {}".format(i+1))
fig.suptitle("each class colors")
plt.show()


# In[ ]:


idx_no_defect = []
idx_class_1 = []
idx_class_2 = []
idx_class_3 = []
idx_class_4 = []
idx_class_trip = []
idx_class_multi = []
for col in range(0, len(train_df), 4):
    img_names = [str(i).split('_')[0] for i in train_df.iloc[col:col+4, 0]]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError
    labels  = train_df.iloc[col:col+4, 1]
    if labels.isna().all():
        idx_no_defect.append(col)
    elif (labels.isna() == [False, True, True, True]).all():
        idx_class_1.append(col)
    elif (labels.isna() == [True, False, True, True]).all():
        idx_class_2.append(col)
    elif (labels.isna() == [True, True, False, True]).all():
        idx_class_3.append(col)
    elif (labels.isna() == [True, True, True, False]).all():
        idx_class_4.append(col)
    elif labels.isna().sum() == 1:
        idx_class_trip.append(col)
    else:
        idx_class_multi.append(col)


# In[ ]:


for idx in idx_no_defect[:5]:
    show_mask_image(idx)


# In[ ]:


for idx in idx_class_1[:5]:
    show_mask_image(idx)


# In[ ]:


for idx in idx_class_2[:5]:
    show_mask_image(idx)


# In[ ]:


for idx in idx_class_3[:5]:
    show_mask_image(idx)


# In[ ]:


for idx in idx_class_4[:5]:
    show_mask_image(idx)


# In[ ]:


for idx in idx_class_trip[:5]:
    show_mask_image(idx)


# In[ ]:


for idx in idx_class_multi[:5]:
    show_mask_image(idx)


# In[ ]:




