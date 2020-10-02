#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import os
import json

print(os.listdir("../input"))


# # Data Description

# In[ ]:


json_data=open("../input/label_descriptions.json").read()
label_descriptions = json.loads(json_data)

# dataset info
label_descriptions['info']


# In[ ]:


# dataset categories
categories_label_df = pd.DataFrame(label_descriptions['categories'])
print("The number of categories : ",len(categories_label_df))
display(categories_label_df)


# In[ ]:


# supercategory class
categories_label_df.groupby('supercategory')['name'].count()


# In[ ]:


# dataset attributes
attributes_label_df = pd.DataFrame(label_descriptions['attributes'])
print("The number of attributes : ",len(attributes_label_df))
display(attributes_label_df.head())


# # Load Train Data

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
print(train_df.shape)
print("number of unique images :",len(set(train_df['ImageId'])))
train_df.head(5)


# **There are 333,415 total data. **
# 
# **Considering only one unique image, the number of images is 45,625 total.**
# 
# **That is, there are multiple ClassIds in one image**

# In[ ]:


# split classid to category & attributes ..
def split_ClassId(data):
    
    data_c = data.copy()

    class_list = []
    attribute_list = []

    for i in range(len(data_c)):
        classid = data_c.iloc[i,4]

        # IF attribute exists, 
        if len(classid) > 2:
            class_attribute_list = classid.split("_")

            class_list.append(class_attribute_list[0])
            attribute_list.append(class_attribute_list[1:])

        else:
            class_list.append(classid)
            attribute_list.append("")

    data_c["class_"] = class_list
    data_c["attributes"] = attribute_list
    
    return data_c


# In[ ]:


# split classid to category & attributes
train_df['Category'] = train_df['ClassId'].apply(lambda x: int(x.split("_")[0]))
train_df['Attributes'] = train_df['ClassId'].apply(lambda x: list(map(int, x.split("_")[1:])))
train_df.head()


# # EDA

# In[ ]:


groupby_category = train_df.groupby('Category')['ImageId'].count()
groupby_category.index = map(int, groupby_category.index)
groupby_category = groupby_category.sort_index()
groupby_category[:5]

fig = plt.figure(figsize=(10, 4))
x = groupby_category.index
y = groupby_category.values

sns.barplot(x,y)
plt.title("Number of images by category", fontsize=20)
plt.xlabel("Category", fontsize=20)
plt.ylabel("# of images", fontsize=20)
plt.show()


# In[ ]:


groupby_category = train_df[['ImageId','Category']].groupby('ImageId').count()
print("Average number of categories per unique image: ",np.mean(groupby_category['Category'][::]))


# ## categories w and w/o attributes

# In[ ]:


train_Having_attributes_df = train_df[train_df['Attributes'].apply(len) > 0]
print("the number of images having attributes:",len(train_Having_attributes_df['ImageId']))
print("the number of unique images having attributes:",len(set(train_Having_attributes_df['ImageId'])))
#print("---------------------------------------------")


# In[ ]:


# The Number of images with Attributes by Category
groupby_category_Having_attributes = train_Having_attributes_df[['ImageId','Category']].groupby('Category').count()
groupby_category_Having_attributes.columns = ['# of imgs']
display(groupby_category_Having_attributes)


# In[ ]:


Having_attributes = list(groupby_category_Having_attributes.index)
print("categories having attributes are: \n", Having_attributes)
print()
print("categories not having attributes are: \n", list(categories_label_df.id[~categories_label_df.id.isin(Having_attributes)]))


# In[ ]:


group_category_df = train_Having_attributes_df[["Category","Attributes"]].groupby(by = "Category").sum().reset_index()
group_category_df = pd.merge(group_category_df,categories_label_df,left_on = 'Category',right_on='id')

# kind of Attributes per category
group_category_df


# **category id 0 - 12 : evenly distributed attributes **
# 
# **category id 27 28 33 : only 20, 61 attribute****

# In[ ]:


# the distribution of # of attributes per images, among images with attributes
train_Having_attributes_desc = train_Having_attributes_df.iloc[:,6].apply(len).describe()
train_Having_attributes_desc


# **The 1,1540 images having attributes have an average of 6-7 attributes**

# ---

# ## Unique Image EDA

# In[ ]:


# The number of ClassId(Category) per unique Image
group_unique_df = train_df[["ImageId", "Height"]].groupby(by = "ImageId").count().reset_index()
group_unique_df.columns = ['ImageId', 'Num_of_ClassId']
display(group_unique_df.head(3))
sns.boxplot(group_unique_df['Num_of_ClassId'])
print(group_unique_df.describe(percentiles = np.arange(0.5, 1.0, 0.05)))


# In[ ]:


class_over20_data_df = group_unique_df[group_unique_df['Num_of_ClassId'] > 20]
print(class_over20_data_df.head())
print(class_over20_data_df.shape)


# In[ ]:


group_unique_df[group_unique_df['Num_of_ClassId']==74]
train_df[train_df.ImageId=='361cc7654672860b1b7c85fe8e92b38a.jpg'].drop_duplicates('ClassId')


# **There are many images with duplicated ClassId.**

# # EDA on Image

# In[ ]:


def show_img(IMG_FILE):
    I = cv2.imread("../input/train/" + IMG_FILE, cv2.IMREAD_COLOR)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    plt.imshow(I) 
    plt.tight_layout()
    plt.show()


# ## make sample data
# - Category : 0, 4, 6, 28, 29, 31, 32
# - no Attributes
# - ImageId == '00000663ed1ff0c4e0132b9b9ac53f6e.jpg'

# In[ ]:


show_img('00000663ed1ff0c4e0132b9b9ac53f6e.jpg')


# **'1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. ****

# In[ ]:


def make_mask(IMG_FILE):
    df = train_df[train_df.ImageId == IMG_FILE].reset_index(drop = True)
    H = df.iloc[0,2]
    W = df.iloc[0,3]
    
    print("Correct Category :", sorted(set((list(df.Category)))))
    # 1d mask 
    mask = np.full(H*W,dtype='int',fill_value = -1)
    
    for line in df[['EncodedPixels','Category']].iterrows():
        EncodedPixels = line[1][0]
        Category = line[1][1]
        
        pixel_loc = list(map(int,EncodedPixels.split(' ')[0::2]))
        iter_num =  list(map(int,EncodedPixels.split(' ')[1::2]))
        for p,i in zip(pixel_loc,iter_num):
            mask[p:(p+i)] = Category
    
    print("Output :",sorted(set(list(mask))))
    #rle
    mask = mask.reshape(W,H).T
    
    return mask


# In[ ]:


# Category : 0, 4, 6, 28, 29, 31, 32
mask = make_mask('00000663ed1ff0c4e0132b9b9ac53f6e.jpg')
plt.imshow(mask,cmap='jet')


# In[ ]:


def Masking_Image(IMG_FILE):
        
    I = cv2.imread("../input/train/" + IMG_FILE, cv2.IMREAD_COLOR)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    mask = make_mask(IMG_FILE)

    fig, ax = plt.subplots(nrows = 1, ncols = 3,figsize = (18,6))
    
    ax[0].imshow(I)
    ax[1].imshow(mask, cmap = 'jet')
    ax[2].imshow(I,interpolation = 'none')
    ax[2].imshow(mask,cmap = 'jet', interpolation = 'none', alpha = 0.6)


# In[ ]:


Masking_Image('00000663ed1ff0c4e0132b9b9ac53f6e.jpg')


# In[ ]:


Masking_Image(train_df.iloc[132,0])


# In[ ]:


Masking_Image(train_df.iloc[2838,0])


# ## ** 2 Issues **
# 
# **1.  Missing Category**
#     - Maybe because the mask between categories overlaps.
# **2. Separate category id 0 and mask's default 0**

# # Binary mask 
# **Solve Problem 1**

# In[ ]:


def make_binary_mask(IMG_FILE):
    df = train_df[train_df.ImageId == IMG_FILE].reset_index(drop = True)
    H = df.iloc[0,2]
    W = df.iloc[0,3]
    binary_mask_list = []
    
    print("Correct Category :", sorted(set((list(df.Category)))))
    for line in df[['EncodedPixels','Category']].iterrows():
    
        binary_mask = np.zeros(H*W,dtype='int')
        EncodedPixels = line[1][0]
        Category = line[1][1]
        print(Category,categories_label_df.iloc[int(Category),2])
        
        pixel_loc = list(map(int,EncodedPixels.split(' ')[0::2]))
        iter_num =  list(map(int,EncodedPixels.split(' ')[1::2]))
        for p,i in zip(pixel_loc,iter_num):
            binary_mask[p:(p+i)] = 1
            
        binary_mask = binary_mask.reshape(W,H).T
        binary_mask_list.append(binary_mask)
    
    return binary_mask_list


# In[ ]:


binary_mask_list = make_binary_mask('00000663ed1ff0c4e0132b9b9ac53f6e.jpg')
plt.imshow(binary_mask_list[7])


# In[ ]:





# In[ ]:





# In[ ]:




