#!/usr/bin/env python
# coding: utf-8

# ## Separating gray and color image
# This kernel explores class imbalance between gray and color images. <br>
# The proportion of "new whale" class is considerably different between gray and color images. <br>
# So, I seperated training and test images according to gray and color images. <br>
# This kernel could be useful for someone who want to make each model of gray and color images. <br>

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

from matplotlib.pyplot import imshow
from IPython.display import HTML


# In[ ]:


img_train_path = os.path.abspath('../input/train')
img_test_path = os.path.abspath('../input/test')
csv_train_path = os.path.abspath('../input/train.csv')


# In[ ]:


df = pd.read_csv(csv_train_path)
df.head()


# In[ ]:


df['Image_path'] = [os.path.join(img_train_path,whale) for whale in df['Image']]
df.head()


# ### Training data

# In[ ]:


img = Image.open(df['Image_path'][5])
plt.imshow(img)
plt.show()


# In[ ]:


len(df)


# In[ ]:


df.Id.value_counts().head()


# In[ ]:


from tqdm import tqdm


# In[ ]:


# For color image, shape[1] = 3 
# For gray image, shape[1] = null
gray_flag = []
for i, row in tqdm(df.iterrows()) : 
    img = Image.open(row['Image_path'])
    try : 
        if np.array(img.getdata()).shape[1] == 3 :
            gray_flag.append(True)
        else : 
            gray_flag.append("Error")
    except : 
        gray_flag.append(False)    


# In[ ]:


df['is_color'] = gray_flag


# In[ ]:


df['is_color'].value_counts().plot(kind='bar') 
plt.xticks(np.arange(2), ['Color', 'Gray'])
plt.show()
print(df['is_color'].value_counts())


# ### Examples of color and gray image

# In[ ]:


color_file_path = df[df['is_color'] == True]['Image_path'][1:5]
gray_file_path = df[df['is_color'] == False]['Image_path'][1:5]


# In[ ]:


for file in color_file_path :
    img = Image.open(file)
    plt.imshow(img)
    plt.show()


# In[ ]:


for file in gray_file_path :
    img = Image.open(file)
    plt.imshow(img)
    plt.show()


# ### Class distribution 

# In[ ]:


df[df['is_color'] == True].Id.value_counts().head()


# In[ ]:


df[df['is_color'] == False].Id.value_counts().head()


# In[ ]:


# Proportion of new whale is different
color_df = df[df['is_color'] == True]
gray_df = df[df['is_color'] == False]

print("Color images")
print(len(color_df[color_df['Id'] == "new_whale"]) , '/' , len(color_df))
print(float(len(color_df[color_df['Id'] == "new_whale"]) / len(color_df)))

print("Gray images")
print(len(gray_df[gray_df['Id'] == "new_whale"]) , '/' , len(gray_df))
print(float(len(gray_df[gray_df['Id'] == "new_whale"]) / len(gray_df)))


# In[ ]:


df.loc [df['Id'] == "new_whale", "new_whale"] = "new_whale" 
df.loc [df['Id'] != "new_whale", "new_whale"] = "other"


# In[ ]:


freq_df = df.groupby(["is_color","new_whale"])["new_whale"].count().unstack("new_whale")
freq_df.plot(kind='bar', figsize=(10,5))

plt.title("New whale x Color ")
plt.xticks(np.arange(2), ['Gray', 'Color'])
plt.show()


# In[ ]:


color_df.to_csv("train_color.csv", index=False)
gray_df.to_csv("train_gray.csv", index=False)


# ### Test data

# In[ ]:


submission = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


test_image_paths= [os.path.join(img_test_path, whale) for whale in submission['Image']]


# In[ ]:


# For color image, shape[1] = 3 w
# For gray image, shape[1] = null
gray_flag_test = []
for i, path in tqdm(enumerate(test_image_paths)) : 
    img = Image.open(path)
    try : 
        if np.array(img.getdata()).shape[1] == 3 :
            gray_flag_test.append(True)
        else : 
            gray_flag_test.append("Error")
    except : 
        gray_flag_test.append(False)    


# In[ ]:


test_df = submission.copy()


# In[ ]:


test_df['is_color'] = gray_flag_test


# In[ ]:


freq_df = test_df.groupby(["is_color"])["is_color"].count()
freq_df.plot(kind='bar', figsize=(10,5))
plt.xticks(np.arange(2), ['Gray', 'Color'])
plt.title("Color")
plt.show()


# In[ ]:


color_test_df = test_df[test_df['is_color'] == True]
grah_test_df = test_df[test_df['is_color'] == False]


# In[ ]:


color_test_df.to_csv("test_color.csv", index=False)
grah_test_df.to_csv("test_gray.csv", index=False)

