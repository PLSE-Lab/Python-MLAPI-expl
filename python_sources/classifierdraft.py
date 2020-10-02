#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import seaborn as sns
import os


from PIL import Image, ImageFilter


# In[ ]:


# import os
# print(os.listdir("../input/celeba-dataset/img_align_celeba/img_align_celeba/"))


# In[ ]:


##img = Image.open('/input/CelebFaces Attributes (CelebA) Dataset/celeba-dataset/images/img_align_celeba/000391.jpg').convert('L')
img = Image.open('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/000391.jpg')


imgplot = plt.imshow(img)
plt.show()


# In[ ]:


get_ipython().run_line_magic('cd', '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/')




# In[ ]:


get_ipython().run_line_magic('ls', '')


# In[ ]:


input_data_1 = pd.read_csv('/kaggle/input/celeba-dataset/list_attr_celeba.csv')
input_data_1.head(5)
### contains all the features


# In[ ]:


input_data_1.shape


# In[ ]:


input_data_2 = pd.read_csv('list_bbox_celeba.csv')
input_data_2.head(5)
### contains the bounding box details


# In[ ]:


input_data_2.shape


# In[ ]:


input_data_3 = pd.read_csv('list_eval_partition.csv')
input_data_3.head(5)
### Shows the recommended partition 0 - training, 1 - valdation, 2 -


# In[ ]:


input_data_3.shape


# In[ ]:


input_data_4 = pd.read_csv('list_landmarks_align_celeba.csv')
input_data_4.head(5)
### face details


# In[ ]:


input_data_4.shape


# ## Pre-processing
# 
# 

# In[ ]:


input_data_1.head()


# In[ ]:


df_attributes = input_data_1
#df_attributes.set_index('image_id', inplace = True)
df_attributes.replace(to_replace=-1, value=0, inplace=True)
df_attributes.head()


# In[ ]:


## checking the distribution of male and female

plt.title("Male-Female distribution")
sns.countplot(y='Male', data=df_attributes, color='b')
plt.show()

##females are more -- imbalanced dataset

#Male : 1, Female: 0


# ## Splitting the data into training, validation and test set

# In[ ]:


input_data_3.head()


# In[ ]:


df_partition = input_data_3
##df_partition.set_index("image_id", inplace=True)
df_partition['partition'].value_counts().sort_index()


# In[ ]:


### Joining male attribute with partition dataset

df_part_attr = df_partition.join(df_attributes['Male'], how='inner')
df_part_attr.head()


# In[ ]:


### Data Augmentation


# In[ ]:





# In[ ]:


###Generating training,test and validation data


# In[ ]:


def generate_df(df,df_type):
    if(df_type is 'train'):
        df_new = df[(df['partition']==0)]
    elif(df_type is 'val'):
        df_new = df[(df['partition']==1)]
    else:
        df_new = df[(df['partition']==2)]
    
    return df_new


# In[ ]:


df_train = generate_df(df_part_attr,'train')
df_val = generate_df(df_part_attr,'val')
df_test = generate_df(df_part_attr,'test')


# In[ ]:


print(df_train.head(5))
print()
print(df_train.shape)
print('--------------------------------------')
print(df_val.head(5))
print()
print(df_val.shape)
print('--------------------------------------')
print(df_test.head(5))
print()
print(df_test.shape)


# In[ ]:


## Converting the Image to MNIST format


# In[ ]:


# 1. First image in converted into mode 'L' i.e black and white
# 2. Image is resized 
# 3. Image is sharpened (smooth pixels)
# 4.Image is pasted on canvas of size 28x28
# 5.Convert the image to array


# In[ ]:


## converting image to black-n-white
def convertImagetoMNISTstyle(path_to_image):
    im = Image.open(path_to_image).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28,28), (255)) #creates white canvas of 28x28 pixels
  
    if width > height:
        nheight = int(round(20.0/width * height), 0)
        if (nheight == 0):
            nheight = 1
      
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0))
        newImage.paste(img, (4,wtop))
    
    else:
        nwidth = int(round((20.0/height*width),0))
        if (nwidth == 0):
            nwidth = 1
      
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0))
        newImage.paste(img, (wleft,4))

    tv = list(newImage.getdata())
  
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
  
    tva_new = np.asarray(tva).reshape(28,28)
  ##print(tva)
    return tva_new
  
  #plt.show()
  #return np_im


# In[ ]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:


### test code

new_image = convertImagetoMNISTstyle('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/000391.jpg')
new_image_arr = np.asarray(new_image).reshape(28,28)
new_image_arr = new_image_arr.reshape(28,28)
new_image_arr.shape




# In[ ]:


df_train.shape


# In[ ]:


# taking all samples
df_train_new = df_train
df_test_new = df_test
df_val_new = df_val


# In[ ]:


df_val_new.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "list_train_images = []\nfor x in df_train_new['image_id']:\n    print(x)\n    path = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'+x\n    list_train_images.append(convertImagetoMNISTstyle(path))\n  ")


# In[ ]:


get_ipython().set_next_input('Image.open(path_to_image).convert');get_ipython().run_line_magic('pinfo', 'convert')


# In[ ]:


Image.open(path_to_image).convert


# In[ ]:




