#!/usr/bin/env python
# coding: utf-8

# #  Google Landmark Retrieval 2020 - EDA
# 
# The purpose of this competition is to retrieve relevant database images to a given query image (ie, the model should retrieve database images containing the same landmark as the query).
# 
# The competition this year is different from previous two years in that it requires you to submit a model instead of the prediction file. From the organizer:
# 
# > Your model must be named submission.zip and be compatible with TensorFlow 2.2. The submission.zip should contain all files and directories created by the tf.saved_model_save function using Tensorflow's SavedModel format.
# 
# This dataset is a large one and has [101.99 GB](https://www.kaggle.com/c/landmark-retrieval-2020/data). 
# 
# ## Acknowledgement
# 
# This EDA kernel takes reference from the following excellent kernels:
# - https://www.kaggle.com/huangxiaoquan/google-landmarks-v2-exploratory-data-analysis-eda
# - https://www.kaggle.com/shivyshiv/exploratory-eda-for-google-landmark-retrieval-202
# - https://www.kaggle.com/seriousran/google-landmark-retrieval-2020-eda
# 
# ## Table of Content
# * [Import Packages](#import-packages)
# * [File Structure](#file-structure)
# * [Train File](#train-file)
# * [Landmark ID](#landmark-id)
# * [Pictures at Train, Test, and Index](#pic-train-test-index)
# * [Pictures by Class](#pic-by-class)
# 

# <a id="import-packages"></a>
# ## Import Packages

# In[ ]:


import os
import glob
import math
import numpy as np 
import pandas as pd 
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import cv2


# <a id="file-structure"></a>
# ## File Structure
# 
# The dataset has one file (`train.csv`) and three folders (`train`, `test`, `index`) in the root path.

# In[ ]:


get_ipython().system('ls ../input/landmark-retrieval-2020')


# The `train`, `test`, and `index` folders each have a three-level hierarchy. The first three digits/letters of the file decides how the file is placed. From the organizer:
# 
# > Each image has a unique id. Since there are a large number of images, each image is placed within three subfolders according to the first three characters of the image id (i.e. image abcdef.jpg is placed in a/b/c/abcdef.jpg).
# 
# For example, a picture named `00009069e8450638.jpg` will be placed at `0/0/0/00009069e8450638.jpg` since the first three digits/letters are `0`, `0`, and `0`. For another example, a picture named `0012f4bb3812e158.jpg` can be found at `0/0/1/0012f4bb3812e158.jpg` since the first three digits/letters are `0`, `0`, and `1`. 
# 
# The file hierarchy is designed in this way probably because of the ease of searching; otherwise, we need to search in a folder consisting of 1 million images which would be slow.

# In[ ]:


get_ipython().system('ls ../input/landmark-retrieval-2020/train')


# In[ ]:


get_ipython().system('ls ../input/landmark-retrieval-2020/train/0')


# In[ ]:


get_ipython().system('ls ../input/landmark-retrieval-2020/train/0/0/')


# From the following, each of the sub-folder in `train` consists of around 400 pictures. 16 folders * 16 folders * 16 folders * 400 images/folder is around 1.5 million images.

# In[ ]:


get_ipython().system('ls ../input/landmark-retrieval-2020/train/0/0/0 | wc -l')


# In[ ]:


get_ipython().system('ls ../input/landmark-retrieval-2020/train/0/0/1 | wc -l')


# In[ ]:


get_ipython().system('ls ../input/landmark-retrieval-2020/train/9/9/f | wc -l')


# On the other hand, for `test` folder, there are not necessarily all folders from `0` to `f`. 

# In[ ]:


get_ipython().system('ls ../input/landmark-retrieval-2020/test/0')


# In[ ]:


get_ipython().system('ls ../input/landmark-retrieval-2020/test/1/0')


# Last but not least, let's count the number of pictures in each main folder.

# In[ ]:


train_list = glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')
test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')
index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')
print("There are {} images in train folder, {} images in test folder, and {} images in index folder.".format(
    len(train_list), len(test_list), len(index_list)))


# <a id="train-file"></a>
# ## Train File (`train.csv`)
# 
# We don't have access to the meta-data of the testing set. Nevertheless, we know we are dealing with 1.5 million pictures. The two columns - `id` and `landmark_id` - are string and integers respectively. 

# In[ ]:


train_file_path = '../input/landmark-retrieval-2020/train.csv'
df_train = pd.read_csv(train_file_path)

print("Training data size:", df_train.shape)
print("Training data columns: {}\n\n".format(df_train.columns))
print(df_train.info())


# In[ ]:


df_train.head(5)


# In[ ]:


df_train.tail(5)


# Great news - no N/A in `train.csv`. 

# In[ ]:


missing = df_train.isnull().sum()
percent = missing/df_train.count()
missing_train_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])
missing_train_data.head()


# <a id="landmark-id"></a>
# ## Landmark ID 
# 
# From the following, there are 81313 classes of landmark. Please note that the landmark_id is not placed in a consecutive manner. Probably images for some of the IDs in between are skipped or taken out by the competition organizer.

# In[ ]:


print("Minimum of landmark_id: {}, maximum of landmark_id: {}".format(df_train['landmark_id'].min(), df_train['landmark_id'].max()))
print("Number of unique landmark_id: {}".format(len(df_train['landmark_id'].unique())))
print(df_train['landmark_id'].unique())


# Consistent with the above finding, the maximum value of x-axis for the following plot is around 80,000 (i.e. there are around 80,000 unique landmark_id). From the shape of the plot, we are dealing with a very unbalanced dataset. We need to be careful on the training tactics and evaluation metric(s). 

# In[ ]:


sns.set()
plt.title('Training set: number of images per class(line plot)')
sns.set_color_codes("pastel")
landmarks_fold = pd.DataFrame(df_train['landmark_id'].value_counts())
landmarks_fold.reset_index(inplace=True)
landmarks_fold.columns = ['landmark_id','count']
ax = landmarks_fold['count'].plot(logy=True, grid=True)
locs, labels = plt.xticks()
plt.setp(labels, rotation=30)
ax.set(xlabel="Landmarks", ylabel="Number of images")


# In[ ]:


df_count = pd.DataFrame(df_train.landmark_id.value_counts().sort_values(ascending=False))
df_count.reset_index(inplace=True)
df_count.columns = ['landmark_id', 'count']
df_count


# Let's take a look at the top 10 most appearing landmark_id. So the top landmark_id `138982` has more than 600 images, whereas the rest has around 1000-2000 images.

# In[ ]:


sns.set()
plt.figure(figsize=(9, 4))
plt.title('Most frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(
    x="landmark_id",
    y="count",
    data=df_count.head(10),
    label="Count",
    order=df_count.head(10).landmark_id)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.show()


# Following is the function to print images in a grid manner.

# In[ ]:


def print_img(class_id, df_class, figsize):
    file_path = "../input/landmark-retrieval-2020/train/"
    df = df_train[df_train['landmark_id'] == class_id].reset_index()
    
    print("Class {} - {}".format(class_id, df_class[class_id].split(':')[-1]))
    print("Number of images: {}".format(len(df)))
    
    plt.rcParams["axes.grid"] = False
    no_row = math.ceil(min(len(df), 12)/3) 
    f, axarr = plt.subplots(no_row, 3, figsize=figsize)

    curr_row = 0
    len_img = min(12, len(df))
    for i in range(len_img):
        img_name = df['id'][i] + ".jpg"
        img_path = os.path.join(
            file_path, img_name[0], img_name[1], img_name[2], img_name)
        example = cv2.imread(img_path)
        # uncomment the following if u wanna rotate the image
        # example = cv2.rotate(example, cv2.ROTATE_180)
        example = example[:,:,::-1]

        col = i % 3
        axarr[curr_row, col].imshow(example)
        axarr[curr_row, col].set_title("{}. {} ({})".format(
            class_id, df_class[class_id].split(':')[-1], df['id'][i]))
        if col == 2:
            curr_row += 1


# Let's get the name of each landmark category (credit goes to @sudeepshouche).

# In[ ]:


# From: https://www.kaggle.com/sudeepshouche/identify-landmark-name-from-landmark-id
url = 'https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv'
df_class = pd.read_csv(url, index_col = 'landmark_id', encoding='latin', engine='python')['category'].to_dict()


# Following are some sample pictures for `landmark_id = 138982`. The bottom of each picture may need to be removed for running any recognition algorithm during preprocessing. On the other hand, they might be helpful with text analysis methodologies. 

# In[ ]:


class_id = 138982
print_img(class_id, df_class, (24, 18))


# Following are some sample pictures for `landmark_id = 126637`. Unlike the above, there is no such thing as human hand-writing at the bottom. On the other hand, the category consists of a diverse topic of images such as plants and ships in different angles and orientation.

# In[ ]:


class_id = 126637
print_img(class_id, df_class, (24, 18))


# Following this line of analysis, let's count the number images under a certain threshold count. 

# In[ ]:


threshold = [2, 3, 5, 10, 20, 50, 100, 200, 1000]
total = len(df_train['landmark_id'].unique())
for num in threshold:
    cnt = (df_train['landmark_id'].value_counts() < num).sum()
    print("Number of classes with {} images or less: {}/{} ({:.2f}%)".format(
        num, cnt, total, cnt/total*100))


# <a id="pic-train-test-index"></a>
# ## Pictures at Train, Test, and Index
# 
# Let's take a brief look at the pictures at Train, Test, and Index folders.
# 
# ### Train

# In[ ]:


plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(4, 3, figsize=(24, 22))

curr_row = 0
for i in range(12):
    example = cv2.imread(train_list[i])
    # uncomment the following if u wanna rotate the image
    # example = cv2.rotate(example, cv2.ROTATE_180)
    example = example[:,:,::-1]
    
    col = i % 4
    axarr[col, curr_row].imshow(example)
    axarr[col, curr_row].set_title(train_list[i])
    if col == 3:
        curr_row += 1


# ### Test

# In[ ]:


plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(4, 3, figsize=(24, 22))

curr_row = 0
for i in range(12):
    example = cv2.imread(test_list[i])
    # uncomment the following if u wanna rotate the image
    # example = cv2.rotate(example, cv2.ROTATE_180)
    example = example[:,:,::-1]
    
    col = i % 4
    axarr[col, curr_row].imshow(example)
    axarr[col, curr_row].set_title(test_list[i])
    if col == 3:
        curr_row += 1


# ### Index

# In[ ]:


plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(4, 3, figsize=(24, 22))

curr_row = 0
for i in range(12):
    example = cv2.imread(index_list[i])
    # uncomment the following if u wanna rotate the image
    # example = cv2.rotate(example, cv2.ROTATE_180)
    example = example[:,:,::-1]
    
    col = i % 4
    axarr[col, curr_row].imshow(example)
    axarr[col, curr_row].set_title(index_list[i])
    if col == 3:
        curr_row += 1


# <a id="pic-by-class"></a>
# ## Pictures by Class

# In[ ]:


class_id = 1
print_img(class_id, df_class, (24, 12))


# In[ ]:


class_id = 7
print_img(class_id, df_class, (24, 12))


# In[ ]:


class_id = 9
print_img(class_id, df_class, (24, 16))


# In[ ]:


class_id = 11
print_img(class_id, df_class, (24, 16))


# In[ ]:


class_id = 12
print_img(class_id, df_class, (24, 16))


# In[ ]:


class_id = 22
print_img(class_id, df_class, (24, 16))


# To be continued .....
