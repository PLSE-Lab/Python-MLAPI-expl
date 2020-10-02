#!/usr/bin/env python
# coding: utf-8

# # Global Wheat Detection
# 
# ## Competition Problem
# - In this competition, we'll detect wheat heads from outdoor images of wheat plants, including wheat datasets from around the globe. Using worldwide data, you will focus on a generalized solution to estimate the number and size of wheat heads. To better gauge the performance for unseen genotypes, environments, and observational conditions, the training dataset covers multiple regions. 
# 
# - We will use more than 3,000 images from Europe (France, UK, Switzerland) and North America (Canada). The test data includes about 1,000 images from Australia, Japan, and China.

# ## What am I predicting?
# - We are attempting to predict bounding boxes around each wheat head in images that have them. If there are no wheat heads, you must predict no bounding boxes.

# ![Wheat Heads](https://storage.googleapis.com/kaggle-media/competitions/UofS-Wheat/descriptionimage.png)

# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
plt.style.use("seaborn")

import matplotlib.patches as patches
from glob import glob
from PIL import Image


# In[ ]:


#check the contents of the main directory
get_ipython().system('ls ../input/global-wheat-detection')


# In[ ]:


#check the count of images in the train direcory
get_ipython().system('ls ../input/global-wheat-detection/train | wc -l')


# - There are 3422 images in the train folder taken from Europe and North America.

# In[ ]:


#before we start reading the data, create path variables for convience
folder_path = '../input/global-wheat-detection/'
TRAIN_IMAGES_PATH = folder_path + 'train/'
TEST_IMAGES_PATH = folder_path + 'test/'
TRAIN_CSV = folder_path + 'train.csv'


# # Reading Data
# - Read the train csv file

# In[ ]:


#read the data
train_df = pd.read_csv(TRAIN_CSV)


# In[ ]:


train_df.head()


# # Basic Statistical Analysis.

# In[ ]:


#get the shape of the dataset
train_df.shape


# In[ ]:


#count the number of images in each directory using Glob function

train_glob = glob(TRAIN_IMAGES_PATH + '*')
test_glob = glob(TEST_IMAGES_PATH + '*')

print("Number of images in the train directory is {}".format(len(train_glob)))
print("Number of images in the test directory is {}".format(len(test_glob)))


# - Most of the test set images are hidden. We got only 10 samples of test data to check if our model is working fine without any errors. 

# In[ ]:


#check if all the images have bounding boxes or not. Check the unique number of images in the train data with bounding boxes.

unique_count = len(train_df["image_id"].unique())
print("Number of unique images in the train dataset: {}".format(unique_count))
print("Number of images without bounding boxes is: {}".format(len(train_glob) - unique_count)) 


# - Out of 3422 images, 49 images doesn't have a bounding boxes that mean there are no wheat heads in these images.
# - So we need to add these images to the train csv so that the model can learn these specific cases where there are no wheat heads.

# In[ ]:


#validate the size of the image. check width and height is equal to 1024

(train_df["width"] == train_df["height"]).all()


# In[ ]:


#check the number of sources in the train data

len(train_df["source"].unique())


# In[ ]:


train_df["source"].value_counts(normalize = True).plot(kind = "barh")
plt.title("Distribution of images from different sources")
plt.xlabel("Percentage")
plt.show()


# ### Total Observations:
# - Train folder has 3422 images in total, taken from Europe and North America.
# - Test folder has only 10 images. Most of the test data is hidden but organisers said that the hidden test data includes about 1,000 images from Australia, Japan, and China.
# - Out of 3422 images, 49 images doesn't have a bounding boxes that mean there are no wheat heads in these images.
# - So we need to add these images to the train csv so that the model can learn these specific cases where there are no wheat heads.
# - All the images have the same size: 1024 x 1024.
# - There are 7 uniques sources of data.

# In[ ]:


#Number of bounding box for each image - check the value counts

train_df["image_id"].value_counts().nlargest(5)


# - The maximum number of bounding box for one image is 116 while the minumum bounding box is 1 in the image.

# ## Extract Bounding Box data
# - Bounding box data is stored in [xmin,ymin,width,height] format

# In[ ]:


#create a new dataframe to store bounding box info
train_bbox_df = train_df[["image_id"]]
train_bbox_df["source"] = train_df["source"]


# In[ ]:


def extract_bbox(bbox_data):
    """Extract bbox data"""
    
    bbox_data = bbox_data.strip("[").strip("]").split(",")
    bbox_xmin = float(bbox_data[0])
    bbox_ymin = float(bbox_data[1])
    bbox_xmax = float(bbox_data[0]) + float(bbox_data[2])
    bbox_ymax = float(bbox_data[1]) + float(bbox_data[3])
    bbox_w = float(bbox_data[2])
    bbox_h = float(bbox_data[3])
    
    return bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_w, bbox_h


# In[ ]:


#extract the bounding box data
train_bbox_df["bbox_xmin"],train_bbox_df["bbox_ymin"], train_bbox_df["bbox_xmax"], train_bbox_df["bbox_ymax"],train_bbox_df["bbox_w"], train_bbox_df["bbox_h"] = zip(*train_df["bbox"].map(extract_bbox))


# - We know that there 49 images without any bounding boxes, so we will add those images also to the dataframe.

# Display function taken and modified from [GlobalWheatDetection EDA](https://www.kaggle.com/aleksandradeis/globalwheatdetection-eda)

# In[ ]:


#function to display the images

def get_all_bboxes(df, image_id):
    image_bboxes = df[df.image_id == image_id]
    
    bboxes = []
    for _,row in image_bboxes.iterrows():
        bboxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_w, row.bbox_h))
    return bboxes

def plot_image_samples(df, rows=3, cols=3, title='Image examples', bln_bbox = True, bln_save = False):
    fig, axs = plt.subplots(rows, cols, figsize=(10,10))
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(len(df), size=1)[0]
            img_id = df.iloc[idx].image_id
            img = Image.open(TRAIN_IMAGES_PATH + img_id + '.jpg')
            axs[row, col].imshow(img)
            
            if bln_bbox == True:                
                bboxes = get_all_bboxes(df, img_id)
                for bbox in bboxes:
                    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
                    axs[row, col].add_patch(rect)
            
            axs[row, col].axis('off')
            
    plt.suptitle(title)
    if bln_save == True:
        plt.savefig('sample.png', dpi=200)
    


# In[ ]:


#compute the bounding box area

train_bbox_df["area"] = train_bbox_df["bbox_w"] * train_bbox_df["bbox_h"]


# In[ ]:


train_bbox_df.head()


# # Exploratory Data Analysis
# - Plot the images from different sources and analyze them.

# In[ ]:


#plot images without bounding boxes

plot_image_samples(train_bbox_df, bln_bbox = False, rows = 3, cols = 3, title = "sample images")


# In[ ]:


#with bounding box
plot_image_samples(train_bbox_df, bln_bbox = True, rows = 3, cols = 3, title = "Image examples with bounding boxes", bln_save = True)


# ## Images from **usask_1**

# In[ ]:


#plot images without bounding boxes
plot_image_samples(train_bbox_df.loc[train_bbox_df["source"] == "usask_1"], bln_bbox = True, rows = 3, cols = 3, title = "Images from `usask_1'")


# ### Images from arvalis_1

# In[ ]:


#plot images without bounding boxes
plot_image_samples(train_bbox_df.loc[train_bbox_df["source"] == "arvalis_1"], bln_bbox = True, rows = 3, cols = 3, title = "Images from `arvalis_1'")


# ### Images from inrae_1

# In[ ]:


#plot images without bounding boxes
plot_image_samples(train_bbox_df.loc[train_bbox_df["source"] == "inrae_1"], bln_bbox = True, rows = 3, cols = 3, title = "Images from `inrae_1'")


# ### Images from ethz_1

# In[ ]:


#plot images without bounding boxes
plot_image_samples(train_bbox_df.loc[train_bbox_df["source"] == "ethz_1"], bln_bbox = True, rows = 3, cols = 3, title = "Images from `ethz_1'")


# # Analyzing the bounding boxes
# - Analysis based on the area.

# In[ ]:


train_bbox_df.area.value_counts().nlargest(5)


# In[ ]:


#basic stats of the area
train_bbox_df["area"].describe()


# In[ ]:


#boxplot to find out any large bounding boxes
fig, ax = plt.subplots(ncols= 2, figsize = (14,6))    

#boxplot for comparison
sns.boxplot(y = "area", data = train_bbox_df, ax=ax[0])
ax[0].set_title("Box plot of bounding box area to analyze abnormal sizes")

#distribution plot
ax[1].set_title("Distribution of bounding box area")
ax[1].set_ylabel("Frequency")
sns.distplot(a = train_bbox_df["area"], ax=ax[1], kde=False, bins = 150)

plt.show()


# - The maximum area of the bounding box is `529788` and minimum area is `2`.
# - Both maximum and minimum area indicates an abnormality, needs to be investigated.
# - But 75% of the bounding boxes have an area less than 8300 units.

# In[ ]:


#from the boxplot we can see that they are 3 instances where the bounding box area is more than 300,000.
train_bbox_df.loc[train_bbox_df["area"] > 300000]


# In[ ]:


plot_image_samples(train_bbox_df.loc[train_bbox_df["area"] > 300000], title = "Images where bounding boxes area is more than 300,000")


# In[ ]:


# we will look at all the data in the outliers
plot_image_samples(train_bbox_df.loc[train_bbox_df["area"] > 100000], title = "Images where bounding boxes area is more than 100,000")


# In[ ]:


# we will look at all the data in the outliers
plot_image_samples(train_bbox_df.loc[train_bbox_df["area"] > 100000], title = "Images where bounding boxes area is more than 100,000")


# **Observations:**
# * The color of the pictures (wheat heads) are very different from each color. Model needs to robust enough to account for these.
# * Color(Brightness) mostly depends on the source of the image. Region of the image and when the photo was taken. (During hot climate?)
# * Since these images are taken vertically, we can use different augmentation techniques like flip/rotation.
# * Many of the smaller bounding boxes are overlapping with each other.
# * The larger bounding boxes are not very clean. Lot of noise has been captured inside the bounding boxes along with the wheat heads. We need to think whether we need to include these bounding box data into the model.

# ## References
# - [GlobalWheatDetection EDA](https://www.kaggle.com/aleksandradeis/globalwheatdetection-eda)
