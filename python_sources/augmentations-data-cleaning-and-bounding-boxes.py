#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import os
import cv2
import csv
import glob
import pandas as pd
import numpy as np
import random
import itertools
from collections import Counter
from math import ceil
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')


# In this kernel, I present some utility functions to do some sanity check on images, as well as some functions that you can reuse for future projects when you want to plot multiple images in a grid. A sneak peek of how a multiple bounding box plot is as such:

# ![sample](https://i.ibb.co/9GXMpWT/img.png)

# **References**
# 
# - [Paperspace DataAugmentationForObjectDetection](https://github.com/Paperspace/DataAugmentationForObjectDetection)
# 
# - [ateplyuk's gwd starter](https://www.kaggle.com/ateplyuk/gwd-starter-efficientdet-train)
# 
# - [Shonenkov's awesome code](https://www.kaggle.com/shonenkov/training-efficientdet)

# # Utilities

# Utility functions are stored here, they are useful and feel free to add these into your arsenal.

# In[ ]:


def skip_csv_header(file):
    has_header = csv.Sniffer().has_header(file.read(1024))
    file.seek(0)
    if has_header:
        next(file)


def total_image_list(image_folder_path):
    total_img_list = [os.path.basename(img_path_name) for img_path_name in glob.glob(os.path.join(image_folder_path, "*.jpg"))]
    return total_img_list

def draw_rect(img, bboxes, color=None):
    img = img.copy()
    bboxes = bboxes[:, :4]
    bboxes = bboxes.reshape(-1, 4)
    for bbox in bboxes:
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
        img = cv2.rectangle(img.copy(), pt1, pt2, color, int(max(img.shape[:2]) / 200))
    return img

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=ceil(len(img_matrix_list) / ncols), ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()


# # Reading and Loading the Dataset

# In[ ]:


train = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")  
image_folder_path = "/kaggle/input/global-wheat-detection/train/"


# I personally like to expand the bounding box coordinates into the form of **x_min, y_min, x_max, y_max**, but currently they are stored in a list of **[x_min,y_min, width of bbox, height of bbox]**. So the next portion will help to expand them out. **This is a personal preference, in actual fact you do not need to do this, it is easier for me to normalize the bboxes**.

# In[ ]:


# train['bbox'] = train['bbox'].apply(lambda x: x[1:-1].split(","))
# train['x_min'] = train['bbox'].apply(lambda x: x[0]).astype('float32')
# train['y_min'] = train['bbox'].apply(lambda x: x[1]).astype('float32')
# train['width'] = train['bbox'].apply(lambda x: x[2]).astype('float32')
# train['height'] = train['bbox'].apply(lambda x: x[3]).astype('float32')
# train = train[['image_id','x_min', 'y_min', 'width', 'height']]
# train["x_max"] = train.apply(lambda col: col.x_min + col.width, axis=1)
# train["y_max"] = train.apply(lambda col: col.y_min + col.height, axis = 1)
# train.head()


# I have since used [Shonenkov's awesome code](https://www.kaggle.com/shonenkov/training-efficientdet) to make the code above more compact.

# In[ ]:


bboxes = np.stack(train['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x_min', 'y_min', 'width', 'height']):
    train[column] = bboxes[:,i]
    
train["x_max"] = train.apply(lambda col: col.x_min + col.width, axis=1)
train["y_max"] = train.apply(lambda col: col.y_min + col.height, axis = 1)
train.drop(columns=['bbox'], inplace=True)


# In[ ]:


train.head()


# # Range Checking on Bounding Box Coordinates

# Furthermore, due to python's internal floating problems, there may be weird values like negative or values that adds up to be more than 1024 in `x_max, y_max`. We need to be careful here. 
# 
# **This is a serious problem that one can run into when you Normalize the bounding box, it may exceed 1 and this will cause an error especially if you decide to augment the images as well.**

# In[ ]:


train[train["x_max"] > 1024]
train[train["y_max"] > 1024]
train[train["x_min"] < 0]
train[train["y_min"] < 0]


# The sole reason that for eg row 31785 has `x_max` more than 1024 is because of the original dataset's labelling. Let's look at the respective problematic rows. For example, in row 31785, the `x_min` provided is 873.200012, and when you add that to the width being 150.800003, it gives you 1024.000015, which exceeds the image size already. So you have to round down. And as far as I feel, bounding boxes, when de-normalized, should necessary be in integer. But this is just my opinion. Let's change these problematic values to 1024

# In[ ]:


x_max = np.array(train["x_max"].values.tolist())
y_max = np.array(train["y_max"].values.tolist())
train["x_max"] = np.where(x_max > 1024, 1024, x_max).tolist()
train["y_max"] = np.where(y_max > 1024, 1024, y_max).tolist()


# We can delete width and height columns because we do not need them, it can be easily pulled out from the images itself.

# In[ ]:


del train["width"]
del train["height"]
del train["source"]
train.head()


# I assign a class "1" which is the label wheat. It may be useful later on should we wish to add in images with no wheat inside the image.

# In[ ]:


train["class"] = "1"


# # Check if image extensions are all jpg

# First, we check if all images in the train folder are all in **.jpg** format. It is better to check because if there are a mixture of image type, we may face troubles later on.

# In[ ]:


def check_file_type(image_folder_path):
    extension_type = []
    file_list = os.listdir(image_folder_path)
    for file in file_list:
        extension_type.append(file.rsplit(".", 1)[1].lower())
    print(Counter(extension_type).keys())
    print(Counter(extension_type).values())
    
check_file_type(image_folder_path)


# Good, seems like all our images in the folder are of **.jpg** format. Next, it is better to append **.jpg** behind all the **image_id** in the dataframe. This will make us manipulate the data easier later.

# In[ ]:


## replace image_id with .jpg behind the image_id
# image_id_list = train["image_id"].tolist()
# image_id_append_jpg = []
# for image_id in image_id_list:
#     image_id_append_jpg.append(image_id + ".jpg")
# train["image_id"] = image_id_append_jpg
# train.head()


## Alternatively like Rohit suggested, an one liner will do the trick

train["image_id"] = train["image_id"].apply(lambda x: str(x) + ".jpg")
train.head()


# In[ ]:


train["image_id"] = train["image_id"].astype("str")


# In[ ]:


train.to_csv("wheat.csv", index=False)


# # Check if there are corrupted images and all images are 1024 by 1024

# Most people will use `df['width'].unique() == df['height'].unique() == [1024]` to check if all images are of 1024x1024 resolution; But we will not be 100% sure if its true in the training folder. So we won't use the same way here.

# In[ ]:


def check_image_size(image_folder_path):
    total_img_list = glob.glob(os.path.join(image_folder_path,"*"))
    counter = 0
    for image in tqdm(total_img_list, desc = "Checking in progress"):
        try:
            img = cv2.imread(image)
            height, width = img.shape[1], img.shape[0]
            if not (height == 1024 and width == 1024):
                counter = counter + 1
        except:
            print("This {} is problematic.".format(image))
    return counter 
        
        


# In[ ]:


check_image_size(image_folder_path)


# Great, indeed all our images are of 1024x1024 in size. And the good thing is, this code also helps us to check for corrupted images as well, so if there is a corrupted image, it will definitely show up that the counter is non zero. And from there you can further check which image is the one causing problem.

# # Sanity Check between train csv and train images

# We will write a function to check if the number of **unique image_ids** match the number of unique **images** in the folder.

# In[ ]:


## our new dataset
wheat = pd.read_csv("wheat.csv") 
image_folder_path = "/kaggle/input/global-wheat-detection/train/"
image_annotation_file = "wheat.csv"


# In[ ]:


wheat.head()


# In[ ]:


def sanity_tally(image_folder_path, image_annotation_file):
    img_dict = {}
    with open(image_annotation_file, "r") as file:
        skip_csv_header(file)
        for row in file:
            try:
                image_name, x_min, y_min, x_max, y_max, class_idx = row.split(",")
                if image_name not in img_dict:
                    img_dict[image_name] = list()
                img_dict[image_name].append(
                    [float(x_min), float(y_min), float(x_max), float(y_max), int(class_idx)]
                )
            except ValueError:
                print("Could not convert float to string, likely that your data has empty values.")
        
    img_annotation_list = [*img_dict]
    total_img_list = total_image_list(image_folder_path)
    if set(img_annotation_list) == set(total_img_list):
        print("Sanity Check Status: True")
    else:
        print("Sanity Check Status: Failed. \nThe elements in wheat/train.csv but not in the train image folder is {}. \nThe elements in train image folder but not in wheat/train.csv is {}".format(
                set(img_annotation_list) - set(total_img_list), set(total_img_list) - set(img_annotation_list)))
        return list(set(img_annotation_list) - set(total_img_list)), list(set(total_img_list) - set(img_annotation_list))


# In[ ]:


set_diff1, set_diff2 = sanity_tally(image_folder_path, image_annotation_file = image_annotation_file)

print("There are {} images without annotations in the train/wheat.csv".format(len(set_diff2)))


# As we can see from the above, there are 49 images without bounding box annotations because they do not have wheats in the image, and hence did not appear in the **train.csv**. It might be an idea that we can put these 49 images inside the train.csv and label them as 0.

# # Plotting Multiple Images

# Here we define a nice function that is useful not only for this competition, but for similar project as well. Note that we used our utility function here to plot them. One can tune the parameters accordingly.

# In[ ]:


def plot_random_images(image_folder_path, image_annotation_file, num = 12):
    img_dict = {}
    with open(image_annotation_file, "r") as file:
        skip_csv_header(file)
        for row in file:
            try:
                image_name, x_min, y_min, x_max, y_max, class_idx = row.split(",")
                if image_name not in img_dict:
                    img_dict[image_name] = list()
                img_dict[image_name].append(
                    [float(x_min), float(y_min), float(x_max), float(y_max), int(class_idx)]
                )
            except ValueError:
                print("Could not convert float to string, likely that your data has empty values.")

    # randomly choose 12 images to plot
    img_files_list = np.random.choice(list(img_dict.keys()), num)
    print("The images' names are {}".format(img_files_list))
    img_matrix_list = []
    
    for img_file in img_files_list:
        image_file_path = os.path.join(image_folder_path, img_file)
        img = cv2.imread(image_file_path)[:,:,::-1]  
        img_matrix_list.append(img)

    
    return plot_multiple_img(img_matrix_list, title_list = img_files_list, ncols = 4, main_title="Wheat Images")


# Here we see a nice grid of 12 images plotted.

# In[ ]:


plot_random_images(image_folder_path, image_annotation_file, num = 12)


# # Plotting Multiples Images with Bounding Boxes

# In object detection with bounding boxes, it is always a good idea to randomly plot some images with their bounding boxes to check for any awry bounding box coordinates. Although I have to say that in this particular competition, there are quite a lot of images with many bounding boxes and hence you have to scrutinize clearly.

# In[ ]:


def random_bbox_check(image_folder_path, image_annotation_file, num = 12):
    img_dict = {}
    labels = ["wheat", "no wheat"]
    with open(image_annotation_file, "r") as file:
        skip_csv_header(file)
        for row in file:
            try:
                image_name, x_min, y_min, x_max, y_max, class_idx = row.split(",")
                if image_name not in img_dict:
                    img_dict[image_name] = list()
                img_dict[image_name].append(
                    [float(x_min), float(y_min), float(x_max), float(y_max), int(class_idx)]
                )
            except ValueError:
                print("Could not convert float to string, likely that your data has empty values.")

    # randomly choose 12 image.
    img_files_list = np.random.choice(list(img_dict.keys()), num)
    print("The images' names are {}".format(img_files_list))
    image_file_path_list = []

    bbox_list = []
    img_matrix_list = []
    random_image_matrix_list = []
    
    for img_file in img_files_list:
        image_file_path = os.path.join(image_folder_path, img_file)
        img = cv2.imread(image_file_path)[:,:,::-1]  
        height, width, channels = img.shape
        bbox_list.append(img_dict[img_file])
        img_matrix_list.append(img)

    
    final_bbox_list = []
    for bboxes, img in zip(bbox_list, img_matrix_list):
        final_bbox_array = np.array([])
        #bboxes is a 2d array [[...], [...]]
        for bbox in bboxes:
            bbox = np.array(bbox).reshape(1,5)
            final_bbox_array = np.append(final_bbox_array, bbox)
        final_bbox_array = final_bbox_array.reshape(-1,5)
        random_image = draw_rect(img.copy(), final_bbox_array.copy(), color = (255,0,0))
        random_image_matrix_list.append(random_image)
    plot_multiple_img(random_image_matrix_list, title_list = img_files_list, ncols = 4, main_title="Bounding Box Wheat Images")    
    


# In[ ]:


random_bbox_check(image_folder_path, image_annotation_file)


# # Augmentations

# Augmentation is an important technique to artifically boost your data size. In particular, when the dataset is small, augmentation prior to training the model will help the network to learn better.

# In[ ]:


# Albumentations
import albumentations as A


# In[ ]:


image_folder_path = "/kaggle/input/global-wheat-detection/train/"
chosen_image = cv2.imread(os.path.join(image_folder_path, "1ee6b9669.jpg"))[:,:,::-1]
plt.imshow(chosen_image)


# And if you are interested in just visualizing one certain image's bounding box plot, you can first extract the chosen image's dataframe, and convert the bounding box of the image into a **2d-array**. Then apply the `draw_rect` function to plot.

# In[ ]:


chosen_image_dataframe = wheat.loc[wheat["image_id"]=="1ee6b9669.jpg",["x_min","y_min","x_max","y_max","class"]]
bbox_array_of_chosen_image = np.array(chosen_image_dataframe.values.tolist())
bbox_array_of_chosen_image.shape


# In[ ]:


draw_chosen_image = draw_rect(chosen_image.copy(), bbox_array_of_chosen_image.copy(), color = (255,0,0))
plt.imshow(draw_chosen_image)


# Below are some snippets of augmentation types you can use, interestingly, Albumentation offers `RandomSunFlare`,`RandomFog` and `RandomSnow`; although all the images seem to be taken in a very good lighting, but it might not be that bad an idea since in the real world, images of wheat may taken in ***different weather conditions.***

# In[ ]:


albumentation_list = [A.RandomSunFlare(p=1), A.RandomFog(p=1), A.RandomBrightness(p=1),
                      A.RandomCrop(p=1,height = 512, width = 512), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.RandomSnow(p=1),
                      A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomContrast(limit = 0.5,p = 1),
                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]

img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)

img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","RandomSunFlare","RandomFog","RandomBrightness",
               "RandomCrop","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]

##reminder of helper function
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")


# # Bounding Boxes with Albumentations

# Recall we are using our **chosen image** as example, for convenience, I will remind you of the chosen images image matrix and its bounding boxes coordinates below. But there is a caveat here, my bounding boxes array is of shape [N,5], where the last element is the labels. But when you want to use Albumentations to plot bounding boxes, it takes in bboxes in the format of `pascal_voc` which is **[x_min, y_min, x_max, y_max]**; it also takes in `label_fields` which are the labels for each bounding box. So we still need to do some simple preprocessing below.

# In[ ]:


chosen_image = cv2.imread(os.path.join(image_folder_path, "1ee6b9669.jpg"))[:,:,::-1]
chosen_image_dataframe = wheat.loc[wheat["image_id"]=="1ee6b9669.jpg",["x_min","y_min","x_max","y_max"]]
bbox_array_of_chosen_image = np.array(chosen_image_dataframe.values.tolist())
labels_of_chosen_image = np.ones((len(bbox_array_of_chosen_image),))

# A caution here, this competition has all labels to be 1, so a neat trick is just to use np.ones to populate the label fields.
# However, when dealing with multiple classes, you should not do this and instead just take the labels from the dataframe accordingly.


# **References:** [Albumentation Documentations](https://github.com/albumentations-team/albumentations_examples/tree/master/notebooks)

# In[ ]:


def draw_rect_with_labels(img, bboxes,class_id, class_dict, color=None):
    img = img.copy()
    bboxes = bboxes[:, :4]
    bboxes = bboxes.reshape(-1, 4)
    for bbox, label in zip(bboxes, class_id):
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
        class_name = class_dict[label]
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1) 
        img = cv2.rectangle(img.copy(), pt1, pt2, color, int(max(img.shape[:2]) / 200))
        img = cv2.putText(img.copy(), class_name, (int(bbox[0]), int(bbox[1]) - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color = (255,255,255), lineType=cv2.LINE_AA)
    return img


# ## Vertical Flip

# In[ ]:


ver_flip = A.Compose([
        A.VerticalFlip(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


ver_flip_annotations = ver_flip(image=chosen_image, bboxes=bbox_array_of_chosen_image, labels=labels_of_chosen_image)
ver_flip_annotations['bboxes'] = [list(bbox) for bbox in ver_flip_annotations['bboxes']]


# In[ ]:


ver_flip_img = draw_rect_with_labels(img = ver_flip_annotations['image'], bboxes = np.array(ver_flip_annotations['bboxes']),
                          class_id = ver_flip_annotations['labels'], class_dict = {0: "background",1: "wheat"}, color=(255,0,0))

## using my good old plotting functions
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=1, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
img_matrix_list = [draw_chosen_image, ver_flip_img]
titles_list = ["Original", "VerticalFlipped"]

plot_multiple_img(img_matrix_list, titles_list, ncols = 2,main_title="Vertical Flip")


# ## Horizontal Flip

# In[ ]:


hor_flip = A.Compose([
        A.HorizontalFlip(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


hor_flip_annotations = hor_flip(image=chosen_image, bboxes=bbox_array_of_chosen_image, labels=labels_of_chosen_image)
hor_flip_annotations['bboxes'] = [list(bbox) for bbox in hor_flip_annotations['bboxes']]


hor_flip_img = draw_rect_with_labels(img = hor_flip_annotations['image'], bboxes = np.array(hor_flip_annotations['bboxes']),
                          class_id = hor_flip_annotations['labels'], class_dict = {0: "background",1: "wheat"}, color=(255,0,0))
    
img_matrix_list = [draw_chosen_image, hor_flip_img]
titles_list = ["Original", "HorizontalFlipped"]

plot_multiple_img(img_matrix_list, titles_list, ncols = 2,main_title="Horizontal Flip")


# # An updated tutorial on Albumentations

# Recently, Albumentations updated their website and has more thorough walkthrough on applying their augmentations. 
# 
# The link is [here](https://albumentations.ai/docs/).

# ## Image Augmentation for classification

# The below are purely for my documentation and learning experience so most of the content are copied verbatim from [Albumentation's website](https://albumentations.ai/docs/getting_started/image_augmentation/).

# **Step 1: Import the required libraries**
# 
#     import cv2
#     import albumentations as A
# 
# **Step 2: Define an augmentation pipeline.**
# 
# To define an augmentation pipeline, you need to create an instance of the `Compose` class. As an argument to the `Compose` class, you need to pass a list of augmentations you want to apply. A call to `Compose` will return a transform function that will perform image augmentation.
# 
# Let's look at an example:
# 
#     transform = A.Compose([
#         A.RandomCrop(width=256, height=256),
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#     ])
# 
# In the example, `Compose` receives a list with three augmentations: `A.RandomCrop`, `A.HorizontalFlip`, and `A.RandomBrighntessContrast`.
# 
# To create an augmentation, you create an instance of the required augmentation class and pass augmentation parameters to it. `A.RandomCrop` receives two parameters, height and width. `A.RandomCrop(width=256, height=256)` means that `A.RandomCrop` will take an input image, extract a random patch with size 256 by 256 pixels from it and then pass the result to the next augmentation in the pipeline (in this case to `A.HorizontalFlip`).
# 
# `A.HorizontalFlip` in this example has one parameter named `p`. `p` is a special parameter that is supported by almost all augmentations. It controls the probability of applying the augmentation. `p=0.5` means that with a probability of 50%, the transform will flip the image horizontally, and with a probability of 50%, the transform won't modify the input image.
# 
# `A.RandomBrighntessContrast` in the example also has one parameter, `p`. With a probability of 20%, this augmentation will change the brightness and contrast of the image received from `A.HorizontalFlip`. And with a probability of 80%, it will keep the received image unchanged.
# 
# The following picture depicts the `Compose` idea wonderfully.
# 
# ALSO THIS [WEBSITE IS AMAZING FOR YOU TO TUNE AND VISUALIZE YOUR AUGMENTATIONS](https://albumentations-demo.herokuapp.com/).

# ![augmentation-pipeline](https://albumentations.ai/docs/images/getting_started/augmenting_images/augmentation_pipeline_visualized.jpg)

# Now we will use the above context on our wheat example.

# In[ ]:


transform = A.Compose([
    A.CoarseDropout(max_height=100, max_width=100, p = 1),
    A.RandomBrightnessContrast(p=0.9),
    A.HueSaturationValue(
                        hue_shift_limit=0.2,
                        sat_shift_limit=0.2,
                        val_shift_limit=0.2,
                        p=0.9,
                        )
])
chosen_image = cv2.imread(os.path.join(image_folder_path, "1ee6b9669.jpg"))[:,:,::-1]
augmented_image = transform(image=chosen_image)['image']
plt.imshow(augmented_image)

