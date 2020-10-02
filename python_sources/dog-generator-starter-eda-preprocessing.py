#!/usr/bin/env python
# coding: utf-8

# This notebook is basically my attempt to introduce myself to the dataset a little bit by getting the full dog image dataset and the label (annotation) for every image. This in turn will allow me to analyze the class distributions for each dog breed, as well plot the dog images.
# 
# Finally, since there are images with multiple dogs in them, I will attempt to separate each dog (with cropping) using the provided annotations that contain the coordinates of each object.

# ## References

# 1. Xml parsing and cropping to specified bounding box - (https://www.kaggle.com/paulorzp/show-annotations-and-breeds)
# 
# 2. Also thanks to K.Amano for his cropping method with interpolation - (https://www.kaggle.com/amanooo/wgan-gp-keras)

# ## Importing libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import glob
import math
import random
import time
import datetime
from collections import defaultdict
from tqdm import tqdm, tqdm_notebook

import xml.etree.ElementTree as ET 

import cv2

print(os.listdir("../input"))


# ## Setting some constants according to the rules

# In[ ]:


image_width = 64
image_height = 64
image_channels = 3
image_sample_size = 10000
image_output_dir = '../output_images/'
image_input_dir = '../input/all-dogs/all-dogs/'
image_ann_dir = "../input/annotation/Annotation/"


# ## Storing the dog breed type in a dictionary by mapping it to its code in the annotations

# Here I've created the dictionary ```dog_breed_dict```, which will map each breed code to its original name.

# In[ ]:


dog_breed_dict = {}
for annotation in os.listdir(image_ann_dir):
    annotations = annotation.split('-')
    dog_breed_dict[annotations[0]] = annotations[1]


# In[ ]:


print(dog_breed_dict['n02097658'])


# ## Creating another dictionary (of lists), in order to map each input filename to a specific dog breed

# The following helper-function will allow to create another similar dictionary but for all of the input images. Since each dog breed can be found multiple times in the dataset, the value part here will be stored as a list.

# In[ ]:


def get_input_image_dict(image_input_dir, labels_dict):
    image_sample_dict = defaultdict(list)
    for image in os.listdir(image_input_dir):
        filename = image.split('.')
        label_code = filename[0].split('_')[0]
        breed_name = labels_dict[label_code]
        #print('Code: {}, Breed: {}'.format(label_code, breed_name))
        if image is not None:
            image_sample_dict[breed_name].append(image)
    
    print('Created label dictionary for input images.')
    return image_sample_dict


# In[ ]:


image_sample_dict = get_input_image_dict(image_input_dir, dog_breed_dict)


# ## Plotting the class distributions of the input images by dog breed

# After we have this information, we can do some EDA. Firstly, we can print the total amount of dog breeds in the dataset and we can also count the total input images using the created ```image_sample_dict```. Secondly, we can plot the class distributions of each breed.

# In[ ]:


def plot_class_distributions(image_sample_dict, title=''):
    class_lengths = []
    labels = []
    total_images = 0
    
    print('Total amount of dog breeds: ', len(image_sample_dict))
    
    for label, _ in image_sample_dict.items():
        total_images += len(image_sample_dict[label])
        class_lengths.append(len(image_sample_dict[label]))
        labels.append(label)
        
    print('Total amount of input images: ', total_images)
        
    plt.figure(figsize = (10,30))
    plt.barh(range(len(class_lengths)), class_lengths)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.ylabel('Dog Breed')
    plt.xlabel('Sample size')
    plt.show()
    
    return total_images


# In[ ]:


total_images = plot_class_distributions(image_sample_dict)


# ## Visualizing the images

# Using the <b>OpenCV</b> imaging library, we can load each image and create an example set of features.

# In[ ]:


def read_image(src):
    img = cv2.imread(src)
    if img is None:
        raise FileNotFoundError
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Using the previously created ```image_sample_dict``` we can plot a matrix of images to see what the dogs look like, along with their actual breed name.

# In[ ]:


def plot_images(directory, image_sample_dict, examples=25, disp_labels=True): 
  
    if not math.sqrt(examples).is_integer():
        print('Please select a valid number of examples.')
        return
    
    imgs = []
    classes = []
    for i in range(examples):
        rnd_class, _ = random.choice(list(image_sample_dict.items()))
        #print(rnd_class)
        rnd_idx = np.random.randint(0, len(image_sample_dict[rnd_class]))
        filename = image_sample_dict[rnd_class][rnd_idx]
        img = read_image(os.path.join(directory, filename))
        imgs.append(img)
        classes.append(rnd_class)
    
    
    fig, axes = plt.subplots(round(math.sqrt(examples)), round(math.sqrt(examples)),figsize=(15,15),
    subplot_kw = {'xticks':[], 'yticks':[]},
    gridspec_kw = dict(hspace=0.3, wspace=0.1))
    
    for i, ax in enumerate(axes.flat):
        if disp_labels == True:
            ax.title.set_text(classes[i])
        ax.imshow(imgs[i])


# In[ ]:


plot_images(image_input_dir, image_sample_dict)


# We can see quite a few images with more than one dog present in the samples.

# In[ ]:


plot_images(image_input_dir, image_sample_dict, examples=36, disp_labels=True)


# There are people present in a few samples as well.

# ## Cropping the images according to the annotations

# The following preprocessing function is used to create the actual features for the future model. The ```dog_images_np``` are first initialized with zeros with a fixed sample size of 25000 images. Each image from the dataset is read and information about the image objects is gathered from the <b>xml</b> file representing each image. 
# 
# After gathering the information on the coordinates, the input image is either cropped, shrunk or expanded, depending on the position of the object and finally, is stored to the array of features. I will also scale the input pixel values of the features to the range [-1, 1], due to the fact the future model will probably be using a <b> tanh </b>activation function.

# In[ ]:


def load_cropped_images(dog_breed_dict=dog_breed_dict, image_ann_dir=image_ann_dir, sample_size=25000, 
                        image_width=image_width, image_height=image_height, image_channels=image_channels):
    curIdx = 0
    breeds = []
    dog_images_np = np.zeros((sample_size,image_width,image_height,image_channels))
    for breed_folder in os.listdir(image_ann_dir):
        for dog_ann in tqdm(os.listdir(image_ann_dir + breed_folder)):
            try:
                img = read_image(os.path.join(image_input_dir, dog_ann + '.jpg'))
            except FileNotFoundError:
                continue
                
            tree = ET.parse(os.path.join(image_ann_dir + breed_folder, dog_ann))
            root = tree.getroot()
            
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            objects = root.findall('object')
            for o in objects:
                bndbox = o.find('bndbox') 
                
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                xmin = max(0, xmin - 4)        # 4 : margin
                xmax = min(width, xmax + 4)
                ymin = max(0, ymin - 4)
                ymax = min(height, ymax + 4)

                w = np.min((xmax - xmin, ymax - ymin))
                w = min(w, width, height)                     # available w

                if w > xmax - xmin:
                    xmin = min(max(0, xmin - int((w - (xmax - xmin))/2)), width - w)
                    xmax = xmin + w
                if w > ymax - ymin:
                    ymin = min(max(0, ymin - int((w - (ymax - ymin))/2)), height - w)
                    ymax = ymin + w
                
                img_cropped = img[ymin:ymin+w, xmin:xmin+w, :]      # [h,w,c]
                # Interpolation method
                if xmax - xmin > image_width:
                    interpolation = cv2.INTER_AREA          # shrink
                else:
                    interpolation = cv2.INTER_CUBIC         # expansion
                    
                img_cropped = cv2.resize(img_cropped, (image_width, image_height), 
                                         interpolation=interpolation)  # resize
                    
                dog_images_np[curIdx,:,:,:] = np.asarray(img_cropped)
                dog_breed_name = dog_breed_dict[dog_ann.split('_')[0]]
                breeds.append(dog_breed_name)
                curIdx += 1
                
    dog_images_np = dog_images_np / 255.  # change the pixel range to [-1, 1]
    return dog_images_np, breeds


# In[ ]:


start_time = time.time()
dog_images_np, breeds = load_cropped_images()
est_time = round(time.time() - start_time)
print("Feature loading time: {}.".format(str(datetime.timedelta(seconds=est_time))))


# After we have gathered the features and their labels, we can easily plot them once again to confirm that they have been cropped correctly and that there aren't multiple dogs in a single image.

# In[ ]:


def plot_features(features, labels, image_width=image_width, image_height=image_height, 
                image_channels=image_channels,
                examples=25, disp_labels=True): 
  
    if not math.sqrt(examples).is_integer():
        print('Please select a valid number of examples.')
        return
    
    imgs = []
    classes = []
    for i in range(examples):
        rnd_idx = np.random.randint(0, len(labels))
        imgs.append(features[rnd_idx, :, :, :])
        classes.append(labels[rnd_idx])
    
    
    fig, axes = plt.subplots(round(math.sqrt(examples)), round(math.sqrt(examples)),figsize=(15,15),
    subplot_kw = {'xticks':[], 'yticks':[]},
    gridspec_kw = dict(hspace=0.3, wspace=0.01))
    
    for i, ax in enumerate(axes.flat):
        if disp_labels == True:
            ax.title.set_text(classes[i])
        ax.imshow(imgs[i])


# In[ ]:


print('Loaded features shape: ', dog_images_np.shape)
print('Loaded labels: ', len(breeds))


# The total dog images seem to be 20579 with 22125 separately cropped dogs.

# In[ ]:


print('Plotting cropped images by specified coordinates..')
plot_features(dog_images_np, breeds, examples=16, disp_labels=True)


# In[ ]:


plt.imshow(dog_images_np[3])

