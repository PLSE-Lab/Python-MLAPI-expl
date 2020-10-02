#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import gc
from sklearn.utils import shuffle
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
ORIGINAL_HEIGHT = 137
ORIGINAL_WIDTH = 236
PROCESSED_HEIGHT = 128
PROCESSED_WIDTH = 128
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Define the size of a training file

# There are 200840=2^3*5*5021 Images. Because of the extremely large prime factor it is not possible to split them evenly and still get a convenient batch size (5021 makes my Kernel crash, 4 is ridiculously slow). Because of this, I define a size for each of the training files and the remainder is put into the last file, which is used for validation.

# In[ ]:


rows_per_file = 20084+1


# In[ ]:


print(f"Size of training files: {rows_per_file} elements per file")
remainder = 200840 % rows_per_file
print(f"Size of last file: {remainder} elements")
num_files = 200840 // rows_per_file
print(f"Number of training files: {num_files}")


# # Image Processing

# Inspired by Iafoss [popular Kernel](https://www.kaggle.com/iafoss/image-preprocessing-128x128). I use modified version, though.

# In[ ]:


def load_to_numpy(file):
    parquet = pd.read_parquet(file)
    return 255 - parquet.iloc[:,1:].values.astype(np.uint8).reshape(-1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH)

def normalize_image(img):
    return (img*(255.0/img.max())).astype(np.uint8)

def get_min_indices(img, min_writing_value=80):
    min_value = img > min_writing_value
    h_min, h_max = np.where(np.any(min_value, axis=0))[0][[0, -1]]
    v_min, v_max = np.where(np.any(min_value, axis=1))[0][[0, -1]]
    return (h_min, h_max, v_min, v_max)

def get_min_indices_with_border(img, min_writing_value=80, border=20):
    h_min, h_max, v_min, v_max = get_min_indices(img[border:-border, border:-border], min_writing_value=min_writing_value)
    return (h_min + border, h_max + border, v_min + border, v_max + border) #indices ignored border, is added again

def cut_and_denoise_image(img, border=20, min_writing_value=80, max_noise=28):
    #cut minimum needed to encease image
    h_min, h_max, v_min, v_max = get_min_indices_with_border(img, border=border, min_writing_value=min_writing_value)
    #add tolerance around minium, making it dependend on border prevents missing part of the character
    h_min= (h_min-border) if h_min>border else 0
    v_min= (v_min-border) if v_min>border else 0
    h_max= (h_max+border) if ORIGINAL_WIDTH-h_max>border else ORIGINAL_WIDTH
    v_max= (v_max+border) if ORIGINAL_HEIGHT-v_max>border else ORIGINAL_HEIGHT
    #cut image
    img = img[v_min:v_max, h_min:h_max]
    #denoise
    img[img < max_noise] = 0
    #add padding to image
    longer_side_length = max(np.ma.size(img, axis=0), np.ma.size(img, axis=1))
    padding = [((longer_side_length - np.ma.size(img, axis=0)) // 2,),
               ((longer_side_length - np.ma.size(img, axis=1)) // 2,)]
    img = np.pad(img, padding, mode="constant")
    #return resized image
    return cv2.resize(img,(PROCESSED_HEIGHT, PROCESSED_WIDTH))


# The images are ordered perfectly (proven in separate Kernel), so as long as they are correctly concatenated, everything will be fine

# In[ ]:


results = []
for file_index in range(0,4):
    print("Dealing with ", f"/kaggle/input/bengaliai-cv19/train_image_data_{file_index}.parquet")
    images = load_to_numpy(f"/kaggle/input/bengaliai-cv19/train_image_data_{file_index}.parquet")
    print("Number of images: ",np.ma.size(images, axis=0))
    print("Collect after loading parquet: ", gc.collect())
    for image_index, img in enumerate(images):
        img = normalize_image(img)
        img = cut_and_denoise_image(img)
        images[image_index,0:PROCESSED_HEIGHT, 0:PROCESSED_WIDTH] = img #saving inplace to save RAM
    images = images[:,0:PROCESSED_HEIGHT, 0:PROCESSED_WIDTH]
    print(images.shape)
    print("Collect after processing images: ", gc.collect())
    results.append(images)
    print(len(results))
    print("Collect after appending: ", gc.collect())


# In[ ]:


#put everything in one array
total = np.zeros([4*50210, PROCESSED_HEIGHT, PROCESSED_WIDTH], dtype=np.uint8)
for i in range(0, 4):
    total[i*50210:(i+1)*50210,:,:] = results[i]
    gc.collect()
del results
gc.collect()


# # Add labels

# In[ ]:


labels = pd.read_csv("/kaggle/input/bengaliai-cv19/train.csv").iloc[:,1:-1].astype(np.uint8)
gc.collect()


# In[ ]:


#shuffling to use last few images in validation file. Not stratified, however
total, labels = shuffle(total, labels, random_state=42)


# In[ ]:


#1-hot-encoding
root_label = pd.get_dummies(labels["grapheme_root"]).values
vowel_label = pd.get_dummies(labels["vowel_diacritic"]).values
consonant_label = pd.get_dummies(labels["consonant_diacritic"]).values
del labels


# # Save to files

# In[ ]:


#train files
for file_index in range(num_files):
    tmp = total[file_index*rows_per_file:(file_index+1)*rows_per_file]
    np.save(f"processed_{rows_per_file}_{PROCESSED_HEIGHT}_{file_index}.npy", tmp)
    tmp = root_label[file_index*rows_per_file:(file_index+1)*rows_per_file]
    np.save(f"root_{rows_per_file}_label_{file_index}.npy", tmp)
    tmp = vowel_label[file_index*rows_per_file:(file_index+1)*rows_per_file]
    np.save(f"vowel_{rows_per_file}_label_{file_index}.npy", tmp)
    tmp = consonant_label[file_index*rows_per_file:(file_index+1)*rows_per_file]
    np.save(f"consonant_{rows_per_file}_label_{file_index}.npy", tmp)
#valid file
tmp = total[num_files*rows_per_file:]
np.save(f"processed_{rows_per_file}_{PROCESSED_HEIGHT}_valid.npy", tmp)
tmp = root_label[num_files*rows_per_file:]
np.save(f"root_{rows_per_file}_label_valid.npy", tmp)
tmp = vowel_label[num_files*rows_per_file:]
np.save(f"vowel_{rows_per_file}_label_valid.npy", tmp)
tmp = consonant_label[num_files*rows_per_file:]
np.save(f"consonant_{rows_per_file}_label_valid.npy", tmp)

