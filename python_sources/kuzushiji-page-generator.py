#!/usr/bin/env python
# coding: utf-8

# # Kuzushiji Page Generator
# ### This notebook introduces a new way to automatically generate pages filled with Kuzushiji symbols using the great [KMNIST dataset](https://github.com/rois-codh/kmnist). The aim is to help pretraining models learning both character detection and classification at the same time before moving to the pages from the original competition dataset. Of course, the resulting pages would not make any sense!
# 
# ### I have generated 20,000 pages using this script and made them available in the [following dataset](https://www.kaggle.com/frlemarchand/synthetic-kmnist-pages).
# 
# ### If you find this notebook useful, please feel free to give it an upvote!

# version 9 notes:
# * add variation in the symbol size within the same page
# * fix bug in the generation of the groundtruth .csv file
# * the groundtruth .csv file now follow the exact same format as the [Kuzushiji competition dataset](https://www.kaggle.com/c/kuzushiji-recognition/data)

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import random
from PIL import Image
from tqdm import tqdm


# # Extract datasets from the compressed files and load them

# In[ ]:


get_ipython().system('unzip ../input/kuzushiji/k49-train-imgs.npz && mv ../working/arr_0.npy k49-train-imgs.npy')
get_ipython().system('unzip ../input/kuzushiji/k49-train-labels.npz && mv ../working/arr_0.npy k49-train-labels.npy')
get_ipython().system('unzip ../input/kuzushiji/k49-train-imgs.npz && mv ../working/arr_0.npy k49-test-imgs.npy')
get_ipython().system('unzip ../input/kuzushiji/k49-train-labels.npz && mv ../working/arr_0.npy k49-test-labels.npy')


# In[ ]:


get_ipython().system('unzip ../input/kuzushiji/kmnist-train-imgs.npz && mv ../working/arr_0.npy kmnist-train-imgs.npy')
get_ipython().system('unzip ../input/kuzushiji/kmnist-train-labels.npz && mv ../working/arr_0.npy kmnist-train-labels.npy')
get_ipython().system('unzip ../input/kuzushiji/kmnist-train-imgs.npz && mv ../working/arr_0.npy kmnist-test-imgs.npy')
get_ipython().system('unzip ../input/kuzushiji/kmnist-train-labels.npz && mv ../working/arr_0.npy kmnist-test-labels.npy')


# In[ ]:


os.listdir("../input/kuzushiji")


# In[ ]:


k49 = np.load('../working/k49-train-imgs.npy')
k49_labels = np.load('../working/k49-train-labels.npy')
k49_mapping = pd.read_csv("../input/kuzushiji/k49_classmap.csv")


# In[ ]:


kmnist = np.load('../working/kmnist-train-imgs.npy')
kmnist_labels = np.load('../working/kmnist-train-labels.npy')
kmnist_mapping = pd.read_csv("../input/kuzushiji/kmnist_classmap.csv")


# # Functions to generate a symbol from one of the three subsets

# In[ ]:


def get_k49(show=False):
    idx = random.randint(0,len(k49)-1)
    img = k49[idx]
    if show:
        plt.imshow(img)
        plt.show()
    return img, k49_mapping.iloc[k49_labels[idx]].codepoint


# In[ ]:


sample = get_k49(show=True)


# In[ ]:


def get_kmnist(show=False):
    idx = random.randint(0,len(kmnist)-1)
    img = kmnist[idx]
    if show:
        plt.imshow(img)
        plt.show()
    return img, kmnist_mapping.iloc[kmnist_labels[idx]].codepoint


# In[ ]:


sample = get_kmnist(show=True)


# In[ ]:


def get_kuzushiji_kanji(show=False):
    kanji_list = os.listdir("../input/kuzushiji/kkanji/kkanji2/")
    selected_kanji = random.choice(kanji_list)
    image_list = os.listdir("../input/kuzushiji/kkanji/kkanji2/"+selected_kanji)
    selected_image = random.choice(image_list)
    image=cv2.imread("../input/kuzushiji/kkanji/kkanji2/{}/{}".format(selected_kanji,selected_image))
    if show:
        plt.imshow(image)
        plt.show()
    return image, selected_kanji


# In[ ]:


sample = get_kuzushiji_kanji(show=True)


# # Create a synthetic Kuzushiji-filled page

# In[ ]:


def get_new_page(page_dimensions = (3900,2400), binary_mask=True):
    
    page = np.zeros(page_dimensions)
    labels = ""
    
    number_of_columns = random.randint(3,8)
    symbols_per_columns = random.randint(10,20)
    margin = random.randint(30,200)
    symbol_size = random.randint(100,250)
    
    for row in range(1,symbols_per_columns):
        for col in range(1,number_of_columns):
            x_location = int((page.shape[1]-margin*2)*col/number_of_columns)
            y_location = int((page.shape[0]-margin*2)*row/symbols_per_columns)
            symbol_size_variation = random.randint(0,10)
            #randomly pick a subtype from the KMNIST dataset.
            condition = random.randint(1,3)
            if condition==1:
                symbol, label = get_kmnist()
                symbol = cv2.resize(symbol, (symbol_size-symbol_size_variation, symbol_size-symbol_size_variation)) 
                if binary_mask:
                    ret,symbol = cv2.threshold(symbol.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                page[y_location:y_location+symbol_size-symbol_size_variation,x_location:x_location+symbol_size-symbol_size_variation] = symbol
            elif condition==2:
                symbol, label = get_k49()
                symbol = cv2.resize(symbol, (symbol_size-symbol_size_variation, symbol_size-symbol_size_variation)) 
                if binary_mask:
                    ret,symbol = cv2.threshold(symbol.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                page[y_location:y_location+symbol_size-symbol_size_variation,x_location:x_location+symbol_size-symbol_size_variation] = symbol
            elif condition==3:
                symbol, label = get_kuzushiji_kanji()
                symbol = cv2.resize(symbol, (symbol_size-symbol_size_variation, symbol_size-symbol_size_variation)) 
                symbol = symbol[:,:,0]
                if binary_mask:
                    ret,symbol = cv2.threshold(symbol.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                page[y_location:y_location+symbol_size-symbol_size_variation,x_location:x_location+symbol_size-symbol_size_variation] = symbol
            #Bug fixed in version 9. 
            labels += "{} {} {} {} {} ".format(label,str(x_location),str(y_location),str(symbol_size-symbol_size_variation),str(symbol_size-symbol_size_variation))

    return page, labels


# In[ ]:


def print_random_pages():
    sample_number = 10
    fig = plt.figure(figsize = (20,sample_number))
    for i in range(0,sample_number):
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(get_new_page()[0])
    plt.tight_layout()
    plt.show()


# # Examples of binary images generated

# In[ ]:


print_random_pages()


# In[ ]:


dest_dir = "../working/synthetic-kmnist-pages"
os.mkdir(dest_dir)


# The dataset is saved in the working directory, as well as the dataframe into a .csv file. The number of generated pages is set to only 100 due to hard drive space limitation.

# In[ ]:


kuzushiji_df = pd.DataFrame(columns=["image_id","labels"])
#The number of output files is limited to 500
number_of_pages = 400
with tqdm(total=number_of_pages) as pbar:
    for idx in range(0,number_of_pages):
        pbar.update(1)
        filename = "{}.png".format(idx)
        binary_image, labels = get_new_page()
        cv2.imwrite("{}/{}".format(dest_dir, filename), binary_image)
        kuzushiji_df = kuzushiji_df.append({"image_id":filename,"labels":labels}, ignore_index=True)
kuzushiji_df.to_csv("{}/synthetic_kmnist_pages.csv".format(dest_dir),index=False)


# In[ ]:


kuzushiji_df.head()


# ### Please check out the [generated dataset](https://www.kaggle.com/frlemarchand/synthetic-kmnist-pages) with 20,000 synthetic pages. If you use the dataset to improve a solution, I would love to hear about it. :)
# 
# ### If this contribution was of any help, please give an upvote to help me know whether this type of kernel is useful to the community!
