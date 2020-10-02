#!/usr/bin/env python
# coding: utf-8

# There are plenty of datasets present over the internet.
# But there might be a situation where you may not find dataset of 
# your desired classes. So you may wish to make your own dataset. \
# <span style="color:red">*Presenting "custom image dataset using icrawler" to cop with the above problem.*</span> \
# You can use this kernel as a script to download images of your
# desired classes from the famous search engines.
# 
# I crawler is a mini framework of web crawlers.
# It also provides built-in crawlers for popular image sites like Flickr and search engines such as Google, Bing and Baidu. \
# For all the use cases refer the documentation https://icrawler.readthedocs.io/en/latest/index.html

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import cv2
import glob
from PIL import Image


# In[ ]:


# installing icrawler
get_ipython().system('pip install icrawler')


# In[ ]:


from icrawler.builtin import  (BingImageCrawler,GoogleImageCrawler)
import logging


# You may wish to make a image classifier for car and bike.
# So, let's use icrawler and download 100 images for both the classes.\
# Note: you can download maximum 1000 images because of search engine api restriction.

# In[ ]:


# downloading images of the car
# pass no. of maximum images in max_num arguement
directory_bing_car = './data/car'
search_filters_bing = dict()
bing_crawler = BingImageCrawler(downloader_threads=4,storage={'root_dir': directory_bing_car},log_level=logging.INFO)
bing_crawler.crawl(keyword='car', filters=search_filters_bing, max_num=100)

# downloading images of the car
# pass no. of maximum images in max_num arguement
directory_bing_bike = './data/bike'
search_filters_bing = dict()
bing_crawler = BingImageCrawler(downloader_threads=4,storage={'root_dir': directory_bing_bike},log_level=logging.INFO)
bing_crawler.crawl(keyword='bike', filters=search_filters_bing, max_num=100)


# In[ ]:


# you can see the images has been downloaded.
get_ipython().system("ls './data/car'")
get_ipython().system("ls './data/bike'")


# In[ ]:


# funtion for checking corrupted images.
def check_corrupted_images(image_folder_path):
    counter = 0
    for image in os.listdir(image_folder_path):
        try:
            img = cv2.imread(image)
        except:
            counter+=1
    return counter
            


# In[ ]:


print("Total number of corrupted images are: {}".format(check_corrupted_images(directory_bing_car)))
print("Total number of corrupted images are: {}".format(check_corrupted_images(directory_bing_bike)))
#If there are corrupted images use the below function to delete those images


# In[ ]:


#use this function to remove the corrupted files
def del_corrupted_images(image_folder_path):
    counter = 0
    for image in os.listdir(image_folder_path):
        try:
            img = cv2.imread(image)
        except:
            os.remove(image)
            print("File Removed!")
            counter+=1
    return counter


# <span style="color:red"> If you like the kernel then please upvote </span>
