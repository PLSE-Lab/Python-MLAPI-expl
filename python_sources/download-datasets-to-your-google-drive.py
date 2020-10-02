#!/usr/bin/env python
# coding: utf-8

# # **This notebook is for linking, authenticating and downloading the Image data sets from json files directly into your Google Drive**

# 
# ###Features:
# 1.   Smaller version of images can be found by replacing "-large" with "-small" at the end of url. In this code, the small picture (around 5 kb on average) are downloaded.
# 2.   This script has been built for Colab users. As the instance gets destroyed every 12 hours, one can't download datasets everytime.
# 3.   Also, if you have a locally available GPU and want to download images to your drive (and then to your computer on one click), you can use this script. 
# 4.   This notebook can be used to download data to drive for any competition that gives urls in JSON files (ofcourse with little modifications).
# 
# 

# ###Instructions:
# 
# 
# 1.   Carefully read the comments mentioned. They are there for a reason!
# 2.   You can also add your code and modify it!
# 3.   Feedback is valuable guys. Lemme know what you thinking. 
# 
# 

# Well, lets get started...

# In[ ]:


#Linking drive to colab to store datasets
get_ipython().system('apt-get install -y -qq software-properties-common python-software-properties module-init-tools')
get_ipython().system('add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null')
get_ipython().system('apt-get update -qq 2>&1 > /dev/null')
get_ipython().system('apt-get -y install -qq google-drive-ocamlfuse fuse')


# In[ ]:


# Generate auth tokens for Colab
from google.colab import auth
auth.authenticate_user()


# In[ ]:


# Generate creds for the Drive FUSE library. Though the link asks you to verify twice, you don't have to!
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
get_ipython().system('google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL')
vcode = getpass.getpass()
get_ipython().system('echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}')


# In[ ]:


# Create a directory and mount Google Drive using that directory.
get_ipython().system('mkdir -p drive')
get_ipython().system('google-drive-ocamlfuse drive')

print 'Files in Drive:'
get_ipython().system('ls drive/')


# In[ ]:


# Create a file in a new directory called "Kaggle" in your Google Drive. This will be your operation base :P
get_ipython().system('echo "This newly created file will appear in your Drive file list. If you are reading this, that means the attempt to integrate was successful" > drive/kaggle/created.txt')


# *Tried integrating kaggle-api into colab's instance. But it's throwing error:401 (Unauthorized). So, you have to manually upload the json files to "kaggle" folder in your drive!
# *

# In[ ]:


#The uploaded files are in .zip format. The following code will unzip them into nice json files. This has to be done only the first time!
get_ipython().system('unzip "drive/kaggle/*.zip" -d drive/kaggle')

#Now, remove those archives
get_ipython().system('rm -f drive/kaggle/*.zip')

#Make directories for the data
get_ipython().system('mkdir drive/kaggle/train drive/kaggle/validation drive/kaggle/test')


# In[ ]:


#Now, to download the train set into your drive from the urls in the JSON files, execute the below.  Also, a file is generated with the ImageURL, imageName, imageId and 
#their labelIds.


import json
import time

train_data = json.load(open('drive/kaggle/train.json'))
get_ipython().system('echo ImageURL, ImgName, ImgId, LabelId >> drive/kaggle/train/train.txt')

for i in range(len(train_data['images'])):
  img_url = train_data['images'][i]['url']
  img_id = train_data['images'][i]['imageId']
  label_id = train_data['annotations'][i]['labelId']
  img_name=img_url.split("/")[-1]
  #print img_name
  img_name_actual = img_name.split("-")[0]
  img_name_small = img_name_actual + "-small"+".jpg"
  #print img_name_actual
  img_url_small = img_url.split("-")[-2]
  img_url_small = img_url_small + "-small"
  print img_url_small
  get_ipython().system('curl $img_url_small > drive/kaggle/train/$img_name_small')
  time.sleep(0.05) 
  get_ipython().system('echo $img_url_small,$img_name_small,$img_id,$label_id >> drive/kaggle/train/train.txt ')
  #time.sleep(0.5)


# In[ ]:


#To download validation data on to your drive...

import json
import time

val_data = json.load(open('drive/kaggle/validation.json'))
get_ipython().system('echo ImageURL, ImgName, ImgId, LabelId >> drive/kaggle/validation/validation.txt ')

for i in range(len(val_data['images'])):
  img_url = val_data['images'][i]['url']
  #print img_url
  img_id = val_data['images'][i]['imageId']
  #print img_id
  label_id = val_data['annotations'][i]['labelId']
  #print label_id
  img_name=img_url.split("/")[-1]
  #print img_name
  img_name_actual = img_name.split("-")[0]
  img_name_small = img_name_actual + "-small"+".jpg"
  #print img_name_actual
  img_url_small = img_url.split("-")[-2]
  img_url_small = img_url_small + "-small"
  print img_url_small
  get_ipython().system('curl $img_url_small > drive/kaggle/validation/$img_name_small')
  time.sleep(0.05)
  get_ipython().system('echo $img_name_actual,$img_id,$label_id >> drive/kaggle/validation/validation.txt ')
  #time.sleep(0.05)


# In[ ]:


#And this is for downloading test data into your drive

import json
import time

test_data = json.load(open('drive/kaggle/test.json'))
#print len(test_data['images'])

for i in range(len(test_data['images'])):
  img_url = test_data['images'][i]['url']
  #print img_url
  img_id = test_data['images'][i]['imageId']
  #print img_id
  img_name=img_url.split("/")[-1]
  #print img_name
  img_name_actual = img_name.split("-")[0]
  img_name_small = img_name_actual + "-small"+".jpg"
  #print img_name_actual
  img_url_small = img_url.split("-")[-2]
  img_url_small = img_url_small + "-small"
  print img_url_small
  get_ipython().system('curl $img_url_small > drive/kaggle/test/$img_name_small')
  time.sleep(0.05)
  


# ###Note
# 
# 1.   This is it for now. Later, I shall include EDA and hopefully the actual CV architecture part! But boy, those data sets are too hot to handle :P
# 
# 2.   And yes, too large too! Anyway, it seems really fun to play with this dataset! Good luck to everyone!!
# 
# **Upvote this kernel if you find it useful so that others can find it easily.**
