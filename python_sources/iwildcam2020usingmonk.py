#!/usr/bin/env python
# coding: utf-8

# #  <div id="MONK">  ** [MONK](https://github.com/Tessellate-Imaging/monk_v1)** </div>

# *Monk is a low code Deep Learning tool and a unified wrapper for Computer Vision.*

# **Monk Features**

# * low-code
# * unified wrapper over major deep learning framework - keras, pytorch, gluoncv
# * syntax invariant wrapper
# 

# **Monk Enables**

# 1. To create, manage and version control deep learning experiments.
# 2. To compare experiments across training metrics.
# 3. To quickly find best hyper-parameters.
# 

# Goals

# - To experiment with Models
# - Understand how easy is it to use Monk

# # **Table of Contents**

# * [MONK](#MONK)
# * [Exploratory Data Analysis/ Data Visualization](#dv)
# * [Installing Monk](#installingmonk)
# * [Importing Pytorch Backend](#pyb)
# * [Creating and Managing experiments](#cme)
# * [Quick Mode Training - Load the data and the model](#train)
# * [EDA Using Monk](#edaM)
# * [See what other models Monk's backend supports](#mod)
# * [Train the classifier](#tc)
# * [Running inference on test images](#inf)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # <div id="dv"> ** Exploratory Data Analysis ** </div>

# Data Visualization

# In[ ]:


import json
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


# Viewing 4 images in train folder

# In[ ]:


f, axarr = plt.subplots(2,2)
img1 = cv2.imread('/kaggle/input/iwildcam-2020-fgvc7/train/96b00332-21bc-11ea-a13a-137349068a90.jpg')
img2 = cv2.imread('/kaggle/input/iwildcam-2020-fgvc7/train/879d74d8-21bc-11ea-a13a-137349068a90.jpg')
img3 = cv2.imread('/kaggle/input/iwildcam-2020-fgvc7/train/9017f7aa-21bc-11ea-a13a-137349068a90.jpg')
img4 = cv2.imread('/kaggle/input/iwildcam-2020-fgvc7/train/90d93c58-21bc-11ea-a13a-137349068a90.jpg')
axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)


# Viewing 4 images in test folder

# In[ ]:


f, axarr = plt.subplots(2,2)
img1 = cv2.imread('/kaggle/input/iwildcam-2020-fgvc7/test/86761d58-21bc-11ea-a13a-137349068a90.jpg')
img2 = cv2.imread('/kaggle/input/iwildcam-2020-fgvc7/test/86767820-21bc-11ea-a13a-137349068a90.jpg')
img3 = cv2.imread('/kaggle/input/iwildcam-2020-fgvc7/test/86763c0c-21bc-11ea-a13a-137349068a90.jpg')
img4 = cv2.imread('/kaggle/input/iwildcam-2020-fgvc7/test/867665c4-21bc-11ea-a13a-137349068a90.jpg')
axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)


# *Converting Json to required CSV format*

# In[ ]:


with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as json_file:
    train_annotations_json = json.load(json_file)


# In[ ]:


train_annotations_json.keys()


# In[ ]:


df_annotations = pd.DataFrame(train_annotations_json["annotations"])


# In[ ]:


df_annotations.head()


# In[ ]:


data_train = df_annotations[["image_id","category_id"]].copy()


# In[ ]:


data_train.head()


# *Adding extension .jpg to image_id*

# In[ ]:


for index,row in data_train.iterrows():
    #print(row)
    #print(index)
    pathname = str(row['image_id'])+'.jpg'
    data_train.loc[index,'image_id']=pathname


# In[ ]:


data_train.shape


# In[ ]:


# Total number of images excluding the corrupted ones
len(data_train.index)


# In[ ]:


data_train.head()


# In[ ]:


data_train.tail()


# In[ ]:


#data_train.image_id.astype('str')
#data_train.category_id.astype('str')


# In[ ]:


data_train.info()


# In[ ]:


data_train.to_csv("train.csv", index=False)


# In[ ]:


with open(r'/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_test_information.json') as json_file:
    test_information_json = json.load(json_file)


# In[ ]:


test_information_json.keys()


# In[ ]:


test_information_images = pd.DataFrame(test_information_json["images"])


# In[ ]:


test_information_images.head()


# In[ ]:


data_test = test_information_images[["file_name","id"]].copy()


# In[ ]:


data_test.head()


# In[ ]:


data_test.rename(columns={'id':'Id'},inplace=True)
# Adding a new column
data_test['Category'] = 0 # 0 is the default value


# In[ ]:


data_test.head()


# # <div id="installingmonk"> **[Installing Monk](https://github.com/Tessellate-Imaging/monk_v1/tree/master/installation)** </div>

# In[ ]:


get_ipython().system('git clone https://github.com/Tessellate-Imaging/monk_v1.git')


# In[ ]:


get_ipython().system('cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt')


# *Imports*

# In[ ]:


# Monk
import os
import sys
sys.path.append("monk_v1/monk/");


# <div id="pyb"> *Using Pytorch backend* </div>

# In[ ]:


#Using pytorch backend 
from pytorch_prototype import prototype


# * To use mxnet backend
# 
# from gluon_prototype import prototype
# 
# * To use keras backend
# 
# from keras_prototype import prototype

# <div id="cme"> Creating and managing experiments </div>
# - Provide project name
# - Provide experiment name
# - For a specific data create a single project
# - Inside each project multiple experiments can be created
# - Every experiment can be have diferent hyper-parameters attached to it

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("iWildCam2020", "Using_Pytorch_Backend");


# This creates files and directories as per the following structure
# workspace
# 
# 
#     |
#     |--------iWildCam2020 (Project name can be different)
#                     |
#                     |
#                     |-----Using_Pytorch_Backend (Experiment name can be different)
#                                 |
#                                 |-----experiment-state.json
#                                 |
#                                 |-----output
#                                         |
#                                         |------logs (All training logs and graphs saved here)
#                                         |
#                                         |------models (all trained models saved here)

# # Load the data and the model

# Docs on  quick mode loading of data and model: https://github.com/Tessellate-Imaging/monk_v1#4
# 
# Tutorials on Monk: https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/1_getting_started_roadmap

# <div id="train"> Quick mode training </div>
# - Using Default Function
#     - dataset_path
#     - model_name
#     - num_epochs

# In[ ]:


gtf.Default(dataset_path="/kaggle/input/iwildcam-2020-fgvc7/train/",
            path_to_csv="train.csv", # updated csv file 
            model_name="resnet18", 
            freeze_base_network=False,
            num_epochs=10); 


# In[ ]:


gtf.Dataset_Percent(20);


# In[ ]:


gtf.Default(dataset_path="/kaggle/input/iwildcam-2020-fgvc7/train/",
            path_to_csv="/kaggle/working/sampled_dataset_train.csv", # updated csv file 
            model_name="resnet18", 
            freeze_base_network=False,
            num_epochs=1);


# <div id="edaM"> EDA in MONK </div>

# In[ ]:


gtf.EDA(check_corrupt=True)


# <div id="mod"> See what other models Monk's backend supports </div>

# In[ ]:


gtf.List_Models();


# # <div id="tc"> Train the classifier </div>

# In[ ]:


get_ipython().system('pip install pillow')


# In[ ]:


import PIL
print('PIL',PIL.__version__)


# In[ ]:


#Start Training
gtf.Train();
#Read the training summary generated once you run the cell and training is completed


# # <div id="inf"> **Running inference on test images** </div>

# Load the experiment in inference mode
# - Set flag eval_infer as True

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("iWildCam2020", "Using_Pytorch_Backend", eval_infer=True);


# Select image and Run inference

# In[ ]:


img_name = "/kaggle/input/iwildcam-2020-fgvc7/test/867611a0-21bc-11ea-a13a-137349068a90.jpg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:


img_name = "/kaggle/input/iwildcam-2020-fgvc7/test/8676382e-21bc-11ea-a13a-137349068a90.jpg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# Running Inference on all test images

# In[ ]:


from tqdm import tqdm_notebook as tqdm
from scipy.special import softmax

for i in tqdm(range(len(data_test))):
    #img_name = "/kaggle/input/iwildcam-2020-fgvc7/test/" + data_test["Id"][i] + ".jpg";
    img_name = "/kaggle/input/iwildcam-2020-fgvc7/test/" + data_test["file_name"][i];
    #Invoking Monk's nferencing engine inside a loop
    prediction = gtf.Infer(img_name=img_name, return_raw=True);
    data_test.loc[i,"Category"] = prediction;
    


# In[ ]:


data_test_new = data_test[["Id","Category"]].copy() # as per the format given ,the csv file must contain Id and Category
data_test_new.to_csv("submission.csv", index=True);


# In[ ]:


get_ipython().system(' rm -r monk_v1')


# In[ ]:


get_ipython().system(' rm -r workspace')


# In[ ]:


get_ipython().system(' rm pylg.log train.csv')


# # **To contribute to Monk AI or Pytorch RoadMap repository raise an issue in the git-repo or DM us on linkedin** 

# * https://www.tessellateimaging.com/
# * Abhishek - https://www.linkedin.com/in/abhishek-kumar-annamraju/
# * Akash - https://www.linkedin.com/in/akashdeepsingh01/

# In[ ]:




