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


import json
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


# # <div id="dv"> ** Exploratory Data Analysis ** </div>

# Data Visualization

# Viewing 4 Train images

# In[ ]:


f, axarr = plt.subplots(2,2)
img1 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_0.jpg')
img2 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_1.jpg')
img3 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_2.jpg')
img4 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_3.jpg')
axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)


# Viewing 4 Test images

# In[ ]:


f, axarr = plt.subplots(2,2)
img1 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Test_10.jpg')
img2 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Test_1005.jpg')
img3 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Test_101.jpg')
img4 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Test_1.jpg')
axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)


# In[ ]:


data_train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")


# In[ ]:


data_train.head()


# If there are multiple classes , the format that Monk accepts currently is
# 
# First column should contain image_id with extension .jpg or .png and the 
# second column must contain the label corresponding to its category.
# 
# Monk will internally convert to the above one hot encoded format

# So creating the required format

# In[ ]:


data_train_new = pd.DataFrame(columns = ['image_id', 'Category']) 


# In[ ]:


data_train_new.head()


# Adding extension .jpg to image_id and adding corresponding label to Category

# In[ ]:


for index,row in data_train.iterrows():
    #print(row)
    #print(index)
    pathname = str(row['image_id'])+'.jpg'
    data_train_new.loc[index,'image_id']=pathname
    if(row['healthy']==1):
        cat = 'healthy'
    elif(row['multiple_diseases']==1):
        cat = 'multiple_diseases'
    elif(row['rust']==1):
        cat = 'rust'
    else:
        cat = 'scab'
    
    data_train_new.loc[index,'Category']=cat


# In[ ]:


data_train_new.head()


# In[ ]:


data_train_new.to_csv("trainWithext.csv", index=False)


# # <div id="installingmonk"> **[Installing Monk](https://github.com/Tessellate-Imaging/monk_v1/tree/master/installation)** </div>

# * git clone https://github.com/Tessellate-Imaging/monk_v1.git
# 
# * cd monk_v1/installation/Linux && pip install -r requirements_cu9.txt
# 
# * (Select the requirements file as per OS and CUDA version)

# In[ ]:


get_ipython().system('git clone https://github.com/Tessellate-Imaging/monk_v1.git')


# * If using Colab install using the commands below
# 
# !cd monk_v1/installation/Misc && pip install -r requirements_colab.txt
# 
# * If using Kaggle uncomment the following command
# 
# #!cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
# 
# * Select the requirements file as per OS and CUDA version when using a local system or cloud
# 
# #!cd monk_v1/installation/Linux && pip install -r requirements_cu9.txt

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

# # <div id="cme"> Creating and managing experiments </div>
# 
# - Provide project name
# - Provide experiment name
# - For a specific data create a single project
# - Inside each project multiple experiments can be created
# - Every experiment can be have diferent hyper-parameters attached to it

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("PlantPathology2020", "Using_Pytorch_Backend");


# # This creates files and directories as per the following structure
# 
# 
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


gtf.Default(dataset_path="/kaggle/input/plant-pathology-2020-fgvc7/images/",
            path_to_csv="trainWithext.csv", # updated csv file 
            model_name="resnet18", 
            freeze_base_network=False,
            num_epochs=20); 


# <div id="edaM"> EDA in MONK </div>

# In[ ]:


gtf.EDA(check_corrupt=True)


# <div id="mod"> See what other models Monk's backend supports </div>

# In[ ]:


gtf.List_Models();


# # <div id="tc"> Train the classifier </div>

# In[ ]:


#Start Training
gtf.Train();
#Read the training summary generated once you run the cell and training is completed


# # <div id="inf"> **Running inference on test images** </div>

# Load the experiment in inference mode
# - Set flag eval_infer as True

# In[ ]:


gtf = prototype(verbose=0);
gtf.Prototype("PlantPathology2020", "Using_Pytorch_Backend", eval_infer=True);


# Accuracy Curve

# In[ ]:


from IPython.display import Image
Image(filename="workspace/PlantPathology2020/Using_Pytorch_Backend/output/logs/train_val_accuracy.png") 


# Loss Curve

# In[ ]:


from IPython.display import Image
Image(filename="workspace/PlantPathology2020/Using_Pytorch_Backend/output/logs/train_val_loss.png") 


# # Select image and Run inference

# In[ ]:


img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/Test_0.jpg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:


img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/Test_1004.jpg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:


img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/Test_10.jpg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# Running Inference on all test images

# In[ ]:


import pandas as pd
from tqdm import tqdm_notebook as tqdm
from scipy.special import softmax
#np.set_printoptions(precision=2)
df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")


# In[ ]:


img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/Test_10.jpg";
predictions = gtf.Infer(img_name=img_name,return_raw=True);

type(predictions)

predictions.keys()

print(predictions["raw"])

print(" Predictions in terms of probabilities")
print(softmax(predictions["raw"]))

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:


for i in tqdm(range(len(df))):
    img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/" + df["image_id"][i] + ".jpg";
    
    #Invoking Monk's nferencing engine inside a loop
    predictions = gtf.Infer(img_name=img_name, return_raw=True);
    x = predictions["raw"]
    out = softmax(x)
    df["healthy"][i] = out[0];
    df["multiple_diseases"][i] = out[1];
    df["rust"][i] = out[2];
    df["scab"][i] = out[3];


# In[ ]:


df.head()


# In[ ]:


df.to_csv("submission.csv", index=False);


# In[ ]:


get_ipython().system(' rm -r monk_v1')


# In[ ]:


get_ipython().system(' rm -r workspace')


# In[ ]:


get_ipython().system(' rm pylg.log trainWithext.csv')


# # Check out 
# 
# # [Monk_Object_Detection](https://github.com/Tessellate-Imaging/Monk_Object_Detection)
# 
# A one-stop repository for low-code easily-installable object detection pipelines.
# 
# and
# 
# # [Monk_Gui](https://github.com/Tessellate-Imaging/Monk_Gui)
# 
# A Graphical user Interface for deep learning and computer vision over Monk Libraries
# 
# also
# 
# # [Pytorch_Tutorial](https://github.com/Tessellate-Imaging/Pytorch_Tutorial)
# 
# A set of jupyter notebooks on pytorch functions with examples

# # **To contribute to Monk AI or Pytorch RoadMap repository raise an issue in the git-repo or DM us on linkedin** 

# * https://www.tessellateimaging.com/
# * Abhishek - https://www.linkedin.com/in/abhishek-kumar-annamraju/
# * Akash - https://www.linkedin.com/in/akashdeepsingh01/
