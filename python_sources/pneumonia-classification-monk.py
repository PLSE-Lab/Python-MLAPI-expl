#!/usr/bin/env python
# coding: utf-8

# # [MONK](https://github.com/Tessellate-Imaging/monk_v1)

# Monk is a low code Deep Learning tool and a unified wrapper for Computer Vision.
# 
# **Monk Features**
# 
# * low-code
# * unified wrapper over major deep learning framework - keras, pytorch, gluoncv
# * syntax invariant wrapper
# 
# **Monk Enables**
# 
# * To create, manage and version control deep learning experiments.
# * To compare experiments across training metrics.
# * To quickly find best hyper-parameters.
# 
# 
# **Goals**
# 
# * To experiment with Models
# * Understand how easy is it to use Monk

# # Table of Contents

# * [Data Visualization](#dv)
# * [Installing MONK](#im)
# * [Importing Pytorch Backend](#ip)
# * [Creating and Managing experiments](#cm)
# * [List of available models](#list)
# * [Quick Mode Training - Load the data and the model](#qmt)
# * [Train the classifier](#tc)
# * [Running inference on test images](#ri)

# <div id="dv"> Data Visualization </div>

# In[ ]:


import cv2
import matplotlib.pyplot as plt
f, axarr = plt.subplots(2,2)
img1 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1008_virus_1691.jpeg')
img2 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person100_virus_184.jpeg')
img3 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0131-0001.jpeg')
img4 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0152-0001.jpeg')
axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)


# # <div id="im"> Install Monk </div>
# 
# git clone https://github.com/Tessellate-Imaging/monk_v1.git

# In[ ]:


get_ipython().system('git clone https://github.com/Tessellate-Imaging/monk_v1.git')


# cd monk_v1/installation/Linux && pip install -r requirements_cu9.txt
# 
# (Select the requirements file as per OS and CUDA version)
# 
# If using Colab install using the commands below
# 
# !cd monk_v1/installation/Misc && pip install -r requirements_colab.txt
# 
# If using Kaggle uncomment the following command
# 
# !cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt

# In[ ]:


get_ipython().system('cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt')


# In[ ]:


get_ipython().system(' pip install pillow==5.4.1')


# # Imports

# In[ ]:


# Monk
import os
import sys
sys.path.append("monk_v1/monk/");


# <div id="ip"> **Importing Pytorch Backend** </div>

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

# # <div id="cm"> Creating and managing experiments </div>
# 
# - Provide project name
# - Provide experiment name
# - For a specific data create a single project
# - Inside each project multiple experiments can be created
# - Every experiment can be have diferent hyper-parameters attached to it

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("PneumoniaClassificationMONK", "UsingPytorchBackend");


# 
# This creates files and directories as per the following structure
# workspace
# 
#     |
#     |--------PneumoniaClassificationMONK (Project name can be different)
#                     |
#                     |
#                     |-----UsingPytorchBackend (Experiment name can be different)
#                                 |
#                                 |-----experiment-state.json
#                                 |
#                                 |-----output
#                                         |
#                                         |------logs (All training logs and graphs saved here)
#                                         |
#                                         |------models (all trained models saved here)

# # <div id="list"> List of available models </div>

# In[ ]:


gtf.List_Models()


# # Training a Pneumonia Vs Normal image classifier

# # <div id="qmt"> Quick mode training </div>
# 
# - Using Default Function
#     - dataset_path
#     - model_name
#     - num_epochs
# 
# 
# Dataset folder structure
# parent_directory
# 
#     |
#     |
#     |------Pneumonia
#             |
#             |------img1.jpg
#             |------img2.jpg
#             |------.... (and so on)
#     |------Normal
#             |
#             |------img1.jpg
#             |------img2.jpg
#             |------.... (and so on) 

# In[ ]:


gtf.Default(dataset_path="/kaggle/input/chest-xray-pneumonia/chest_xray/train", 
            model_name="resnet50", 
            freeze_base_network=False,
            num_epochs=25); 


# # <div id="tc"> Training the classifier </div>

# In[ ]:


#Start Training
gtf.Train();
#Read the training summary generated once you run the cell and training is completed


# # Validating the trained classifier
# 
# 
# Load the experiment in validation mode
# 
# - Set flag eval_infer as True

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("PneumoniaClassificationMONK", "UsingPytorchBackend",eval_infer=True);


# # Load the validation dataset

# In[ ]:


gtf.Dataset_Params(dataset_path="/kaggle/input/chest-xray-pneumonia/chest_xray/val");
gtf.Dataset();


# # Run validation

# In[ ]:


accuracy, class_based_accuracy = gtf.Evaluate();


# # <div id="ri"> Running inference on test images </div>
# 
# 
# # Load the experiment in inference mode
# 
# - Set flag eval_infer as True

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("PneumoniaClassificationMONK", "UsingPytorchBackend",eval_infer=True);


# # Select image and Run inference

# In[ ]:


img_name = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0005-0001.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:


img_name = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person103_bacteria_489.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:


img_name = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person108_bacteria_506.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# Visualize and study comparison metrics

# Training v/s Validation Accuracy Curves

# In[ ]:


from IPython.display import Image
Image(filename="/kaggle/working/workspace/PneumoniaClassificationMONK/UsingPytorchBackend/output/logs/train_val_accuracy.png") 


# Training v/s Validation loss Curves

# In[ ]:


from IPython.display import Image
Image(filename="/kaggle/working/workspace/PneumoniaClassificationMONK/UsingPytorchBackend/output/logs/train_val_loss.png") 


# # Check out
# 
# [Monk_Object_Detection](https://github.com/Tessellate-Imaging/Monk_Object_Detection)
# 
# A one-stop repository for low-code easily-installable object detection pipelines.
# 
# and
# 
# [Monk_Gui](https://github.com/Tessellate-Imaging/Monk_Gui)
# 
# A Graphical user Interface for deep learning and computer vision over Monk Libraries
# 
# also
# 
# [Pytorch_Tutorial](https://github.com/Tessellate-Imaging/Pytorch_Tutorial)
# 
# A set of jupyter notebooks on pytorch functions with examples
