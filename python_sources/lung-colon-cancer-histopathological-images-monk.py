#!/usr/bin/env python
# coding: utf-8

# # [MONK](https://github.com/Tessellate-Imaging/monk_v1)

# **Monk is a low code Deep Learning tool and a unified wrapper for Computer Vision.**

# # About the dataset 
# 
# Original Article
# Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019
# 
# Relevant Links
# https://arxiv.org/abs/1912.12142v1
# https://github.com/tampapath/lung_colon_image_set
# 
# 
# Dataset BibTeX
# @article{,
# title= {LC25000 Lung and colon histopathological image dataset},
# keywords= {cancer,histopathology},
# author= {Andrew A. Borkowski, Marilyn M. Bui, L. Brannon Thomas, Catherine P. Wilson, Lauren A. DeLand, Stephen M. Mastorides},
# url= {https://github.com/tampapath/lung_colon_image_set}
# }
# 
# 
# This dataset contains 25,000 histopathological images with 5 classes. All images are 768 x 768 pixels in size and are in jpeg file format.
# The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and 500 total images of colon tissue (250 benign colon tissue and 250 colon adenocarcinomas) and augmented to 25,000 using the Augmentor package.
# There are five classes in the dataset, each with 5,000 images, being:
# 
# * Lung benign tissue
# * Lung adenocarcinoma
# * Lung squamous cell carcinoma
# * Colon adenocarcinoma
# * Colon benign tissue

# # Table of Contents

# * [Install Monk](#im)
# * [Import](#import)
# * Creating and managing experiments
#     * [For lung cancer](#cml)
#     * [For colon cancer](#cmc)
# * Training the classifier - Quick mode training
#     * [For lung cancer](#tl)
#     * [For colon cancer](#tc)
# * [List of available models](#lm)

# <div id='im'> Install MONK </div>

# In[ ]:


get_ipython().system('git clone https://github.com/Tessellate-Imaging/monk_v1.git')


# If using Colab install using the commands below
# 
# !cd monk_v1/installation/Misc && pip install -r requirements_colab.txt
# 
# If using Kaggle uncomment the following command
# 
# #!cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
# 
# Select the requirements file as per OS and CUDA version when using a local system or cloud
# 
# #!cd monk_v1/installation/Linux && pip install -r requirements_cu9.txt

# In[ ]:


get_ipython().system('cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt')


# # <div id="import"> Imports </div>

# In[ ]:


# Monk
import os
import sys
sys.path.append("monk_v1/monk/");


# In[ ]:


#Using pytorch backend 
from pytorch_prototype import prototype


# <div id="cml"> Creating and managing experiments for lung cancer </div>
# 
# 
# - Provide project name
# - Provide experiment name
# - For a specific data create a single project
# - Inside each project multiple experiments can be created
# - Every experiment can be have diferent hyper-parameters attached to it

# In[ ]:


gtfL = prototype(verbose=1);
gtfL.Prototype("Lung-cancer", "Using-pytorch-backend");


# This creates files and directories as per the following structure
# workspace
# 
#     |
#     |--------Lung-cancer (Project name can be different)
#                     |
#                     |
#                     |-----Using-pytorch-backend (Experiment name can be different)
#                                 |
#                                 |-----experiment-state.json
#                                 |
#                                 |-----output
#                                         |
#                                         |------logs (All training logs and graphs saved here)
#                                         |
#                                         |------models (all trained models saved here)

# # <div id="lm"> List of available models </div>

# In[ ]:


gtfL.List_Models()


# # <div id="tl"> Training the classifier for lung cancer </div>

# 
# Quick mode training
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
#     |------lung_n
#             |
#             |------img1.jpg
#             |------img2.jpg
#             |------.... (and so on)
#     |------lung_scc
#             |
#             |------img1.jpg
#             |------img2.jpg
#             |------.... (and so on)
#      |------lung_aca
#             |
#             |------img1.jpg
#             |------img2.jpg
#             |------.... (and so on)

# In[ ]:


gtfL.Default(dataset_path="/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/lung_image_sets", 
            model_name="resnet18", 
            num_epochs=5);

#Read the summary generated once you run this cell.


# In[ ]:


#Start Training
gtfL.Train();

#Read the training summary generated once you run the cell and training is completed


# In[ ]:


from IPython.display import Image
img_name = "/kaggle/working/workspace/Lung-cancer/Using-pytorch-backend/output/logs/train_val_loss.png"
Image(filename=img_name)


# In[ ]:


from IPython.display import Image
img_name = "/kaggle/working/workspace/Lung-cancer/Using-pytorch-backend/output/logs/train_val_accuracy.png"
Image(filename=img_name)


# <div id="cmc"> Creating and managing experiments for colon cancer </div>
# 
# 
# - Provide project name
# - Provide experiment name
# - For a specific data create a single project
# - Inside each project multiple experiments can be created
# - Every experiment can be have diferent hyper-parameters attached to it

# In[ ]:


gtfC = prototype(verbose=1);
gtfC.Prototype("Colon-cancer", "Using-pytorch-backend");


# This creates files and directories as per the following structure
# workspace
# 
#     |
#     |--------Colon-cancer (Project name can be different)
#                     |
#                     |
#                     |-----Using-pytorch-backend (Experiment name can be different)
#                                 |
#                                 |-----experiment-state.json
#                                 |
#                                 |-----output
#                                         |
#                                         |------logs (All training logs and graphs saved here)
#                                         |
#                                         |------models (all trained models saved here)

# # <div id="tc"> Training the classifier for colon cancer </div>

# 
# Quick mode training
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
#     |------colon_aca
#             |
#             |------img1.jpg
#             |------img2.jpg
#             |------.... (and so on)
#     |------colon_n
#             |
#             |------img1.jpg
#             |------img2.jpg
#             |------.... (and so on)  

# In[ ]:


gtfC.Default(dataset_path="/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/colon_image_sets", 
            model_name="resnet152", 
            num_epochs=5);

#Read the summary generated once you run this cell.", 


# In[ ]:


#Start Training
gtfC.Train();

#Read the training summary generated once you run the cell and training is completed


# In[ ]:


from IPython.display import Image
img_name = "/kaggle/working/workspace/Colon-cancer/Using-pytorch-backend/output/logs/train_val_accuracy.png"
Image(filename=img_name)


# In[ ]:


from IPython.display import Image
img_name = "/kaggle/working/workspace/Colon-cancer/Using-pytorch-backend/output/logs/train_val_loss.png"
Image(filename=img_name)

