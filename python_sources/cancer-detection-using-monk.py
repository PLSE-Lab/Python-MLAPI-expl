#!/usr/bin/env python
# coding: utf-8

# # [Monk Library](https://github.com/Tessellate-Imaging/monk_v1)
# 
# 
# Monk is an opensource low-code tool for computer vision and deep learning
# 
# **Monk features**
# 
# * low-code
# * unified wrapper over major deep learning framework - keras, pytorch, gluoncv
# * syntax invariant wrapper
# 
# **Monk Enables**
# 
# * users to create, manage and version control deep learning experiments
# * users to compare experiments across training metrics
# * users to quickly find best hyper-parameters

# # ** Table of contents **

# * [DATASET](#data)
# * [Install Monk](#install)
# * [Creating and Managing experiments](#cm)

# # <div id="data"> Details on the dataset !!! </div>

# The original DATASET is https://www.kaggle.com/c/histopathologic-cancer-detection/data

# The modified dataset contains images of original dataset is renamed , cropped to central 32*32 region , converted to gray and .jpeg format.It is cropped because in the original dataset a positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. 
# 
# 20% of train images of original dataset is in validation folder
# 80% of train images of original dataset is in train folder
# All images of test folder of original dataset is in test folder
# 
# Both train and validation folder of this modified dataset has CANCER and Normal folder
# images are put to this CANCER and Normal folder based on the label provided in the train_labels.csv of the original dataset.
#          

# More information on Histopathologic Cancer Detection-Identify metastatic tissue in histopathologic scans of lymph node sections https://www.cancer.org/cancer/cancer-basics/lymph-nodes-and-cancer.html

# # <div id="install"> Install Monk </div>

# git clone https://github.com/Tessellate-Imaging/monk_v1.git
# 
# * If using Colab install using the commands below
# 
# !cd monk_v1/installation/Misc && pip install -r requirements_colab.txt
# 
# * If using Kaggle uncomment the following command
# 
# !cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
# 
# * Select the requirements file as per OS and CUDA version when using a local system or cloud
# 
# !cd monk_v1/installation/Linux && pip install -r requirements_cu9.txt
# (Select the requirements file as per OS and CUDA version)

# In[ ]:


get_ipython().system('git clone https://github.com/Tessellate-Imaging/monk_v1.git')


# In[ ]:


get_ipython().system('cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt')


# In[ ]:


get_ipython().system(' pip install pillow==5.4.1')


# Imports

# In[ ]:


# Monk
import os
import sys
sys.path.append("monk_v1/monk/");


# In[ ]:


#Using keras backend 
from keras_prototype import prototype


# * To use mxnet-gluon backend
# 
# from gluon_prototype import prototype
# 
# 
# * To use pytorch backend
# 
# from pytorch_prototype import prototype

# <div id="cm"> Creating and managing experiments</div>
# - Provide project name
# - Provide experiment name
# - For a specific data create a single project
# - Inside each project multiple experiments can be created
# - Every experiment can be have diferent hyper-parameters attached to it

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("Cancer-Detection-Using-MONK", "Using-Keras-Backend");


# 
# This creates files and directories as per the following structure
# 
# 
# workspace
# 
#     |
#     |--------Cancer-Detection-Using-MONK (Project name can be different)
#                     |
#                     |
#                     |-----Using-Keras-Backend (Experiment name can be different)
#                                 |
#                                 |-----experiment-state.json
#                                 |
#                                 |-----output
#                                         |
#                                         |------logs (All training logs and graphs saved here)
#                                         |
#                                         |------models (all trained models saved here)

# 
# # Quick mode training
# 
# - Using Default Function
#     - dataset_path
#     - model_name
#     - num_epochs
# 

# Dataset folder structure
# 
# parent_directory
# 
#     |
#     |
#     |------CANCER
#             |
#             |------img1.jpeg
#             |------img2.jpeg
#             |------.... (and so on)
#     |------Normal
#             |
#             |------img1.jpeg
#             |------img2.jpeg
#             |------.... (and so on)

# In[ ]:


gtf.List_Models()


# In[ ]:


gtf.Default(dataset_path="/kaggle/input/cancer/train", 
            model_name="resnet152_v2", 
            num_epochs=10);

#Read the summary generated once you run this cell.


# In[ ]:


gtf.update_save_intermediate_models(False);
gtf.Reload();


# In[ ]:


#Start Training
gtf.Train();
#Read the training summary generated once you run the cell and training is completed


# Validating the trained classifier

# Load the experiment in validation mode
# - Set flag eval_infer as True

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("Cancer-Detection-Using-MONK", "Using-Keras-Backend", eval_infer=True);


# Load the validation dataset

# In[ ]:


gtf.Dataset_Params(dataset_path="/kaggle/input/cancer/validation");
gtf.Dataset();


# # Run validation

# In[ ]:


accuracy, class_based_accuracy = gtf.Evaluate();


# # Running inference on test images

# Load the experiment in inference mode
# - Set flag eval_infer as True

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("Cancer-Detection-Using-MONK", "Using-Keras-Backend", eval_infer=True);


# Select image and Run inference

# In[ ]:


img_name = "/kaggle/input/cancer/test/c2 (10004).jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:


img_name = "/kaggle/input/cancer/test/c2 (10).jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:


img_name = "/kaggle/input/cancer/test/c2 (10014).jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# # Train Validation Accuracy Curve

# In[ ]:


from IPython.display import Image
Image(filename="workspace/Cancer-Detection-Using-MONK/Using-Keras-Backend/output/logs/train_val_accuracy.png") 


# # Train Validation Loss Curve

# In[ ]:


from IPython.display import Image
Image(filename="workspace/Cancer-Detection-Using-MONK/Using-Keras-Backend/output/logs/train_val_loss.png") 


# # Repeating using pytorch backend with a different model

# In[ ]:


#Using pytorch backend 
from pytorch_prototype import prototype


# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("Cancer-Detection-Using-MONK", "Using-Pytorch-Backend");


# In[ ]:


gtf.List_Models();


# In[ ]:


gtf.Default(dataset_path="/kaggle/input/cancer/train", 
            model_name="densenet201",
            num_epochs=10);

#Read the summary generated once you run this cell.


# In[ ]:


# Need not save intermediate epoch weights
gtf.update_save_intermediate_models(False);
gtf.Reload();


# In[ ]:


#Start Training
gtf.Train();

#Read the training summary generated once you run the cell and training is completed


# In[ ]:


# Compare experiments


# In[ ]:


# Invoke the comparison class
from compare_prototype import compare


# **Creating and managing comparison experiments**
# 
#   - Provide project name

# In[ ]:


# Create a project 
gtf = compare(verbose=1);
gtf.Comparison("Campare-backends");


# **Add experiments**

# In[ ]:


gtf.Add_Experiment("Cancer-Detection-Using-MONK", "Using-Keras-Backend");
gtf.Add_Experiment("Cancer-Detection-Using-MONK", "Using-Pytorch-Backend");


# **Run Analysis**

# In[ ]:


gtf.Generate_Statistics();


# Visualize and study comparison metrics
# 

# Training Accuracy Curves

# In[ ]:


from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/train_accuracy.png") 


# Training Loss Curves

# In[ ]:


from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/train_loss.png") 


# Validation Accuracy Curves

# In[ ]:


from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/val_accuracy.png") 


# Validation loss curves

# In[ ]:


from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/val_loss.png")


# Training time curves

# In[ ]:


from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/stats_training_time.png") 


# Best Validation accuracies

# In[ ]:


from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/stats_best_val_acc.png") 


# In[ ]:


get_ipython().system(' rm -r monk_v1')


# In[ ]:


get_ipython().system(' rm -r workspace')


# In[ ]:


get_ipython().system(' rm pylg.log')

