#!/usr/bin/env python
# coding: utf-8

# # [MONK](https://github.com/Tessellate-Imaging/monk_v1)

# Install Monk
# 
# git clone https://github.com/Tessellate-Imaging/monk_v1.git
# 
# cd monk_v1/installation/Linux && pip install -r requirements_cu9.txt
# 
# (Select the requirements file as per OS and CUDA version)

# In[ ]:


get_ipython().system('git clone https://github.com/Tessellate-Imaging/monk_v1.git')


# If using Colab install using the commands below
# 
# !cd monk_v1/installation/Misc && pip install -r requirements_colab.txt
# 
# If using Kaggle uncomment the following command
# 
# !cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
# 
# Select the requirements file as per OS and CUDA version when using a local system or cloud
# 
# !cd monk_v1/installation/Linux && pip install -r requirements_cu9.txt

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


#Using keras-gluon backend 
from keras_prototype import prototype


# Creating and managing experiments
# 
# - Provide project name
# - Provide experiment name
# - For a specific data create a single project
# - Inside each project multiple experiments can be created
# - Every experiment can be have diferent hyper-parameters attached to it

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("Covid19", "Using-keras-backend");


# 
# This creates files and directories as per the following structure
# workspace
# 
#     |
#     |--------Covid19 (Project name can be different)
#                     |
#                     |
#                     |-----Using-keras-backend (Experiment name can be different)
#                                 |
#                                 |-----experiment-state.json
#                                 |
#                                 |-----output
#                                         |
#                                         |------logs (All training logs and graphs saved here)
#                                         |
#                                         |------models (all trained models saved here)

# List of available models

# In[ ]:


gtf.List_Models()


# # Training  image classifier

# Quick mode training
# 
# - Using Default Function
#     - dataset_path
#     - model_name
#     - num_epochs

# Dataset folder structure
# parent_directory
# 
#     |
#     |
#     |------Covid
#             |
#             |------img1.jpg
#             |------img2.jpg
#             |------.... (and so on)
#     |------Normal
#             |
#             |------img1.jpg
#             |------img2.jpg
#             |------.... (and so on) 
#     |------Viral Pneumonia
#             |
#             |------img1.jpg
#             |------img2.jpg
#             |------.... (and so on) 

# In[ ]:


gtf.Default(dataset_path="/kaggle/input/covid19-image-dataset/Covid19-dataset/train", 
            model_name="densenet201", 
            num_epochs=25);

#Read the summary generated once you run this cell.


# In[ ]:


#Start Training
gtf.Train();

#Read the training summary generated once you run the cell and training is completed


# # Validating the trained classifier

# Load the experiment in validation mode
# 
# - Set flag eval_infer as True

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("Covid19", "Using-keras-backend", eval_infer=True);


# Load the validation dataset

# In[ ]:


gtf.Dataset_Params(dataset_path="/kaggle/input/covid19-image-dataset/Covid19-dataset/test");
gtf.Dataset();


# Run validation

# In[ ]:


accuracy, class_based_accuracy = gtf.Evaluate();


# # Running inference on test images

# Load the experiment in inference mode
# 
# - Set flag eval_infer as True

# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("Covid19", "Using-keras-backend", eval_infer=True);


# Select image and Run inference

# In[ ]:


img_name = "/kaggle/input/covid19-image-dataset/Covid19-dataset/test/Covid/0108.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:


img_name = "/kaggle/input/covid19-image-dataset/Covid19-dataset/test/Normal/0121.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:


img_name = "/kaggle/input/covid19-image-dataset/Covid19-dataset/test/Viral Pneumonia/0115.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)


# In[ ]:




