#!/usr/bin/env python
# coding: utf-8

# # Ten class images classification - CIFAR-10 - Acc 87%
# ### **Yassine Ghouzam, PhD**
# #### 22/07/2017
# 
# # Du to web access restriction on kaggle kernels , i can't run this notebook on kaggle but you can view it 
# # Please check my ipython notebook at <a href="http://nbviewer.jupyter.org/gist/YassineGhouzam/e4536eae104a770c715045ae1b9f3894"> http://nbviewer.jupyter.org/gist/YassineGhouzam/e4536eae104a770c715045ae1b9f3894</a>
# 
# * **1. Introduction**
# * **2. Data preparation**
#     * 2.1 Load data
#     * 2.2 Normalization
#     * 2.3 Reshape
#     * 2.4 Label encoding
#     * 2.5 Split training and valdiation set
# * **3. CNN**
#     * 3.1 Define the model
#     * 3.2 Set the optimizer and annealer
#     * 3.3 Data augmentation
# * **4. Evaluate the model**
#     * 4.1 Training and validation curves
#     * 4.2 Confusion matrix
# * **5. Prediction and submition**
#     * 5.1 Predict and Submit results

# # 1. Introduction
# 
# This is a sequential Convolutional Neural Network for image classification trained on CIFAR-10 dataset.
# 
# CIFAR-10  is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
# 
# I choosed to build a CNN with keras API (Tensorflow backend) which is very intuitive. Firstly, I will prepare the data (Images belonging to 10 categories of object) then ill focus on the CNN modeling and evaluation.
# 
# I achieved 87% of accuracy with this CNN trained in 5h on a single CPU (I5 2500k). For those who have a >= 3.0 GPU capabilites (GTX 650 - Recent GPUs), you can use tensorflow-gpu with keras. Computation will be much much faster !!!
# 
# **For computational reasons i set the number of steps (epochs) to 2, if you want to achieve 87+% of accuracy set it to 80.**
# 
# This Notebook follows three main parts:
# 
# * The data preparation
# * The CNN modeling and evaluation
# * The results prediction and submission
# 
# 
# 
# 
# <img src="https://kaggle2.blob.core.windows.net/competitions/kaggle/3649/media/cifar-10.png"></img>
