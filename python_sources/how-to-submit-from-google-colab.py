#!/usr/bin/env python
# coding: utf-8

# Kaggle kernel is obviously great. You can access almost all the libraries to be used for data analysis without explicitly installing them. You don't have to donwload the data such that you don't have to suffer from memory shortage of your local machine. 
# 
# Yet sometimes especially when a competition is close to the end, kernel becomes unstable;(
# 
# One great solution to avoid your flustration is to use **google colab ** in the rainy day. Google colab allows you to perform data analysis almost like you do in Kaggle kernels: load data on cloud, perform EDA, train and test your models, and even submit to Kaggle!
# 
# In this kernel, I show you how to do it in your google colab. I use the data in this MNIST competiion as an example.

# # To Start Google Colaboratory ...
# You need to have a google account. Then
# 
# 
# ### - open your google drive
# ### - click (+new) button on the left top or right click --> More --> Colaboratory
# 
# ![](https://i.ibb.co/jR3cpVN/open-colab.png)
# 
# You can already start your google colab! Thanks google, this is so easy:D
# 

# # Last thing you need: Kaggle.json
# To use data in kaggle on Google colab, you need to make an API in kaggle. To do it,
# 
# ### - go to Kaggle
# ### - click your account (on the right top) --> My Account
# ### - click "Create New API Token" to download kaggle.json
# 
# ![](https://i.ibb.co/HFycVz3/MINST-API.png)
# 
# Once you donwload a kaggle.json, go back to your google colab. 
# 
# Please follow the link down below, which leads to my google colab showing steps from loading data to submission!
# 
# 

# [how to use google colab for kaggle competition](https://colab.research.google.com/drive/17MsV4f8Bap8y2MU8BcKBNXRdTNE2ftP3)

# # Want GPU?
# If you want to use GPU, then on google colab
# 
# ### - edit --> notebook setting --> change "hardware accelerator" to GPU 
# 
# ![](https://i.ibb.co/dr4SMNy/colab-gpu.png)
# 
# Sorry for Japanese but you know what is located where:)
