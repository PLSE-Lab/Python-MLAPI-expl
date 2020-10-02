#!/usr/bin/env python
# coding: utf-8

# ## CNN - It is a special architecture of ANN (Artificial Neural Networks) proposed by Yann Lacun in 1988. It is mainly used in image classification. 

# **Image Classification : Accepting an input image and defining its final class. While classifying an image, the image is passed through a series of Convolutional, Non Linear, Pooling layers and then the output is generated.**

# Steps:
# 1. Convolution
# 1. Max Pooling
# 1. Flattening
# 1. Full Conncetion

# **Convolution : An imput image is taken and a shorter matrix is formed. This matrix is genrally a feature detector matrix from which a feature map is created.**
# 
# -> For example as shown in figure a 3x3 matrix is picked up from given matrix and each value is multiplied with itself to extract a feature map finally. 
# 
# ![img1.png](attachment:img1.png)
# 
# Many feture maps are created to obtain the first convolution layer. 
# 
# ![img2.png](attachment:img2.png)
# 

# Pooling: It preserves the feature of an image as well as lists only main features (reduces image size and also reduces number of parameters), basically prevents overfitting. 
# 
# 
# Many pooling techniques are availables : 
# * Max Pooling
# * Min Pooling
# * Sum Pooling
# * Avg Pooling
# 
# But generally max pooling is more efficient in case of feature extraction since it preserve highlighted features.
# 
# Max Pooling: Here a max value is selected from each section. For example as shown in figure below from each square a max value is extracted to get pooled feature map.
# 
# ![img3.jpg](attachment:img3.jpg)
# 

# Flattening : It is performed to pass pooled feature map to neural network one by one. 
# For example as shown in figure below flattening is done on a pooled feature map to convert it to a vertical list.
# 
# ![img%204.png](attachment:img%204.png)

# Full Connection: Whole ANN is added to CNN(Ann is added after performing previous steps)
# In CNN fully connected layers are used. These layers help to classify images (by activating required neurons responsible for each class. 
# 
# For example as shown below: neurons related to cat were activated after applying a full connection.
# 
# ![img5.jpg](attachment:img5.jpg)

# CNN mainly deals with vision and recognition.

# Combined: A complete setup for classifying an input image using CNN  is shown below.
# 
# ![img5.png](attachment:img5.png)
# 
# Gereally multiple convolution and pooling steps are performed over the image to get higly corelated features only. 
# 
# 

# ### ReLU: It is an activation function which is mainly used during the convolution steps. 
# ### Why ReLU ? -> It is used to increase non-linearity in the network. Since images are itself non-linear hence the activation function is applied. 
# 
# ### fi(x) = max(x,0) -> Clarified linear boundaries of an image more effectively. 

# # Please upvote if you like this tutorial. Thanks !! 

# In[ ]:




