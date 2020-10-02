#!/usr/bin/env python
# coding: utf-8

# **PyTorch Classification Model Based On Imgaug library generation process.**
# 
# **IMGAUG library:** [GitHub](https://github.com/aleju/imgaug)
# 
# The idea of the issue focused on the far-reaching extension of the collection of X-ray images contained in the Pneumonia X-Rays dataset available on the Kaggle platform. Then the data subjected to additional generation and augmentation served to train and validate the model of the convolutionary neural network.
# 
# * **Model Brief:**
# * **Data download process, unziping and clearing useless data.**
# * **Libraries import, definition of required classes and methods.**
# * **Samples import, files compability verification and new samples generation process based on imgaug library:**
# In this section preprocessing of the images were performed. Basic dataset contains almost 6000 pictures. 
# Number of images was multiplied to ~25000 by sequential operations based on imgaug library. 
# To diversify augumentation process, random values of the function parameters were used in each function call.
# 
# * **Training pictures visualisation:**
# After sequential generation and augumentation of train and test samples, pictures were presented as below:
# ![](https://camo.githubusercontent.com/5a0a6bcde8e71f253e5c525f1894431b3092d317/68747470733a2f2f692e6962622e636f2f337372526752442f706f6272616e652e706e67)
# 
# * **Parameters definition, images to tensors transformation, and model definition:**
# Images size were set on 224x224 with 1 channel, batch_size was equal 16, two output classes called: 'NORMAL' and 'PNEUMONIA'. Model was trained on 20 epochs.
# During the transform process img -> tensor extra options of PyTorch transforms library were used.
# 
# * **Model Summary and training process:**
# Model contains 6 convolution layers, 2 pooling functions, 3 linear fully connected layers, relu functions as an activator. Dropouts were used to prevent overfitting. 
# ![](https://camo.githubusercontent.com/ef42ca7aa08129da9d38883cb13ae96beeb0eb77/68747470733a2f2f692e6962622e636f2f52685a48594d442f6d6f64656c2e706e67)
# 
# * **Training results visualisation:**
# After each epoch loss and accuracy value were collected and presented as it follows:
# ![](https://camo.githubusercontent.com/a06657cc0869e75df7151f814e197935bcfe63ea/68747470733a2f2f692e6962622e636f2f365264396437762f706f6272616e652d312e706e67)
# 
# **Accuracy validated on Test Dataset was equal ~84%**
# 
# 
# 
# [Entire Code Of The Model](https://nbviewer.jupyter.org/github/JMcsLk/PneumoniaXRayPyTorch/blob/master/PyTorch_X_Rays_Conv2D.ipynb)
# 
# If you find it helpful and interesting - please let me know!
