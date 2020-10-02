#!/usr/bin/env python
# coding: utf-8

# The prediction has two stages.
# 
# 1. First we predict the correct class of image. 
# 1. Next we predict the correct class attribute of the image.
# 
# 

# # Predict the correct class of image.
# 
# The encoded pixels have overlapping class information. We are creating only one label for a image. The overlapping class information is lost.(Here I deviate from problem statement.)
# We train our model using resnet34. First on 224 \* 224 image. Then 512 \* 512 images.
# 
# This model can predict the class of pixel.
# But we still need to predict the attribute for correct submission.
# 

# **Train & predict class with 224 * 224 images.**
# 
# https://www.kaggle.com/nikhilikhar/fastai-imaterialist-224?scriptVersionId=15267678

# **Train & predict class with 512 * 512 images.**
# 
# https://www.kaggle.com/nikhilikhar/fastai-imaterialist-512?scriptVersionId=15344715
# 
# ![](https://www.kaggleusercontent.com/kf/15344715/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Qm6QeHm4GjTmzga9issP0g.TpI_j_vg40mH4oe__0RcultxIEoZU_kp7FLXBii_pbTVVnPrn9TazDNcjOp9xevBa8t54tNKWGf9kV2nb6dH5KPoWb6ujLMFRTkHXI7kyEZr6NfjQa3ngWXYT1-l-5J4bp__XpApK6kwXsEx4eoubw.zwCrE1BYd85ba4HHXjlwJw/__results___files/__results___19_0.png)

# # Predict the correct class attribute of the image.
# 
# From above 512 * 512 model we predict the correct class. Each test image has one or more class. 
# The result is stored in form of dataframe.
# 
# From the model we create a multi-label trainig data. Each image is showing all predicted classes. (Here I deviate from problem statement.)
# 
# We will train on this newly created data to find the correct attribute inside the images.
# 
# BG => no attribute
# 
# ![Sample Predicted labels](https://www.kaggleusercontent.com/kf/15346189/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..YG6sba3Jcmj9dvH87k8CtA.RvtOcdQGxT_GnWZ8a3uQloiAsjxTxWFosm1uyYwfbyesrTURZuH852JZh44ztftB1U60apeqJReE0et6iv-9JoQSJMBVXKn0iPybpuK3aW0tDOJWWCBqccT0rD4xtEMBHvOV9-uyDKfqiARMWJiA4ONVh39bl8n77lyRGMwec8Drl1MnM4rrNv9BkCGfETh9.gftdDrmyaQS4zwO3GvLHQA/__results___files/__results___10_0.png)

# **Train model for multi class label classification**
# 
# https://www.kaggle.com/nikhilikhar/fastai-imaterialist-multilabel-classification?scriptVersionId=15346189
# 
# This kernel was developed on small set of data. 
# 
# I wanted to train with more images and I met some of Kaggle kernel limitations (See below).
# 
# Non working kernel -> https://www.kaggle.com/nikhilikhar/fastai-imaterialist-multilabel-classification?scriptVersionId=15368548

# **Class and attribute prediction**
# 
# We use multiclass label classification model to predict the correct attribute. (Incomplete because above is working due to kaggle limitations.)
# 
# https://www.kaggle.com/nikhilikhar/fastai-imaterialist-multilabel-segmentation?scriptVersionId=15404255
# 

# # Kaggle Kernel Limitations
# 
# * Kaggle kernel can run for max 9 hrs.
# * Kaggle kernel has memory limit of [17179869184 byte approx 17Gb](https://www.kaggle.com/nikhilikhar/fastai-imaterialist-multi-label-data/log?scriptVersionId=15401077). Program will exit with code 137. I m not sure why I hit this limit. I was no where near this limit.
# * Kaggle kernel doesn't support more than 500 output file. This happen when I tired to [create test label separetly](https://www.kaggle.com/nikhilikhar/fastai-imaterialist-multi-label-data/log?scriptVersionId=15397908) to avoid memory limit.
# * Kaggle was not able to create a new process when using `preds,y = learn.get_preds(ds_type=DatasetType.Test)`. And it was giving [memory error](https://www.kaggle.com/nikhilikhar/fastai-imaterialist-multilabel-segmentation?scriptVersionId=15371701#L210).
# 
# 

# # Kaggle resources
# I found these kernels useful in developing the solution.
# 
# * https://www.kaggle.com/pednoi/training-mask-r-cnn-to-be-a-fashionista-lb-0-07
# * https://www.kaggle.com/go1dfish/u-net-baseline-by-pytorch-in-fgvc6-resize
# * https://www.kaggle.com/solpaul/fastai-factory-approach-using-mask-images

# In[ ]:




