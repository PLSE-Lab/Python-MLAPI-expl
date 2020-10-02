#!/usr/bin/env python
# coding: utf-8

# ## The purpose of this notebook
# 
# The discussion topic: 
# 
# [An easy and convenient way to look up GCS bucket addresses!](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165739)
# 
# For those of us who are using tfrecords and doing most of their training on Colab, it is very important to have an easy way to look up the Google Cloud Storage (GCS) bucket addresses of various public data sets. It is not enough to do it just once -- you have to update the addresses every few days because they change. This notebook provides a simple and convenient tool for identifying the most current GCS bucket addresses -- the addresses are printed to the screen every time you run the notebook.
# 
# The notebook includes the competition data, the tfrecords datasets made by me, the tfrecords datasets made by [Chris Deotte](https://www.kaggle.com/cdeotte), and the dataset storing [advanced hair augmentation]() images made by [Roman]().

# In[ ]:


from kaggle_datasets import KaggleDatasets

n=65 # used just for decoration


# ## Competition data

# In[ ]:


print(KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification'))


# ## Alexey Pronin's work
# 
# For more information see the following discussion topics: 
# 
# [TF/ TPU: from .tfrec to cross-validation. Step-by-step](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/158395)
# 
# [Implementation of Stratified Group K-Folds](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/156002)
# 
# 
# [Shades of Gray prepossessed data (both JPEG's and tfrecords)](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/161719)

# In[ ]:


print("1024x1024, train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-1024-train-part-1'))
print(KaggleDatasets().get_gcs_path('siim-tfrec-1024-train-part-2'))
print(KaggleDatasets().get_gcs_path('siim-tfrec-1024-part-3'))

print("")
print("="*n)

print("\n1024x1024, test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-1024-test'))


# In[ ]:


print("768x768, train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-768-train-part-1'))
print(KaggleDatasets().get_gcs_path('siim-tfrec-768-train-part-2'))

print("")
print("="*n)

print("\n768x768, test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-768-test'))


# In[ ]:


print("768x768, Color constant data train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-768-train-part-1'))
print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-768-train-part-2'))

print("")
print("="*n)

print("\n768x768, Color constant data external (ISIC 2019), only train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-768-external'))

print("")
print("="*n)

print("\n768x768, Color constant data test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-768-test'))


# In[ ]:


print("640x640, Color constant data train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-640-train'))

print("")
print("="*n)

print("\n640x640, Color constant data external (ISIC 2019), only train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-640-external'))

print("")
print("="*n)

print("\n640x640, Color constant data test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-640-test'))


# In[ ]:


print("512x512, train:\n")

print(KaggleDatasets().get_gcs_path('siim-512x512-tfrec-q95'))

print("")
print("="*n)

print("\n512x512, External data (ISIC 2019), only train:\n")

print(KaggleDatasets().get_gcs_path('siim-external-data-isic-2019-tfrec-512'))

print("")
print("="*n)

print("\n512x512, test:\n")

print(KaggleDatasets().get_gcs_path('siim-512x512-tfrec-q95-test'))


# In[ ]:


print("512x512, Color constant data train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-512-train'))

print("")
print("="*n)

print("\n512x512, Color constant data external (ISIC 2019), only train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-512-external'))

print("")
print("="*n)

print("\n512x512, Color constant data test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-512-test'))


# In[ ]:


print("384x384, train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-384-train'))

print("")
print("="*n)

print("\n384x384, test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-384-test'))


# In[ ]:


print("384x384, Color constant data train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-384-train'))

print("")
print("="*n)

print("\n384x384, Color constant data external (ISIC 2019), only train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-384-external'))

print("")
print("="*n)

print("\n384x384, Color constant data test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-384-test'))


# In[ ]:


print("256x256, train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-256-train'))

print("")
print("="*n)

print("\n256x256, test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-256-test'))


# In[ ]:


print("256x256, Color constant data train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-256-train'))

print("")
print("="*n)

print("\n256x256, Color constant data external (ISIC 2019), only train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-256-external'))

print("")
print("="*n)

print("\n256x256, Color constant data test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-256-test'))


# In[ ]:


print("224x224, train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-224-train'))

print("")
print("="*n)

print("\n224x224, test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-224-test'))


# In[ ]:


print("128x128, Color constant data train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-128-train'))

print("")
print("="*n)

print("\n128x128, Color constant data external (ISIC 2019), only train:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-128-external'))

print("")
print("="*n)

print("\n128x128, Color constant data test:\n")

print(KaggleDatasets().get_gcs_path('siim-tfrec-cc-128-test'))


# ## Chris Deotte's work
# 
# For more information see the following discussion topics: 
# 
# [Triple Stratified Leak-Free KFold CV](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526)
# 
# [How To Use Last Years 2019 Comp Data (and 2018, 2017)](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164910)

# In[ ]:


print("Chris, 1024x1024, train and test:\n")

print(KaggleDatasets().get_gcs_path('melanoma-1024x1024'))


# In[ ]:


print("Chris, 768x768, train and test:\n")

print(KaggleDatasets().get_gcs_path('melanoma-768x768'))

print("")
print("="*n)

print("\nChris, 768x768, external data:\n")

print(KaggleDatasets().get_gcs_path('isic2019-768x768'))


# In[ ]:


print("Chris, 512x512, train and test:\n")

print(KaggleDatasets().get_gcs_path('melanoma-512x512'))

print("")
print("="*n)

print("\nChris, 512x512, external data:\n")

print(KaggleDatasets().get_gcs_path('isic2019-512x512'))


# In[ ]:


print("Chris, 384x384, train and test:\n")

print(KaggleDatasets().get_gcs_path('melanoma-384x384'))

print("")
print("="*n)

print("\nChris, 384x384, external data:\n")

print(KaggleDatasets().get_gcs_path('isic2019-384x384'))


# In[ ]:


print("Chris, 256x256, train and test:\n")

print(KaggleDatasets().get_gcs_path('melanoma-256x256'))

print("")
print("="*n)

print("\nChris, 256x256, external data:\n")

print(KaggleDatasets().get_gcs_path('isic2019-256x256'))


# In[ ]:


print("Chris, 192x192, train and test:\n")

print(KaggleDatasets().get_gcs_path('melanoma-192x192'))

print("")
print("="*n)

print("\nChris, 192x192, external data:\n")

print(KaggleDatasets().get_gcs_path('isic2019-192x192'))


# In[ ]:


print("Chris, 128x128, train and test:\n")

print(KaggleDatasets().get_gcs_path('melanoma-128x128'))

print("")
print("="*n)

print("\nChris, 128x128, external data:\n")

print(KaggleDatasets().get_gcs_path('isic2019-128x128'))


# ## Roman's work
# 
# For more information see the following discussion topics: 
# 
# [Advanced hair augmentation](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159176)
# 
# [Advanced hair augmentation in TensorFlow](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/163909)
# 
# and the following public notebook:
# 
# [SIIM: Advanced Hair Augmentation in TensorFlow](https://www.kaggle.com/graf10a/siim-advanced-hair-augmentation-in-tensorflow)

# In[ ]:


print("Roman, hairs files (originally designed for 256x256 image size):\n")

print(KaggleDatasets().get_gcs_path('melanoma-hairs'))

