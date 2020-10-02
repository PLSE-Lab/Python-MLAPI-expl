#!/usr/bin/env python
# coding: utf-8

# # Use Both Image and Tabular Data
# 
# Kaggle's Melanoma Classification competition provides both **image data** and **tabular data** about each sample. Our task is to use both types of data to predict the probability that a sample is malignant. How can we build a model that uses both **images** and **tabular data**?
# 
# Three ideas come to mind.
# 
# * Build a CNN image model and find a way to input the tabular data into the CNN image model
# * Build a Tabular data model and find a way to extract image embeddings and input into the Tabular data model
# * Build 2 separate models and ensemble
# 
# In this notebook, we explore the third idea. A model that uses only image data is [here][1] and scores LB 0.910. A model that uses only tabular data is [here][2] and scores LB 0.700. We will make a simple ensemble of the two and thus utilize both the provided image and provided tabular data. The resultant LB score is 0.915 demonstrating that each type of data adds additional information.
# 
# To learn more ideas about how to build models that utilize both image and tabular data or to share your ideas, please participate in the Kaggle discussion [here][3].
# 
# 
# [1]: https://www.kaggle.com/ajaykumar7778/melanoma-tpu-efficientnet-b5-dense-head
# [2]: https://www.kaggle.com/titericz/simple-baseline
# [3]: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155251

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


image_sub = pd.read_csv('../input/melanomapreds/submissionImage.csv')
tabular_sub = pd.read_csv('../input/melanomapreds/submissionTabular.csv')
tabular_sub.head()


# In[ ]:


sub = image_sub.copy()
sub.target = 0.9 * image_sub.target.values + 0.1 * tabular_sub.target.values
sub.to_csv('submission.csv',index=False)


# In[ ]:


plt.hist(sub.target,bins=100)
plt.ylim((0,100))
plt.show()

