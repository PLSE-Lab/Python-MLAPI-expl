#!/usr/bin/env python
# coding: utf-8

# ## About Competition
# 
# Human brain research is among the most complex areas of study for scientists. We know that age and other factors can affect its function and structure, but more research is needed into what specifically occurs within the brain. With much of the research using MRI scans, data scientists are well positioned to support future insights. In particular, neuroimaging specialists look for measurable markers of behavior, health, or disorder to help identify relevant brain regions and their contribution to typical or symptomatic effects.
# 
# In this competition, you will predict multiple assessments plus age from multimodal brain MRI features. You will be working from existing results from other data scientists, doing the important work of validating the utility of multimodal features in a normative population of unaffected subjects. Due to the complexity of the brain and differences between scanners, generalized approaches will be essential to effectively propel multimodal neuroimaging research forward.
# 
# The Tri-Institutional Georgia State University/Georgia Institute of Technology/Emory University Center for Translational Research in Neuroimaging and Data Science (TReNDS) leverages advanced brain imaging to promote research into brain health. The organization is focused on developing, applying and sharing advanced analytic approaches and neuroinformatics tools. Among its software projects are the GIFT and FIT neuroimaging toolboxes, the COINS data management system, and the COINSTAC toolkit for federated learning, all aimed at supporting data scientists and other neuroimaging researchers.
# 
# Making the leap from research to clinical application is particularly difficult in brain health. In order to translate to clinical settings, research findings have to be reproduced consistently and validated in out-of-sample instances. The problem is particularly well-suited for data science, but current approaches typically do not generalize well. With this large dataset and competition, your efforts could directly address an important area of brain research.

# In[ ]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting

from pandas_profiling import ProfileReport


# In[ ]:


import os
bp = '/kaggle/input/trends-assessment-prediction'
print(os.listdir(bp))


# In[ ]:


print('Reading data...')
loading_data = pd.read_csv(bp+'/loading.csv')
train_data = pd.read_csv(bp+'/train_scores.csv')
sample_submission = pd.read_csv(bp+'/sample_submission.csv')
print('Reading data completed')

print('Size of loading_data', loading_data.shape)
print('Size of train_data', train_data.shape)
print('Size of sample_submission', sample_submission.shape)
print('test size:', len(sample_submission)/5)


# In[ ]:


ProfileReport(train_data)


# ## Sample submission

# In[ ]:


sample_submission.to_csv('sample_submission.csv', index=False)


# **More contents will be added soon**
