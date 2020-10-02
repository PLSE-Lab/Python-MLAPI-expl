#!/usr/bin/env python
# coding: utf-8

# ****Introduction****
# 
# **Objective**
# **We want to analyze Bollywood songs using ML techniques like LSA, LDA and guided LDA to answer questions like: **
# 
# 
# 1) How did music genres/topics evolve over time?
# 
# 2) Is there a favorite/dominant genre of each lyricist?
# 
# 3) If there a favorite genre for singer?
# 
# This analysis is aimed for a fun learning experience of some basic NLP techniques.
# 
# **Tasks**
# 
# 1) Dataset: We have already downloaded a complete dataset of songs since 1931 till 2019...We will share it on Kaggle soon.
# 
# 2) Text Pre processing: Stemming, Spelling Corrections, Normalizations, etc..
# 
# 3) Apply LSA/LDA/Guided LDA
# 
# **Relevant articles**
# LSA
# LDA
# Guided LDA
# 
# **Final Results/ Visualizations**
# 
# **Future Work**
# Word2Vec/Glove?
# RNN?
# Stock Market News/Earnings Transcripts/

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

