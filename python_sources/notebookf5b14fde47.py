#!/usr/bin/env python
# coding: utf-8

# This is my first effort on Kaggle.
# In this by using different 30 columns we have to predict the column Diagnosis which is having two stage of cancer M (Malignant ) and B (Begin)
# In this analysis I have used Basic Machine learning Algorithm to predict these 
# but with a detailed explanation.
# It is good for Beginner like as me .
# Lets Start
#  

# In[ ]:


# first we will import some libraries used for the this anlysis
import pandas as pd # This is used for data manipulation like as SQl, and data preprocessing and CSV I/0
import numpy as np # Used for linear algebra 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

