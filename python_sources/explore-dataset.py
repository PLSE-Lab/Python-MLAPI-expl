#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


main_df = pd.read_csv("../input/KaggleV2-May-2016.csv")
main_df.head()


# **Data Visualizations:**
# * Age with frequency of no shows and show ups
# * Gender with no shows and show ups
# * sms_recieved , scholarship and hipertension diabetes and so on
# 
# 
# **Cleaning up Data:**
# * FIrst we need to set Textual categorical variables to numerical categorical variables
# * Male => 1; Female=>0
# * No_show => 1; Show_up=>0
# 

# In[ ]:


sns.distplot(main_df[main_df['No-show'] == 'No']['Age'],color='b',hist=False)
sns.distplot(main_df[main_df['No-show'] == 'Yes']['Age'],color='y',hist=False)
plt.savefig('age_hist.png')


# In[ ]:


def mkBarChart(column):
    plt.figure()
    sns.countplot(x="No-show", data=main_df, hue=column,palette="Blues_d")
    plt.savefig(column+".png")
    
mkBarChart("Gender")
mkBarChart("SMS_received")
mkBarChart("Scholarship")
mkBarChart("Hipertension")
mkBarChart("Diabetes")
mkBarChart("Alcoholism")
mkBarChart("Handcap")


# In[ ]:


#clean the dataset
for elem in main_df.columns:
    if(main_df[elem].isnull().sum() > 0):
        print(elem,main_df[elem].isnull().sum())
main_df['Gender'].replace(to_replace='M',value='1',inplace=True)
main_df['Gender'].replace(to_replace='F',value='0',inplace=True)
main_df['No-show'].replace(to_replace='Yes',value='1',inplace=True)
main_df['No-show'].replace(to_replace='No',value='0',inplace=True)
main_df.Neighbourhood.value_counts(normalize=True)


# In[ ]:




