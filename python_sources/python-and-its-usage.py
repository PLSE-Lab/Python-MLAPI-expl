#!/usr/bin/env python
# coding: utf-8

# **Lets Python!!!**
# 
# Over the years, python's popularity has grown. Now that we have a dataset that shows just that, lets dive down and see how much! (by the numbers)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
path1='../input/python developer survey 2018 external sharing/Python Developer Survey 2018 external sharing/python_psf_external_18.csv'
path2='../input/python developers survey 2017_ raw data sharing/Python Developers Survey 2017_ Raw Data Sharing/pythondevsurvey2017_raw_data.csv'

df_2017=pd.read_csv(path1)
df_2018=pd.read_csv(path2)
def plot_onepie(cols,col_num):
    labels=[]
    values=[]
    title=cols[col_num]
    for a in df_2017.iloc[:,col_num].unique():
        if pd.isna(a):
            percentage=df_2017.iloc[:,col_num].isna().sum() * 100 / df_2017.iloc[:,col_num].count()
        else:
            percentage=(df_2017.iloc[:,col_num]==a).sum() * 100 / df_2017.iloc[:,col_num].count()
        
        labels.append(a)
        values.append(percentage)
        #print(cols[1],a,percentage)
    
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title(title)  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

# Any results you write to the current directory are saved as output.


# Lets see what we are dealing with!

# In[ ]:


df_2017.head(1)


# K, so we have huge column names that are actually questions that people answered in the survey. Since this is a survey, its analytics is to be done a bit differently than the other datasets. Lets see what are the values for the columns that are the answers.

# In[ ]:


#print(len(df_2017.columns))#284
cols=[]
cols=list(df_2017.columns)
print(df_2017.iloc[:,1].unique())


# As suspected, Its a survey so its definitely is a Yes or no. Lets carry on. In the beginning of the code, I created a function that would just print out the pie chart for a column no. that I would pass it. Lets see it in action.

# In[ ]:


plot_onepie(cols,1)


# In[ ]:


plot_onepie(cols,26)


# To reference specific questions or columns, we can create a mapping for the columns contained in the dataframe. Or we could also use integer indexing but here we use this approach and build a question bank with index.

# In[ ]:


ques_bank_index=pd.DataFrame({'Question':cols , 'index':[a for a in range(len(cols))]} )
ques_bank_index.head(5)


# In[ ]:


ques_bank_index


# After we are done with the question bank, lets plot anothe chart, the one that asks the age!

# In[ ]:


plot_onepie(cols,282)


# Moving on from pies (so to speak!), we make use of the column names and the data contained in the columns to figure out how many people replied yes to using a certain language. If you look at the quetion bank, We have question starting from the index 3 that asks, "Java:What other language(s) do you use?" . The interesting thing about this is that the pattern remains the same until column 24. We have a pattern and the pattern is that "whatever is left of the colon is the programming language name , hence the question is being asked about that programming language! i.e. besides Python, Do you use X language? We use this pattern and extract our data to see which language stands out besides Python."

# In[ ]:


count_dict={}
lang=''
lang_list=[]
val_list=[]
#print(df_2017.iloc[:,a].isna().count())
for a in range(3,24):
    lang=cols[a].split(':')[0]
    #print(lang)
    val=(df_2017.iloc[:,a]==lang).sum()
    lang_list.append(lang)
    val_list.append(val)
#print(count_dict.keys(),[count_dict[a] for a in count_dict.keys()])
count_dict['language']=lang_list
count_dict['value']=val_list
df_lang_count=pd.DataFrame(count_dict)
df_lang_count=df_lang_count.sort_values(by='value',ascending=False)
df_lang_count


# More to come soon! stay tuned!
