#!/usr/bin/env python
# coding: utf-8

# # What are the factors affecting Graduate Admissions in America for Students?

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
import seaborn as sns
import matplotlib.pyplot as plt
# reading dataset
df=pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

#importing plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()


# # Data Statistics and sneek-peek into the data:

# In[ ]:


#General data statistics
display(df.head())
df.info()
display(df.describe())


# ## Key Highlights from dataset:
# * Average GRE Score: 316.47
# * Average TOEFL Score: 107.19
# * Average CGPA: 8.58
# * With Research: 56% of applicants

# Checking if plotly and cufflinks are working correctly

# In[ ]:


#data = [go.Histogram(x=df["GRE Score"])]
# checking if plotly and cufflinks are working correctly
df['GRE Score'].iplot(kind="hist", bins=40,title="GRE Score Distribution")


# In[ ]:


layout1 = cf.Layout(
    height=600,
    width=800,
    margin=dict(
        l=200
    )
)
df.corr().iplot(kind='heatmap',colorscale='spectral', title = 'Correlation between different maps', 
    layout=layout1)


# ## Following are highest correlated items with Chance of admit:
# * CGPA
# * GRE Score
# * TOEFL Score

# In[ ]:


df['Admit Chance']=pd.cut(np.array(df['Chance of Admit ']),3, labels=["bad", "medium", "good"])
#new_labels[:5]


# # How good is your Acceptance chance to a University based on your Scores:
# **NOTE: You can even turn the markers on and off by clicking on the legends in the below charts**

# In[ ]:


scores_attr=['CGPA', 'GRE Score', 'TOEFL Score']
for i in scores_attr:
    df.iplot(x=i,y='University Rating',categories='Admit Chance',colors=['green','blue','red'],
            xTitle=i,yTitle='University Rating',title=f'Chances of Acceptance based on your {i}')


# # How does your SOP and LOR affect your chances of getting accepted?

# In[ ]:


color_dict={'good':'seagreen','medium':'skyblue','bad':'indianred'}
df_grouped=df.groupby(['SOP','LOR ','Admit Chance']).size().reset_index(name='counts')
#df_grouped.head()
df_grouped.iplot(kind='bubble',x='SOP',y='LOR ',xTitle='SOP',yTitle='LOR',title='Distribution of SOP and LOR with acceptance chances',
                 size='counts',text='Admit Chance',colors=df_grouped['Admit Chance'].map(color_dict).tolist())


# ### This one is quite natural, students with good SOPs and good LORs have better acceptance chances. Although there are some exceptions.

# # Ideally students who are good at academics should have good GRE and TOEFL Score. Lets check this hypothesis below
# Zoom in, rotate, check the values in the 3d plot below

# In[ ]:


studious_students=df[df['CGPA'] > 8]
studious_students.iplot(kind='scatter3d', x='GRE Score', y='TOEFL Score',z='CGPA',mode='markers', xTitle='GRE Score',yTitle='TOEFL Score',zTitle='CGPA',
                        title='GRE vs TOEFL vs CGPA')


# Our hypothesis seems to be true.

# ## Now, just to check the relationship between the SOP and LOR of students with Research. General guess would be students with research should have a good LOR and SOP.

# In[ ]:


df_research_grouped=df.groupby(['SOP','LOR ','Research']).size().reset_index(name='counts')


# In[ ]:


import plotly.tools as tls

fig = tls.make_subplots(rows=1, cols=2, shared_yaxes=True)
                       
df_non_research=df_research_grouped[df_research_grouped['Research']==0]
df_research=df_research_grouped[df_research_grouped['Research']==1]
fig.append_trace({'x': df_non_research.SOP, 'y': df_non_research['LOR '],'text':df_non_research['counts'],'type': 'scatter', 'name': 'Non Research','mode':'markers'}, 1, 1)
fig.append_trace({'x': df_research.SOP, 'y': df_research['LOR '], 'type': 'scatter','text':df_research['counts'], 'name': 'Research','mode':'markers'}, 1, 2)
fig['layout']['xaxis1'].update(title='SOP')
fig['layout']['xaxis2'].update(title='SOP')
fig['layout']['yaxis1'].update(title='LOR')
fig['layout'].update(hovermode= 'closest')
fig['layout'].update(title='SOP vs LOR for applicants with Research & Non Research experience')

cf.iplot(fig)


# ## If you hover around and check the counts at the top right corners for both the plots. It is evident that research does help making your SOP and LOR better.

# # To Summarize:
# * CPGA plays the most important role in admissions followed by GRE score and TOEFL.
# * Good SOPs and LORs are essential to get into the best universities
# * Research makes your SOP and LOR better.
# * Studious students generally tend to do good at GRE and TOEFL.
# 
# > Disclamer: This is a very small dataset and the comments above are in accordance to the data given. To perform more analysis, it is essential that we have more data points and features.

# In[ ]:




