#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import seaborn as sns
from matplotlib_venn import venn3
warnings.filterwarnings("ignore")
import matplotlib.pylab as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# Read Multiple Choice
mc = pd.read_csv('../input/multipleChoiceResponses.csv')

def create_plot(question_number,parts,drop_parts):

    list_question_parts = []
    for part in range(1,parts+1):
        if part not in drop_parts:
            list_question_parts.append('Q' + str(question_number) + '_Part_' + str(part))
    
    ide_qs = mc[list_question_parts].drop(0)

    ide_qs.columns=ide_qs.mode().values[0]
    ide_qs_binary = ide_qs.fillna(0).replace('[^\\d]',1, regex=True)

    color_pal = sns.color_palette("Blues", parts)

    (ide_qs_binary.sum() / ide_qs_binary.count()).sort_values().plot(kind='barh', figsize=(10, 10),
         color=color_pal)

    plt.show()
    
def create_plot_from_single_column(question_number,parts):
    color_pal = sns.color_palette("Blues", parts)
    mc['Q' + str(question_number)].value_counts()[0:parts].sort_values().plot(kind='barh', figsize=(10, 10),color=color_pal)


# # Who is completing the survey

# ### Select the title most similar to your current role
# 

# In[ ]:


create_plot_from_single_column(6,10)


# ### In what industry is your current employer/contract 

# In[ ]:


create_plot_from_single_column(7,10)


# ### In which country do you currently reside?

# In[ ]:


create_plot_from_single_column(3,10)


# # Platforms and tools 

# ### Which of the following cloud computing services have you used at work or school in the last 5 years? (Select all that apply)
# - We give credits to our community to encourage them to use GCP. Believe we've given out credits to our ~1K data scientists, so shouldn't make a huge difference but worth bearing in mind

# In[ ]:


create_plot(15,7,[None])


# ### Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)

# In[ ]:


create_plot(27,20,[19])


# ### Which of the following machine learning products have you used at work or school in the last 5 years? (Select all that apply)
# 

# In[ ]:


create_plot(28,43,[42])


# ### Which of the following big data and analytics products have you used at work or school in the last 5 years? (Select all that apply)
# - Kaggle has a light BigQuery integration, which allows our users to access the BigQuery public datasets. That probably has some impact on the BigQuery numbers 

# In[ ]:


create_plot(30,25,[24])


# ### Which of the following relational database products have you used at work or school in the last 5 years? (Select all that apply)

# In[ ]:


create_plot(29,28,[27])


# ### Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply)

# In[ ]:


create_plot(13,15,[None])


# ### Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply)
# - Kaggle Kernels numbers inflated because it's a survey of the Kaggle community
# - Question if some of JupyterHub/Binder responders are actually just Jupyter Notebook users

# In[ ]:


create_plot(14,11,[10])


# # Languages and Libraries

# ### What programming languages do you use on a regular basis? (Select all that apply)

# In[ ]:


create_plot(16,18,[None])


# ### What specific programming language do you use most often?

# In[ ]:


create_plot_from_single_column(17,17)


# ### What machine learning frameworks have you used in the past 5 years? (Select all that apply)

# In[ ]:


create_plot(19,19,[None])


# ### Of the choices that you selected in the previous question, which ML library have you used the most?

# In[ ]:


create_plot_from_single_column(20,18)


# ### What data visualization libraries or tools have you used in the past 5 years? (Select all that apply)

# In[ ]:


create_plot(21,13,[None])


# # What data scientists work on 

# ### Which types of data do you currently interact with most often at work or school? (Select all that apply)

# In[ ]:


create_plot(31,12,[None])


# ### During a typical data science project, what percent of your time is spent engaged in the following tasks?

# In[ ]:


color_pal = sns.color_palette("Blues", 6)
df_pct_of_time = mc[['Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6']][1:].dropna().astype(float).mean()
df_pct_of_time.index = ['Gathering data','Cleaning data','Visualizing data','Model building/selection','Putting models into production','Finding inights and communicating results']
df_pct_of_time.sort_values().plot(kind='barh', figsize=(10, 10),color=color_pal)


# ### Does your current employer incorporate machine learning methods into their business?

# In[ ]:


create_plot_from_single_column(10,6)


# ### Select any activities that make up an important part of your role at work (Select all that apply)

# In[ ]:


create_plot(11,7,[None])


# # Learning Resources

# ### On which online platforms have you begun or completed data science courses? (Select all that apply)
# - Kaggle Learn is likely overstated 

# In[ ]:


create_plot(36,12,[12])


# ### Who/what are your favorite media sources that report on data science topics? (Select all that apply)

# In[ ]:


create_plot(38,18,[None])


# In[ ]:




