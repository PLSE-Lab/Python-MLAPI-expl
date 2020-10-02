#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings

pd.set_option('display.max_columns', 300)
warnings.filterwarnings('ignore')


# In[ ]:


multiple_choice_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
multiple_choice_df.head()


# In[ ]:


def plot_bivariant(df, x1, x2, mode='group', percentage=False, legend_orientation='h'):
    temp = df.groupby([x1, x2]).agg('size').reset_index()
    data = []
    for i in temp[x2].unique():
        temp_df = temp[temp[x2].isin([i])]
        col = 0
        if percentage:
            temp_df.loc[:, 'percentage'] = temp_df[0].apply(lambda x: (x*100)/temp_df[0].sum())
            col = 'percentage'
        data.append(
            go.Bar(name=i, x=temp_df[x1].unique(), y=temp_df[col].values)
        )
    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(barmode=mode, legend_orientation=legend_orientation)
    fig.show()


# ## Which age group work on which tech?
# ### Insights
# - Among 18-21, Basic Statistics like (MS Excel, Google sheets) is quite popular.
# - Among 25-29, Local Development Environments(RStudio, JupyterLab, etc.) are most popular then any other age group.
# - Among 30-39, Bussiness Intelligence and Cloud based software are more populars.

# In[ ]:


plot_bivariant(multiple_choice_df.iloc[1:, :], x1='Q1', x2='Q14', mode='group', percentage=True)


# ## Which are popular tools among different genders?

# In[ ]:


plot_bivariant(multiple_choice_df.iloc[1:, :], x1='Q2', x2='Q14', percentage=True)


# ## India USA comparison on statstical tools usage?
# ### Insights
# - In India percentage of Basic Statstical Software are more popular than any other.
# - In USA percentage of Advanced Statstical Software are more popular than any other.

# In[ ]:


plot_bivariant(
    multiple_choice_df[multiple_choice_df['Q3'].isin(['India', 'United States of America'])].iloc[1:, :],
    x1='Q3', x2='Q14')


# ## Which tools are used by different professionals?
# ### Insights
# - Data Scientist are mostly using Cloud based data softwares (AWS, Azure, GCP, etc.)
# - Bussiness Intelligence Software (Tableau, Salesforce) are most common among Data Analyts.
# - Students are more inclined to Basic stastical Software and local development Environments.

# In[ ]:


plot_bivariant(multiple_choice_df.iloc[1:, :], x1='Q5', x2='Q14', percentage=True, legend_orientation='h')


# ## Size of companies vs Data Tools

# In[ ]:


plot_bivariant(multiple_choice_df.iloc[1:, :], x1='Q8', x2='Q14', legend_orientation='h')


# In[ ]:




