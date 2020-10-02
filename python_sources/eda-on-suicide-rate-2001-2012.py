#!/usr/bin/env python
# coding: utf-8

# # Description
# **According to the recent report of WHO, India is the 6th most depressed country in the world. What makes us so,have you ever pondered? A picture speaks more than 1000 words,a well knwon saying,with this thought here is my notebook that shows an explanatory analysis of the Suicide rate in India from 2001-2012.**
# 
# ![http://www.ngopulse.org/sites/default/files/styles/article-top-image-w320/public/image/images/depression%20image.jpg?itok=ZDs6Nq24](http://www.ngopulse.org/sites/default/files/styles/article-top-image-w320/public/image/images/depression%20image.jpg?itok=ZDs6Nq24)

# ## About the dataset
# **This dataset contains yearly suicide details of all states/u.t of India by various parameters from 2001-2012.The parameters are as follows:-**
# * Gender
# * Age Group
# * Total
# * Type Code - It mainly shows the causes which is categorised as Social,Educational,Professional,Social Status
# * Type - It further categorise the Type code that is the causes

# **Importing Libraries**

# In[ ]:


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


#Plotly libraries
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# **Loading data**

# In[ ]:


data=pd.read_csv('../input/suicides-in-india/Suicides in India 2001-2012.csv')
data.info()


# In[ ]:


data.sample(10)


# ## Data Cleaning
# **Before performing EDA on the dataset,data cleaning is important step which involves renmaing some of the entries and dropping some of the rows.**

# In[ ]:


# rename states

data.replace('A & N Islands (Ut)', 'A & N Islands', inplace=True)
data.replace('Chandigarh (Ut)', 'Chandigarh', inplace=True)
data.replace('D & N Haveli (Ut)', 'D & N Haveli', inplace=True)
data.replace('Daman & Diu (Ut)', 'Daman & Diu', inplace=True)
data.replace('Lakshadweep (Ut)', 'Lakshadweep', inplace=True)
data.replace('Delhi (Ut)', 'Delhi', inplace=True)


# In[ ]:


# rename Type

data.replace('Bankruptcy or Sudden change in Economic', 
           'Bankruptcy or Sudden change in Economic Status', inplace=True)
data.replace('By Other means (please specify)', 'By Other means', inplace=True)
data.replace('Not having Children(Barrenness/Impotency',
           'Not having Children (Barrenness/Impotency', inplace=True)


# In[ ]:


data = data.drop(data[(data.State == 'Total (Uts)') | (data.State == 'Total (All India)') | 
               (data.State == 'Total (States)')].index)


# In[ ]:


data=data.drop(data[(data.Type =='By Other means')|(data.Type=='Other Causes (Please Specity)')|
                    (data.Type=='Others (Please Specify)')|(data.Type=='Causes Not known')].index)


# In[ ]:


data = data.drop(data[data.Total==0].index)


# ## EDA 

# **Using plotly an animation slider is plotted for States to show the changing number in the total counts of suicide. Any instance from 2001-2012 can be paused and analysed.**

# In[ ]:


fig = px.bar(data, x="State", y="Total", color="State",
  animation_frame="Year", animation_group="Total", range_y=[0,20000],width=1000)
py.offline.iplot(fig)


# **The below cell shows the most number of suicide counts in states,arranged in highest to lowest.We can observe that states such as Andhra Pradesh,Karnataka,Tamil Nadu counts 4000+ signifying an alarming rate.**

# In[ ]:


temp_state = data.groupby('State').count()['Total'].reset_index().sort_values(by='Total',ascending=False)
temp_state.style.background_gradient(cmap='Reds')


# **Counts of suicide basis on gender. The above pie chart shows that male are more prone to have suicidal instincts than female.**

# In[ ]:


counts = data['Gender'].value_counts().sort_index()
print(counts)
# Plot a pie chart
counts.plot(kind='pie', title='Gender Count',figsize=(10,8))

plt.legend()
plt.show()


# **There are many Causes and sub causes as stated in the dataset.Observing which causes are more responsible for the suicide rates.**

# In[ ]:


# splitting data as per the type code

cause = data[data['Type_code'] == 'Causes']
edu_status = data[data['Type_code'] == 'Education_Status']
means_adpt = data[data['Type_code'] == 'Means_adopted']
prof = data[data['Type_code'] == 'Professional_Profile']
soc_status = data[data['Type_code'] == 'Social_Status']


# In[ ]:


def plot_type(data, Title, X_lab):
    p_type = data.groupby('Type').sum()['Total']
    sort_df = p_type.sort_values(ascending = False)

    fig = sort_df.plot(kind='bar', figsize = (10,6), title = Title + '\n', width = 0.75)
    fig.set_xlabel('\n' + X_lab )
    fig.set_ylabel('Count\n')
    sns.set_style('whitegrid')
    sns.set_palette('Set2')  


# In[ ]:


# plot by Cause
plot_type(cause, 'Suicide by cause', 'Cause')


# In[ ]:


#plot by the educational causes
plot_type(edu_status, 'Suicide by Education Status', 'Education Status')


# In[ ]:


# plot by means adopted
plot_type(means_adpt, 'Suicide by Means Adopted', 'Means Adopted')


# In[ ]:


# suicide by professional profile
plot_type(prof, 'Suicide by Professional Profile', 'Professional Profile')


# In[ ]:


# suicide by social Status
plot_type(soc_status, 'Suicide by Social Status', 'Social Status')


# ## Top 3 states having more suicides rates 
# **Andhra Pradesh,Tamil Nadu,Karnataka ranks the top 3 states having the larming rate for suicide counts.What are the causes that are most reponsible, which age group is more affected ?**

# In[ ]:


#Splitting the data as per the State
State1 = data[data['State']=='Karnataka']
State2 = data[data['State']=='Tamil Nadu']
State3 = data[data['State']=='Andhra Pradesh']


# In[ ]:


def plot_for_State_by_age(data):
    plt.figure(figsize=(12,6))
    data = data[['Age_group','Gender','Total']]
    edSort = data.groupby(['Age_group','Gender'],as_index=False).sum().sort_values('Total',ascending=False)
    sns.barplot(x='Age_group',y='Total',hue='Gender',data=edSort,palette='RdBu')


# In[ ]:


def plot_for_State_by_type(data):
    plt.figure(figsize=(12,6))
    data = data[['Type_code','Gender','Total']]
    edSort = data.groupby(['Type_code','Gender'],as_index=False).sum().sort_values('Total',ascending=False)
    sns.barplot(x='Type_code',y='Total',hue='Gender',data=edSort,palette='ch:2.5,-.2,dark=.3')


# ## Karnataka 

# In[ ]:


#plotting as per the age_group
plot_for_State_by_age(State1)


# In[ ]:


#plotting as per the differnet causes
plot_for_State_by_type(State1)


# ## Tamil Nadu 

# In[ ]:


#plotting as per the age_group
plot_for_State_by_age(State2)


# In[ ]:


#plotting as per the differnet causes
plot_for_State_by_type(State2)


# ## Andhra Pradesh 

# In[ ]:


#plotting as per the age_group
plot_for_State_by_age(State3)


# In[ ]:


#plotting as per the differnet causes
plot_for_State_by_type(State3)


# ## Conclusion
# 
# **Many analysis can be drawn from the data even we can come up with a thesis about the causes and effects,but what we lack is to provide that comfort enviromnent to open up regarding those feelings.Its time to chnage ,it's time to ask are you fine?**
