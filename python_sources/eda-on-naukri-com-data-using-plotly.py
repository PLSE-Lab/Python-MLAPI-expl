#!/usr/bin/env python
# coding: utf-8

# <img src="https://upload.wikimedia.org/wikipedia/commons/9/9b/Naukri_Logo.png" width="400px">

# ### Exploring Naukri.com data

# In[ ]:


import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        PATH = os.path.join(dirname, filename)


# # Table of contents
# 
# * [Read Data](#section-one)
# * [Structure and Summary](#section-two)
# * [Exploratory Data Analysis](#section-three)
#     - [Job Experience](#sub-section-one)
#     - [Role Category](#sub-section-two)
#     - [Location](#sub-section-three)
#     - [Job Title](#sub-section-four)
#     - [Word Clouds](#sub-section-five)
#         * [Functional Area](#sub-sub-section-one)
#         * [Key Skills](#sub-sub-section-two)

# <a id="section-one"></a>
# ## Read Data

# In[ ]:


df = pd.read_csv(PATH)


# <a id="section-two"></a>
# # Structure and Summary

# In[ ]:


df.head(10)


# In[ ]:


df.tail(10)


# In[ ]:


print("There are {} rows and {} columns in the dataset.".format(df.shape[0], df.shape[1]))


# In[ ]:


df.isnull().sum()


# Seems like there are a few missing values in our data. We can choose to keep them unless we plan on building a statisitical model.

# <a id="section-three"></a>
# # Exploratory Analysis

# Let's try to get a picture of what kind of roles are typically posted on Naukri.com. We will first look at job experience ranges.

# <a id="sub-section-one"></a>
# ### Job Experience

# In[ ]:


job_exp = df["Job Experience Required"].value_counts().nlargest(n=10)
job_exp_all = df["Job Experience Required"].value_counts()
fig = make_subplots(1,2, 
                    subplot_titles = ["Top 10 experience ranges", 
                                      "All experience ranges"])
fig.append_trace(go.Bar(y=job_exp.index,
                          x=job_exp, 
                          orientation='h',
                          marker=dict(color=job_exp.values, coloraxis="coloraxis", showscale=False),
                          texttemplate = "%{value:,s}",
                          textposition = "inside",
                          name="Top 10 experience ranges",
                          showlegend=False),
                
                 row=1,
                 col=1)
fig.update_traces(opacity=0.7)
fig.update_layout(coloraxis=dict(colorscale='tealrose'))
fig.append_trace(go.Scatter(x=job_exp_all.index,
                          y=job_exp_all, 
                          line=dict(color="#008B8B",
                                    width=2),
                          showlegend=False),
                 row=1,
                 col=2)
fig.update_layout(showlegend=False)
fig.show()


# We can see that the most popular range of experience posted on Naukri is between 2 - 5 years. This would typically fall under the range of entry level to junior level candidates. 
# 
# This makes sense because typically more seasoned candidates tend to look for alternative ways of searching for jobs such as through referrals. A lot of companies offer bonus cash to employees whose referrals get hired. So it works both ways.

# <a id="sub-section-two"></a>
# ### Role Category

# In[ ]:


role = df['Role Category'].value_counts().nlargest(n=10)
fig = px.pie(role, 
       values = role.values, 
       names = role.index, 
       title="Top 10 Role Categories", 
       color=role.values,
       color_discrete_sequence=px.colors.qualitative.Prism)
fig.update_traces(opacity=0.7,
                  marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_x=0.5)
fig.show()


# It's hardly surprising that one of the top job portals in India has a huge number of programming jobs. Software/IT services jobs are extremely popular in India. Most of the biggest MNCs of the world have their IT offices in India.

# To reiterate our previous observation, the words/phrases with the most weight are it, software, programming, maintenance, software, applciation etc.

# <a id="sub-section-three"></a>
# ### Location

# In[ ]:


location = df['Location'].value_counts().nlargest(n=10)
fig = px.bar(y=location.values,
       x=location.index,
       orientation='v',
       color=location.index,
       text=location.values,
       color_discrete_sequence= px.colors.qualitative.Bold)

fig.update_traces(texttemplate='%{text:.2s}', 
                  textposition='outside', 
                  marker_line_color='rgb(8,48,107)', 
                  marker_line_width=1.5, 
                  opacity=0.7)

fig.update_layout(width=800, 
                  showlegend=False, 
                  xaxis_title="City",
                  yaxis_title="Count",
                  title="Top 10 cities by job count")
fig.show()


# Bangalore and Mumbai are called the IT capital and the financial capital of India respectively. Thus, it's expected that these two cities will have the maximum number of jobs among cities in India.

# <a id="sub-section-four"></a>
# ### Job Title

# In[ ]:


title = df['Job Title'].value_counts().nlargest(n=10)
fig = make_subplots(2, 1,
                    subplot_titles=["Top 10 Job Titles", 
                                    "Top Job Titles in Mumbai and Bengaluru"])
fig.append_trace(go.Bar(y=title.index,
                        x=title.values,
                        orientation='h',
                        marker=dict(color=title.values, coloraxis="coloraxis"),
                        texttemplate = "%{value:,s}",
                        textposition = "inside",
                        showlegend=False),
                  row=1,
                  col=1)
fig.update_layout(coloraxis_showscale=False)
fig.update_layout(height=800, 
                  width=800,
                  yaxis=dict(autorange="reversed"),
                  coloraxis=dict(colorscale='geyser'),
                  coloraxis_colorbar=dict(yanchor="top", y=1, x=0)
)

fig.append_trace(go.Bar(y=df[df['Location'].isin(['Mumbai'])]['Job Title'].value_counts().nlargest(n=10).index,
                        x=df[df['Location'].isin(['Mumbai'])]['Job Title'].value_counts().nlargest(n=10).values,
                        marker_color='#008080',
                        orientation='h',
                        showlegend=True,
                        name="Mumbai"),
                row=2,
                col=1)

fig.append_trace(go.Bar(y=df[df['Location'].isin(['Bengaluru'])]['Job Title'].value_counts().nlargest(n=10).index,
                        x=df[df['Location'].isin(['Bengaluru'])]['Job Title'].value_counts().nlargest(n=10).values,
                        marker_color='#00CED1',
                        orientation='h',
                        showlegend=True,
                        name="Bengaluru"),
                row=2,
                col=1,
                )
fig.update_layout(legend=dict(x=1,
                              y=0.3))
fig.show()


# We compare top job titles across all locations as well drill down to our top two job posting cities, Bengaluru and Mumbai. We see that overall Business Developement Executives seem to have the highest demand across all locations in India. They are followed by Sales Executives and PHP Developers. 
# 
# When we drill down to the city level, Mumbai seems to have a higher demand for Business Developement Managers/Executives. Conversely, Software Engineering roles seem to be in a high demand in Bengaluru. This information is congruent with our previous analysis of the two cities.

# <a id="sub-section-five"></a>
# ### Word Clouds

# <a id="sub-sub-section-one"></a>
# #### Functional Area

# In[ ]:


functional_words = df['Functional Area'].dropna().to_list()
tokenizer = RegexpTokenizer(r'\w+')
tokenized_list = [tokenizer.tokenize(i) for i in functional_words]
tokenized_list = [w for l in tokenized_list for w in l]

tokenized_list = [w.lower() for w in tokenized_list]
string = " ".join(w for w in tokenized_list)
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                min_font_size = 10).generate(string) 
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# The top functional areas are:
# * IT Software
# * Software Application
# * Programming Maintenance
# * Application Programming
# * Business Developement

# <a id="sub-sub-section-two"></a>
# #### Key Skills

# In[ ]:


skills = df['Key Skills'].to_list()
skills = [str(s) for s in skills]
skills = [s.strip().lower()  for i in skills for s in i.split("|")]
string = " ".join(w for w in skills)
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                min_font_size = 10).generate(string) 
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# The top skills are: 
# * Business Developement
# * Software Developer
# * Sales Executive
# * Customer Service
# * Technical Support

# It should come as no surprise that Business Developement and Software Developer are the skills that are most in demand because our top job titles exactly required these skills.

# I hope all of you enjoyed this EDA. Have a good one. :)
