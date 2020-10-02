#!/usr/bin/env python
# coding: utf-8

# ![machineLearning enginners analysis](https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwi8jP-voY_mAhWyY98KHU4HCDgQjRx6BAgBEAQ&url=https%3A%2F%2Fnorthconcepts.com%2Fblog%2F2017%2F11%2F30%2Fmachine-learning-artificial-intelligence-conferences%2F&psig=AOvVaw2fB7ZLsiT6H6hINCP5mIa0&ust=1575110319422465)
# 
# 
# kaggle published 2019 Kaggle ML & DS Survey. So many usefull info to mine. <br> <br>
# I decided to check about my region and see how are other kagglers working. In this notebook I compared Iranian kagglers and Turkish kagglers.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


fellowship_of_the_kaggle = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv")


# ## 1. let's see how is the distribution of users in all over the world

# In[ ]:


fellowship_of_the_kaggle["Q3"] = fellowship_of_the_kaggle.Q3.apply(lambda x: "Iran" if x == "Iran, Islamic Republic of..." else x)
IT = fellowship_of_the_kaggle[(fellowship_of_the_kaggle["Q3"] == "Iran") | (fellowship_of_the_kaggle["Q3"] == "Turkey")]

country=fellowship_of_the_kaggle.Q3.value_counts()

fig = go.Figure(go.Treemap(
    labels=country.index,
    parents = ["World"]*len(country),
    values =  country
))

fig.show()


# as we see there are many ML giants like India, Usa, ... . middle east countries are there too : <br>
# * Egypt : 122 
# * Pakistan : 210
# * Iran : 91
# * Turkey : 288
# * Saudi Arabia : 50
# * Israel : 104

# ## 2. compare age of kagglers

# In[ ]:


plt.figure(figsize=(15, 8))

ax = sns.countplot(x="Q1", hue="Q3", data=IT)
plt.xlabel("Age distribution")
plt.xticks(rotation=0, size=13)

for p in ax.patches:
    ax.annotate(format(p.get_height(), ".0f"), (p.get_x() + p.get_width() / 2. ,p.get_height()), 
              ha="center", va="center", xytext=(0, 10), textcoords = "offset points")


# good data. as we see 20-30 has most kagglers. <br>
# 40-49 in turkey kagglers was amazing for me there are 33 engineers.

# ## 3. gender ratio

# In[ ]:


fig = make_subplots(rows=1, cols=2, specs=[[{"type" : "pie"}, {"type" : "pie"}]], subplot_titles=("IRAN", "TURKEY"))

fig.add_trace(
    go.Pie(labels=IT[IT.Q3 == "Iran"].Q2.value_counts()[:2].index, values=IT[IT.Q3 == "Iran"].Q2.value_counts()[:2].values),
    row=1, col=1
)

fig.add_trace(
    go.Pie(labels=IT[IT.Q3 == "Turkey"].Q2.value_counts()[:2].index, values=IT[IT.Q3 == "Turkey"].Q2.value_counts()[:2].values),
    row=1, col=2
)

fig.update_layout(height=400, showlegend=True)
fig.show()


# as expected men are more. 

# ## 4. It's about MONEY

# In[ ]:


fig = make_subplots(rows=2, cols=2, specs=[[{"type" : "bar"}, {"type" : "bar"}], [{"type" : "bar"}, {"type" : "bar"}]]
                    , subplot_titles=("compensation in Iran", " compensation in TURKEY", " Money Spent in Iran in this field", "money Spent in Turkey in this field"))

fig.add_trace(
    go.Bar(x=IT[IT.Q3 == "Iran"].Q10.value_counts().index, y=IT[IT.Q3 == "Iran"].Q10.value_counts().values, name="Iran"),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=IT[IT.Q3 == "Turkey"].Q10.value_counts().index, y=IT[IT.Q3 == "Turkey"].Q10.value_counts().values, name="Turkey"),
    row=1, col=2
)

fig.add_trace(
    go.Bar(x=IT[IT.Q3 == "Iran"].Q11.value_counts().index, y=IT[IT.Q3 == "Iran"].Q11.value_counts().values, name="Iran"),
    row=2, col=1
)

fig.add_trace(
    go.Bar(x=IT[IT.Q3 == "Turkey"].Q11.value_counts().index, y=IT[IT.Q3 == "Turkey"].Q11.value_counts().values, name="Turkey"),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False )
fig.show()


# huge difference in  **compensation** .<br> <br> but you must get the point. Iran's Rial (iran's Monetary unit) is really really really really cheap ( bad for us ). so now you can guess why there is this much difference.

# ## 5. let's check coding background of kagglers

# In[ ]:


fig = make_subplots(rows=2, cols=2, specs=[[{"type" : "pie"}, {"type" : "pie"}], [{"type" : "pie"}, {"type" : "pie"}]]
                    , subplot_titles=("background coding in analyse (IRAN)", " background coding in analyse(Turkey)", "background coding in Machine Learning(Iran)", "background coding in Machine Learning(Turkey)"))

fig.add_trace(
    go.Pie(labels=IT[IT.Q3 == "Iran"].Q15.value_counts().index, values=IT[IT.Q3 == "Iran"].Q15.value_counts().values, name="Iran"),
    row=1, col=1
)

fig.add_trace(
    go.Pie(labels=IT[IT.Q3 == "Turkey"].Q15.value_counts().index, values=IT[IT.Q3 == "Turkey"].Q15.value_counts().values, name="Turkey"),
    row=1, col=2
)

fig.add_trace(
    go.Pie(labels=IT[IT.Q3 == "Iran"].Q23.value_counts().index, values=IT[IT.Q3 == "Iran"].Q23.value_counts().values, name="Iran"),
    row=2, col=1
)

fig.add_trace(
    go.Pie(labels=IT[IT.Q3 == "Turkey"].Q23.value_counts().index, values=IT[IT.Q3 == "Turkey"].Q23.value_counts().values, name="Turkey"),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=True )
fig.show()


# as we can see most of engineers have less than 5years of experience. (just like me)<br>
# but be aware, this field is new somehow.

# ## 6. environmenets we use

# In[ ]:


iran_df = IT[IT.Q3 == "Iran"].Q14.value_counts()
turkey_df = IT[IT.Q3 == "Turkey"].Q14.value_counts()

fig = go.Figure(data=[
    go.Bar(x=iran_df.index[:2],y=iran_df.values[:2],name="IRAN"),
    go.Bar(x=turkey_df.index[:2],y=turkey_df.values[:2],name="Turkey")
])
fig.update_layout(barmode='group', height=400)
fig.show()


# Both prefer local enviremnet
# as I checked [Sahib singh](https://www.kaggle.com/sahib12) notebook in India and USA we had same result

# ## 7. At Last, Programming Language

# In[ ]:


lang = {}
lang['Python']=IT[IT.Q3 == "Iran"].Q18_Part_1.value_counts().sum()
lang['R']=IT[IT.Q3 == "Iran"].Q18_Part_2.value_counts().sum()
lang['SQL']=IT[IT.Q3 == "Iran"].Q18_Part_3.value_counts().sum()
lang['C']=IT[IT.Q3 == "Iran"].Q18_Part_4.value_counts().sum()
lang['C++']=IT[IT.Q3 == "Iran"].Q18_Part_5.value_counts().sum()
lang['Java']=IT[IT.Q3 == "Iran"].Q18_Part_6.value_counts().sum()
lang['Javascript']=IT[IT.Q3 == "Iran"].Q18_Part_7.value_counts().sum()
lang['TypeScript']=IT[IT.Q3 == "Iran"].Q18_Part_8.value_counts().sum()
lang['Bash']=IT[IT.Q3 == "Iran"].Q18_Part_9.value_counts().sum()
lang['MATLAB']=IT[IT.Q3 == "Iran"].Q18_Part_10.value_counts().sum()
lang['None']=IT[IT.Q3 == "Iran"].Q18_Part_11.value_counts().sum()
lang['Other']=IT[IT.Q3 == "Iran"].Q18_Part_12.value_counts().sum()

iran_lang_df = pd.DataFrame(lang.items(), columns=["Language", "Count"])

lang2 = {}
lang2['Python']=IT[IT.Q3 == "Turkey"].Q18_Part_1.value_counts().sum()
lang2['R']=IT[IT.Q3 == "Turkey"].Q18_Part_2.value_counts().sum()
lang2['SQL']=IT[IT.Q3 == "Turkey"].Q18_Part_3.value_counts().sum()
lang2['C']=IT[IT.Q3 == "Turkey"].Q18_Part_4.value_counts().sum()
lang2['C++']=IT[IT.Q3 == "Turkey"].Q18_Part_5.value_counts().sum()
lang2['Java']=IT[IT.Q3 == "Turkey"].Q18_Part_6.value_counts().sum()
lang2['Javascript']=IT[IT.Q3 == "Turkey"].Q18_Part_7.value_counts().sum()
lang2['TypeScript']=IT[IT.Q3 == "Turkey"].Q18_Part_8.value_counts().sum()
lang2['Bash']=IT[IT.Q3 == "Turkey"].Q18_Part_9.value_counts().sum()
lang2['MATLAB']=IT[IT.Q3 == "Turkey"].Q18_Part_10.value_counts().sum()
lang2['None']=IT[IT.Q3 == "Turkey"].Q18_Part_11.value_counts().sum()
lang2['Other']=IT[IT.Q3 == "Turkey"].Q18_Part_12.value_counts().sum()

turkey_lang_df = pd.DataFrame(lang2.items(), columns=["Language", "Count"])


fig = make_subplots(rows=1, cols=2, specs=[[{"type" : "pie"},{"type" : "pie"}]], subplot_titles=('IRAN',"Turkey"))

fig.add_trace(go.Pie(labels=iran_lang_df.Language, values=iran_lang_df.Count, name="Iran"), row=1, col=1)
fig.add_trace(go.Pie(labels=turkey_lang_df.Language, values=turkey_lang_df.Count, name="Turkey"), row=1, col=2)

fig.update_layout(height=800, showlegend=True)

fig.update_traces(hole=.4)# to create donut like pie chart

fig.show()


# I could guess Python be first<br><br><br>
# 
# 
# 
# AT ALL, When I checked other users notebooks there was same results in many cases, just in some fileds because of political and weird situation of some countries ( like my country :D ) something were different(for example everything about money).
# <br>
# 
# 
# thanks for reading.
