#!/usr/bin/env python
# coding: utf-8

#  # ** Stack_OverFlow EDA Analysis **
#  # <a id="1">1. Load Libraries</a>

# In[2]:


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import plotly.offline as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import random
from plotly import tools
from plotly.tools import FigureFactory as ff

from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color


# In[3]:


so_data=pd.read_csv('../input/survey_results_public.csv')


# # 2. size of the Dataset
#     

# In[3]:


print("size of the data",so_data.shape)


#  there are 98855 developers participating in this survey 

# # <a id=3> 3.Gilmse of Data</a>

# In[6]:


so_data.head()


# # <a id=4> 4.checking for missing values </a>

# In[7]:


total=so_data.isnull().sum().sort_values(ascending=False)


# In[8]:


percent=(so_data.isnull().sum()/so_data.isnull().count()*100).sort_values(ascending=False)
missing_data=pd.concat([total,percent] ,axis=1 ,keys=['Total','Percent'])
#missing_data


# # 5. Exploring the Survey Data
# ## <a id='5-1'> 5.1Coding as Hobby or Not?</a>

# In[6]:





hb=so_data['Hobby'].value_counts()
data=[go.Pie(labels=hb.index,
             values=hb.values,
             marker=dict(colors=random_colors(2)),
             
             )]
layout=go.Layout(title="% of developers coding as hobby")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


#  80.8 % of developers are coding as hobby and 19.2 % coding for other purposes
#  ## <a id='5-2'>5.2. How maney developers are contributing to open source</a>

# In[7]:


open=so_data['OpenSource'].value_counts()
data=[go.Pie(labels=open.index,
             values=open.values,
             marker=dict(colors=random_colors(2)),
            
             )]
layout=go.Layout(title="% of developers contributing to open source")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# 43.6% developers are contributing to open source

# ## <a id='5-3'>5.3 From which country Highest no of developers belong to ?

# In[21]:


temp=so_data['Country'].dropna().value_counts().sort_values(ascending=False).head(10)


# In[29]:



x=["united states","india","Germany","Uk","Canada","russia","France","Brazil","Poland","Australia"]
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
y=so_data['Country'].dropna().value_counts().sort_values(ascending=False).head(10)


width=1/1.5
plt.bar(x,y,width,color='blue')
xticks=x
plt.xticks(rotation=90)
plt.grid()
plt.xlabel='country'
plt.ylabel='count'
plt.title='country with high no of developers'
plt.show()

#plt.bar(xlabel,ylabel ,title="country with highest no of developers",color=["#080808"])


# **united states have highest no of developer which is 20309 **

# In[29]:


temp=so_data['Country'].dropna().value_counts().sort_values(ascending=False).head(10)
x=["united states","india","Germany","Uk","Canada","russia","France","Brazil","Poland","Australia"]

data=[go.Pie(labels=x,
             values=temp.values,
             marker=dict(colors=random_colors(6)),
            
             )]
layout=go.Layout(title="% country with greater # of developers")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# #### united states have 32.7% of top 10 developer majority countries 

# ## <a id='5-4'>5.4. How maney developers are enrolled in formal,degree-granting college or university</a>

# In[47]:


temp=so_data["Student"].value_counts()
len(so_data["Country"])



# In[36]:


print(temp)


# In[37]:


y=so_data.Student.value_counts()
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
x=["Not_student","Full_Time_Student","Part_Time_student"]

xticks=x
xlabel=x
ylabel=y
plt.grid()
title="Number of developers who are currently enrolled in formal education"
width=1/1.5
plt.bar(x,y,width,color=random_colors(3))
plt.show()


# In[4]:


std=so_data["Student"].value_counts()
data=[go.Pie(labels=std.index,
            values=std.values,
            marker=dict(colors=random_colors(3)),
            )]
layout=go.Layout(title="No of developers currently enrolled")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# ### 70399 developers out of 98855 (74.2%) are not currently enrolled in formal education

# ## <a id="5-5">5.5 Employment status of developers

# In[22]:


temp=so_data.Employment.isnull().value_counts()
print("Mensioned:",temp[False],"Missing_values:",temp[True])
percent_employed=(temp[False])/(temp[True]+temp[False])*100
print(" %age of developers mensioned :",percent_employed)
print("%age of developers not mensioned",temp[True]/98855*100)


# In[15]:


employed=so_data.Employment.value_counts()
x=["full-time","independent","Not employed","part-time","not-interested","retired"]
y=employed
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
xticks=x
plt.xticks(rotation=90)
yticks=y
xlabel=x
ylabel=y
title="%age of employement categories"
width=1/1.5
plt.bar(x,y,width,color=random_colors(6))


# ## <a id="5-6"> 5.6 Highest Formal Education of developers</a>

# In[4]:


y=so_data.FormalEducation.value_counts()

x=["Bechlor","Master","degreeless-study","secondary_school","Associate","doctoral","primar/elemantry","Professional","No_formal_education"]
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
width=1/1.5
xticks=x
plt.xticks(rotation=90)
yticks=y

xlabel=x
ylabel="developers with formal education"
plt.bar(x,y,width,color=random_colors(9))


# In[5]:


x=["Bechlor","Master","degreeless-study","secondary_school","Associate","doctoral","primar/elemantry","Professional","No_formal_education"]

fe=so_data["FormalEducation"].value_counts()
data=[go.Pie(labels=x,
            values=fe,
            marker=dict(colors=random_colors(9)),
            )]
layout=go.Layout(title="%of developers with formal education")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# ## formal education status:
# * 46.1% developers have bechlor degree.
# * 22.6 % developers have master degree
# * 12.4 % developers have no formal degree
# * 9.45 % have secondary school degree
# * 3.14 % have Associate degree
# * 2.34 % have doctoral degree
# * 1.75 % have primary /elemantary formal education.
# * 1.53 % have some profession education
# * 0.739 % no formal education

# ## <a id="5-7">5.7 undergraduate Major of developers</a>

# In[16]:


so_data.head(4)
major=so_data.UndergradMajor.value_counts()
print(major)


# In[7]:


major=so_data.UndergradMajor.value_counts()

data=[go.Pie(labels=major.index,
            values=major.values,
            marker=dict(colors=random_colors(12)),
            )]
layout=go.Layout(title="% of developers with undergraduate major")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# ### 63.7 % developers have computer science,computer engineering,or software engineering as undergraduation major

# ## <a id="5-8"> 5.8 company size of the developer </a>

# In[21]:


so_data.CompanySize.value_counts()


# In[26]:


y=so_data.CompanySize.value_counts()
x=y.index
fig=plt.Figure()
ax=fig.add_subplot(1,1,1)
xlabel=x
ylabel=y
xticks=x
plt.xticks(rotation=90)
yticks=y
plt.grid()
width=1/1.5
plt.bar(x,y,width,color=random_colors(8))


# ## most of the developers work in 20 -99 size company (23.8% see below)

# In[11]:


y=so_data.CompanySize.value_counts()

data=[go.Pie(labels=y.index,
             values=y,
             marker=dict(colors=random_colors(8)),
            )]
layout=go.Layout(title="%age of developer by compay size")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# ## <a id="5-9"> 5.9 Types of developers</a>

# In[12]:


temp=pd.DataFrame(so_data.DevType.dropna().str.split(";").tolist()).stack()
grouped=temp.value_counts()

x=grouped.index
y=(grouped/grouped.sum())*100

fig=plt.Figure()
ax=fig.add_subplot(1,1,1)
xlabel=x
xticks=x
plt.grid()
plt.xticks(rotation=90)
ylabel=y
width=1/1.5
plt.bar(x,y,width,color=random_colors(20))
plt.show()


# ##  about 19% of developers are Back-end developers
# ##  about  16 % of developers are Full- stack developers

# ## <a id="5-10"> 5.10 are the developers satisfied with current job?</a>

# In[3]:


y=so_data.JobSatisfaction.value_counts()
data=[go.Pie(labels=y.index,
            values=y.values,
            marker=dict(colors=random_colors(7)),
            )]
layout=go.Layout(title="% of developers satisfied with current job")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# # Interesting! 
# ## only 18% of developers are extremly satified with current job

# ## <a id="5-11"> 5.11 Career satisfaction of developers</a>

# In[4]:


y=so_data.CareerSatisfaction.value_counts()
data=[go.Pie(labels=y.index,
            values=y.values,
            marker=dict(colors=random_colors(7)),
            )]
layout=go.Layout(title="% of developers satisfied with their career")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# ## 18.7 % developers are extremly satisfied with their career

# ## <a id="5-12"> 5.12 languages worked with by the developers?</a>

# In[59]:


temp=pd.DataFrame(so_data.LanguageWorkedWith.dropna().str.split(";").tolist()).stack()
lang=temp.unique()
print("Total language are:",len(lang))
print("which are :",lang)
print(temp.value_counts())



y=temp.value_counts().head(10)
fig=plt.figure()
ax=plt.axes()
x=y.index
plt.xticks=x
plt.yticks=y

plt.plot(x,y,color="red")





# ## Languages overview:
# * there are 38 languages ,in which developers code.
# * javascript is the most popular language.
# * hack is the least popular language.

# In[65]:


temp=pd.DataFrame(so_data.LanguageWorkedWith.dropna().str.split(";").tolist()).stack()
y=temp.value_counts().head(10)
fig=plt.figure()
ax=plt.axes()
x=y.index.unique()


plt.scatter(x,y,color="red",alpha=0.8)


# Dear all , this is my first entry into kaggle.com .stay blessed , more to come. 
# more challenges mean more ways to learn.

# ## <a id="5-13"> 5.13 Top databases ?</a>

# In[4]:


dbs=pd.DataFrame(so_data.DatabaseWorkedWith.dropna().str.split(";").tolist()).stack()
top_db=dbs.value_counts().head(10)
data=[go.Bar(
    x=top_db.index,
    y=top_db.values,
    marker=dict(color=random_colors(10)),
)]
layout=go.Layout(title="top 10 databases")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# ## The Most Popular database is MySQL

# ## <a id="5-14"> 5.14 Database of the Next year?</a>

# In[5]:


nextdb=pd.DataFrame(so_data.DatabaseDesireNextYear.dropna().str.split(";").tolist()).stack()
temp=nextdb.value_counts().head(10)
data=[go.Bar(
    x=temp.index,
    y=temp.values,
    marker=dict(color=random_colors(10)),
)]
layout=go.Layout(title="Popular databases of next year")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# ## Mysql will retain its 1'st position ,wheras sql server will go from 2nd to 4rth position.

# In[3]:


db_2018=pd.DataFrame(so_data.DatabaseWorkedWith.dropna().str.split(";").tolist()).stack()
db_2019=pd.DataFrame(so_data.DatabaseDesireNextYear.dropna().str.split(";").tolist()).stack()
top_2018=db_2018.value_counts().head(10)
top_2019=db_2019.value_counts().head(10)

trace1=go.Bar(
    x=top_2018.index,
    y=top_2018.values,
  
    name="Top 10 databases of 2018",
)
trace2=go.Bar(
    x=top_2019.index,
    y=top_2019.values,
  
    name="Top 10 databases of 2019",
)
data=[trace1,trace2]
layout=go.Layout(title="databases analysis 2018 vs 2019")

fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# ## ** MongoDB ** will be at 2nd position in 2019

# ## <a id="5-15"> 5.15 Top 10 plateforms of 2018</a>

# In[10]:


plate_form=pd.DataFrame(so_data.PlatformWorkedWith.dropna().str.split(";").tolist()).stack()
top_pf18=plate_form.value_counts().head(10)
data=[go.Bar(
    x=top_pf18.index,
    y=top_pf18.values,
    marker=dict(color=random_colors(10)),
)]
layout=go.Layout(title="top 10 plateforms of 2018")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# ## <a id="5-16">5.16 Top 10 plateforms of 2019</a>

# In[9]:


next_pf=pd.DataFrame(so_data.PlatformDesireNextYear.dropna().str.split(";").tolist()).stack()
top_pf19=next_pf.value_counts().head(10)
data=[go.Bar(
    x=top_pf19.index,
    y=top_pf19.values,
    marker=dict(color=random_colors(10)),
)]
layout=go.Layout(title="top 10 plateforms of 2019")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# In[ ]:





# In[ ]:




