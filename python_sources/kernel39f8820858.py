#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# This is my first attempt at python.
# The results derived are as follows:
# 1)Highest number of people have enrolled for Web Development and Business Finance.
# 2)There are highest number of all level, beginner level courses and equal number of intermediate courses for web development and business finance.
# 3)Taking into account the number of paid and free courses, we see that only 310 number of courses are unpaid.
# 4)There is a positive relation between number of reviews and number of subscribes which shows that trust of people on the course, to an extent depends on the amount of reviews and they prefer going through the reviews before enrolling for the course.
# 5)Around 531 and 598 number of people have chosen graphic design and musical instrument courses when the price is less than 80.
# 6)As the price gets higher, suppose above 100, more number have taken web development and business finance courses.
# 7)It is interesting to note that the amount of free courses are higher for business finance and web development as the price is high and has highest number of subscribers.
# 8)Also there is a course named "Piano hand coordination" with 0 number of subscribers.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df=pd.read_csv("../input/udemy-courses/udemy_courses.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


sns.set_style("whitegrid")


# In[ ]:


g=sns.FacetGrid(df,col="subject")
g.map(plt.hist,"num_subscribers",bins=10)


# In[ ]:


fig,ax=plt.subplots(figsize=(10,6))
ax=sns.heatmap(df.corr(),annot=True,cmap="viridis")


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x="subject",data=df)


# In[ ]:


df["course_title"].value_counts().head(10)


# In[ ]:


df["price"].value_counts().head(15).sort_values(ascending=False)


# In[ ]:


sns.countplot(x="subject",hue="level",data=df)
plt.legend(loc='best', bbox_to_anchor=(0.5, 0.5))


# In[ ]:


levels=df["level"].value_counts().drop("52")
levels


# In[ ]:


ax = df['level'].value_counts().drop("52").plot(kind ='bar', figsize = (6,4), width = 0.8)
ax.set_title('Levels vs Amount of courses', fontsize = 15)
ax.set_ylabel('Amount of courses', fontsize = 15)
ax.set_xlabel('Levels', fontsize = 15)


# In[ ]:


sns.jointplot(x="num_reviews",y="num_subscribers",data=df,kind="scatter",color="red",space=0.4,height=8,ratio=4)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df[(df["subject"]=="Web Development") & (df["price"]>"100")].count()


# In[ ]:


df[(df["subject"]=="Graphic Design") & (df["price"]<"80")].count()


# In[ ]:


df[(df["subject"]=="Business Finance") & (df["price"]>"100")].count()


# In[ ]:


df[(df["subject"]=="Musical Instruments") & (df["price"]<"80")].count()


# In[ ]:


df[["course_title","num_subscribers"]].max().head()


# In[ ]:


df[["course_title","num_subscribers"]].min().head()


# In[ ]:




