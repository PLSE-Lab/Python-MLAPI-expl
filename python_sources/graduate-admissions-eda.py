#!/usr/bin/env python
# coding: utf-8

# Fall is the busiest season for graduate admissions. And this dataset by Mohan S Acharya has been created to predict the chances of a graduate admission based on certain parameters. I would try to explore the dataset and present my findings.
# 
# First of all, let's load the required libraries.

# In[20]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# Now, let's load the data and have a look at it.

# In[58]:


df = pd.read_csv("../input/Admission_Predict.csv")
df.head()


# Let's a look at the data types, the shape of the data and whether there are any null values.

# In[59]:


df.info()


# - Number of rows - 400
# - Number of columns - 9
# - Number of columns with NULL values - 0
# 
# There are no null values in the dataset, so that's good. But the number of samples is just 400, which is low. It has 9 columns out of which the `Serial No.`  field seems redundant to me. So let's drop that column first, and then have a look at the distribution of the other columns.

# In[61]:


df.drop("Serial No.", axis=1, inplace=True)
df.columns = df.columns.map(lambda x: x.strip().replace(" ","_"))


# In[63]:


for column in df.columns:
    if(df[column].nunique() < 10):
        fig = sns.countplot(df[column])
    else:
        fig = sns.distplot(df[column], hist_kws={"alpha": 1}, kde=False)
    plt.title("Distribution of " + column.replace("_", " "))
    plt.show()


# A couple of thoughts that I am having currently after seeing these plots:
# - My guess is that the University Rating column relates to Ambitious, Moderate and Safe standards that the students chose. 
# - I am guessing the SOP field is how the students felt their SOPs where. It is interesting to note that out of 400 students, none rated their SOP to be in the range 2.5-3.0. Can I bucket this column too into Low, Medium and High?
# - We have a fairly equal representation of applications with and without research.

# Let's now have a look at how various marks vary with the `Chance of Admit` column. 

# In[73]:


fig = sns.scatterplot(x="CGPA", y="Chance_of_Admit", data=df)
plt.title("CGPA vs Chance of Admit")
plt.show()

fig = sns.scatterplot(x="TOEFL_Score", y="Chance_of_Admit", data=df)
plt.title("TOEFL Score vs Chance of Admit")
plt.show()

fig = sns.scatterplot(x="GRE_Score", y="Chance_of_Admit", data=df)
plt.title("GRE Score vs Chance of Admit")
plt.show()


# In[75]:


df[["CGPA", "GRE_Score", "TOEFL_Score", "Chance_of_Admit"]].corr()


# All the three columns have a strong positive correlation with the `Chance of Admit` column which is evident from the graphs too. If the `Chance of Admit` column has been estimated by students, then the correlation can be easily explained. As a student if you have good marks, you would estimate your chances of getting an admit to be on the higher end.
# 
# Now let's have a look at the `SOP`, the `LOR` and the `Research`  columns.

# In[97]:


fig = sns.boxplot(x="SOP", y="Chance_of_Admit", data=df)
plt.title("SOP vs Chance of Admit")
plt.show()

fig = sns.boxplot(x="LOR", y="Chance_of_Admit", data=df)
plt.title("LOR vs Chance of Admit")
plt.show()

fig = sns.boxplot(x="Research", y="Chance_of_Admit", data=df)
plt.title("Research vs Chance of Admit")
plt.show()


# A similar linearly increasing trend for `LOR` and `SOP` columns too. A couple of outliers for sure. People with low `LOR` and `SOP` score having relatively higher chances of admit. It might be interesting to look into these specific candidates. People with `Research` have a higher chance of admit. Overall no suprising results.

# It might be interesting to see the relationship between `CGPA` and `LOR`, assuming the LORs are mostly from college professors. So let's have a look.

# In[92]:


fig = sns.boxplot(x="LOR", y="CGPA", data=df)
plt.title("CGPA vs LOR")
plt.show()

fig = sns.boxplot(x="LOR", y="CGPA", data=df[df.Research == 0])
plt.title("CGPA vs LOR - Without Research")
plt.show()

fig = sns.boxplot(x="LOR", y="CGPA", data=df[df.Research == 1])
plt.title("CGPA vs LOR - With Research")
plt.show()


# There is definitely some relation in one's `CGPA` and the strength of `LOR`. If a student has a `CGPA` of 8.5 or more, there are high chances of him securing a decent LOR.
# 
# 
# However, these graphs look interesting. It turns out that there is a relation between `CGPA` and `Research`. 

# In[89]:


fig = sns.boxplot(x="Research", y="CGPA", data=df)
plt.title("Research Experience vs CGPA")
plt.show()


# This is interesting. So students having `Research` experience apparently have higher `CGPA` scores. I wonder why?
# 
# Let's also a take a look at `Research` vs `LOR` and `Research` vs `SOP`.

# In[96]:


fig = sns.boxplot(x="Research", y="LOR", data=df)
plt.title("Research Experience vs LOR")
plt.show()

fig = sns.boxplot(x="Research", y="SOP", data=df)
plt.title("Research Experience vs SOP")
plt.show()


# So having a research experience is critical for a high `SOP` score and a high `LOR` score. This kind of makes sense since students without research experience might think that their SOPs are not strong enough. But I am not sure why is research indicative of a high LOR score. Maybe the supervisors got to know the students better. But then for an application, one needs to submit multiple LORs. Is this column an average of all of them? 
# 
# Anyway, let's take a look to see if there is a relation between a student's `SOP` and `TOEFL_Score`.

# In[90]:


fig = sns.boxplot(x="SOP", y="TOEFL_Score", data=df)
plt.title("SOP vs TOEFL Score")
plt.show()


# Although there is a relation between `TOEFL Score` and `SOP`, the whiskers extend out quite far. So a good `TOEFL Score` doesn't guarentee a good `SOP` score. And that's actually valid.
# 
# Let's take a look to see if there is a relation between a student's `CGPA` and `GRE_Score`.

# In[91]:


fig = sns.scatterplot(x="CGPA", y="GRE_Score", data=df)
plt.title("CGPA vs GRE Score")
plt.show()


# I have heard people saying that GRE is extremely easy, but it seems to have a correlation with `CGPA`. Students who have scored well in their undergraduates, have scored well in GRE too. 

# Well, that's the end of the EDA. Here are my conclusions:
# - Higher CGPA, GRE, and TOEFL marks translate to increased chances of admit.
# - Research and CGPA seem to be related. Students with Research experience seem to have a higher CGPA.
# - Research is also related to higher LOR scores and higher SOP scores. So having a research background seems to be crucial.
# - A higher score in GRE is strongly correlated with a higher CGPA.
# - A high score in TOEFL doesn't translate into a strong SOP, proving that a SOP is so much more than syntactically correct English. 

# Thank you for going through the EDA. Please let me know if I missed out on something, or if I should have done something better. And if you liked my work, please UPVOTE. 
