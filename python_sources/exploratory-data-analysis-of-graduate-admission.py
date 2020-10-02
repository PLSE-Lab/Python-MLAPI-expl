#!/usr/bin/env python
# coding: utf-8

# ## Context-
# ### This dataset is created for prediction of Graduate Admissions from an Indian perspective.
# ### The dataset contains several parameters which are considered important during the application for Masters Programs. The parameters included are :
# ### 1. GRE Scores ( out of 340 ) 2. TOEFL Scores ( out of 120 ) 3. University Rating ( out of 5 ) 4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 ) 5. Undergraduate GPA ( out of 10 ) 6. Research Experience ( either 0 or 1 ) 7. Chance of Admit ( ranging from 0 to 1 )
# 
# ## Acknowledgements-
# ### This dataset is inspired by the UCLA Graduate Dataset. The test scores and GPA are in the older format. The dataset is owned by Mohan S Acharya.
# 
# ## Inspiration-
# ### This dataset was built with the purpose of helping students in shortlisting universities with their profiles. The predicted output gives them a fair idea about their chances for a particular university.
# 

# In[ ]:


# importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#importing the data
df= pd.read_csv('../input/Admission_Predict.csv')


# In[ ]:


#getting the head of data
df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# ### Observation - The data conist of 400 students with 9 attributes.

# In[ ]:


# getting the data types of data
df.dtypes


# In[ ]:


## getting the mean,count etc of data
df.describe()


# ## Observation-
# ### The maximum GRE score = 340, minimum = 290, mean score = 316.
# ### The maximum TOEFL score=120, minimum=92,mean score = 107.
# ### The maximum University Rating is 5, minimum rating is 1, mean rating = 3.
# ### Maximum C.G.P.A score = 9.92, minimum= 6.8, mean = 8.5
# ### Maximum No. of research = 1 ,minimum = 0.
# 

# In[ ]:


##bar plot
plt.subplots(figsize=(10,4))
sns.barplot(x="Research",y="Chance of Admit " ,data=df)
plt.subplots(figsize=(18,4))
sns.barplot(x="GRE Score",y="Chance of Admit ",data=df)
plt.subplots(figsize=(18,4))
sns.barplot(x="TOEFL Score",y="Chance of Admit ",data=df)
plt.subplots(figsize=(10,4))
sns.barplot(x="SOP",y="Chance of Admit ",data=df)
plt.subplots(figsize=(10,4))
sns.barplot(x="University Rating",y="Chance of Admit ",data=df)
plt.figure(figsize=(10,4))
sns.barplot(x= "LOR ",y="Chance of Admit ",data=df)
plt.subplots(figsize=(23,4))
sns.barplot(x="Chance of Admit ",y="CGPA",data=df)


# ## Observation-
# ### The candidates , who have higher marks in GRE has greater chance of admission.
# ### The students with higher TOEFL Score has greater chances of admission.
# ### Higher the rating of University, higher is the chance of getting admission.
# ### Higher the rating of Statement of Pupose, Higher is the chance of getiing Admission.
# ### The cadidates who have higher number of letter of recommendation has a greater chance of admission.
# ### Candidates having higher CGPA has greater chance of admission.
# 

# In[ ]:


#Pair plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="Chance of Admit ",
             vars=['CGPA','TOEFL Score','GRE Score'], size=5)
plt.show()


# ## Observation
# ### The students who have a high score in TOEFL,GRE,CGPA has a greater chance of admission.
# ### The students who have higher GRE, TOEFL score has more chances of admission.

# In[ ]:


## Boxplots
plt.subplots (figsize=(18, 5))
sns.boxplot(x="GRE Score",y="Chance of Admit ",data=df)

plt.subplots (figsize=(15,5))
sns.boxplot(x="TOEFL Score",y="Chance of Admit ",data=df)

plt.subplots (figsize=(22,5))
sns.boxplot(x="Chance of Admit ",y="CGPA",data=df)

plt.subplots(figsize=(15,5))
sns.boxplot(x="LOR ",y="Chance of Admit ",data=df)

plt.subplots(figsize=(15,5))
sns.boxplot(x="SOP",y="Chance of Admit ",data=df)

plt.subplots(figsize=(15,5))
sns.boxplot(x="Research",y="Chance of Admit ",data=df)


            


# ## Observation-
# ### The candidates , who have higher marks in GRE has greater chance of admission.
# ### The students with higher TOEFL Score has greater chances of admission.
# ### Higher the rating of University, higher is the chance of getting admission.
# ### Higher the rating of Statement of Pupose, Higher is the chance of getiing Admission.
# ### The cadidates who have higher number of letter of recommendation has a greater chance of admission.
# ### Candidates having higher CGPA has greater chance of admission.

# ## Heat Map

# In[ ]:


#getting the correlation of data
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()


# ## Observation-
# ### The 3 most important features for admission to the Master: CGPA, GRE SCORE, and TOEFL SCORE.
# ### The 3 least important features for admission to the Master: Research, LOR, and SOP.

# >  ***Any queries and recommendation are welcomed,please do comment.***
# 

# In[ ]:




