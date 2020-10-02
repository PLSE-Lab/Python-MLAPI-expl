#!/usr/bin/env python
# coding: utf-8

# > <h1><b>TABLE OF CONTENTS</b></h1>
# <ul>
#     <a href='#1'><li>1.Data Overview</li></a>
# </ul>
# <ul>
#     <a href='#2'><li>2.Data Analysis and Cleaning</li></a>
# </ul>
# <ul>
#     <a href='#3'><li>3.Data Visualization</li></a>
#         <ul>
#              <a href='#4'><li>3.0.Correlation Map</li></a>
#              <a href='#5'><li>3.1.Line Plot</li></a>
#              <a href='#6'><li>3.2.Scatter Plot</li></a>
#              <a href='#7'><li>3.3.Count Plot</li></a>
#              <a href='#8'><li>3.4.Count Plot & Melting The Data</li></a>
#              <a href='#9'><li>3.5.Histogram Plot</li></a>
#              <a href='#10'><li>3.6.Strip Plot</li></a>
#              <a href='#11'><li>3.7.Factor Plot</li></a>
#         </ul>
# </ul>
# <ul>
#     <a href='#12'><li>4.Conclusion</li></a>
#     <a href='#13'><li>5.References</li></a>
# </ul>

# <p id='1'><h2><b>1.Data Overview</b></h2></p>
# 
# Hello everyone,  
# In this kernel we will analyze Students Performance in Exam dataset.  
# Let's start with what we have in this dataset.
# We have 8 columns and 1000 rows in this dataset they are like : 
# * gender
# * race/ethnicity
# * parental level of education
# * lunch
# * test preparation course
# * math score
# * reading score
# * writing score

# <p id='2'><h2><b>2.Data Analysis and Cleaning</b></h2></p>

# First, we import libraries.  
# We import numpy and pandas libraries to analyse the data.  
# We import matplot and seaborn libraries to visualize the data.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))


# Reading csv file.

# In[ ]:


data = pd.read_csv("../input/StudentsPerformance.csv")


# Informations about the data.

# In[ ]:


data.info()  


# To look first 5 rows and last 5 rows in the data we use head and tail as below.

# In[ ]:


data.head()

#data.tail()


# To look the data's columns.

# In[ ]:


data.columns


# Looking for summation of missing values in the data and cleaning them if there are some missing values.

# In[ ]:


data.isnull().sum()

data.dropna(inplace=True,axis=0) # Inplace for saving to the data after dropped.


# Size of the data.

# In[ ]:


data.shape


# Informations about the data like min, max, mean etc.

# In[ ]:


data.describe()


# To look direct proportion or inverse proportion between columns.

# In[ ]:


data.corr() 


# <h3><b>Concatenating a Data</b></h3>
# We separate the data to 3 datas, with random rows in it. After that we concatenate these 3 datas. We can use this method to understand the data easily. If ignore_index=False then rows' index values continue unordered. For this method we will use 9 samples.

# In[ ]:


data_head = data.head(3)
data_middle = data.iloc[500:503,:]
data_tail = data.tail(3)
concatenated_data = pd.concat([data_head,data_middle,data_tail],axis=0,ignore_index=False)
concatenated_data


# <p id='3'><h2><b>3.Data Visualization</b></h2></p>

# <p id='4'><h3><b>3.0.Correlation Map</b></h3></p>
# 
# With correlation map we can look correlation between columns easily.  
# For example we can say the highest direct proportion is between math score and total score so in short if you have a high math score so your total score is high too.
# 

# In[ ]:


f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(),annot=True,linewidths =.5,fmt=".1f",ax=ax)
plt.show()


# <p id='5'><h3><b>3.1.Line Plot</b></h3></p>
# We use "sample()" to select 20 random rows from the data and random_state to select the rows in the same order. 

# In[ ]:


data_random = data.sample(20,random_state=42)
data_random.index = np.arange(0,len(data_random))


# In[ ]:


data1 = data_random
data1["math score"].plot(figsize=(12,4.5),kind = "line",color = "red",label="Math Score",linewidth = 1,alpha=1,grid=True,linestyle='-.')
data1["reading score"].plot(kind = "line",color="green",label="Reading Score",linewidth=1,alpha=1,grid=True,linestyle="-")
data1["writing score"].plot(kind = "line",color = "black",label = "Writing Score",linewidth=1,alpha=1,grid=True,linestyle=":")
plt.legend(loc="upper right")
plt.xlabel('Students',FontSize = 10,color = "purple")
plt.ylabel("Scores",FontSize = 10, color = "green")
plt.title("Scores for 20 Students with Line Plot",FontSize = 12)
plt.savefig("Graphic.png")
plt.show() 


# <p id='6'><h3><b>3.2.Scatter Plot</b></h3></p>
# Here we will look correlation between genders and their mathematic scores.  
# We create a new dataframe and make females 1, males 0 then to plot on scatter we make gender column's type integer.

# In[ ]:


data2 = data.copy()
data2.gender = ["1" if each == "female" else "0" for each in data2.gender]
data2.gender = data2["gender"].astype(int)


# In[ ]:


data2.plot(kind = 'scatter',x="math score",y="gender",color="black",figsize=(17,8))
plt.xlabel("Mathematic Score",FontSize = 18)
plt.ylabel("Gender",FontSize = 18)
plt.title("Scatter Plot",FontSize = 20)
plt.show()


# <p id='7'><h3><b>3.3.Count Plot</b></h3></p>
# The visualization above, we have changed gender to integer type. So for this visualization we make integer values to F and M first.

# In[ ]:


data2.gender = ["F" if each == 1 else "M" for each in data2.gender]


# In[ ]:


p = sns.countplot(x='gender', data = data2, palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=0) 
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()


# In this visualization we will look for students whose scores are under and above 85 depending of Parental Level of Education. We add a new column to out first data named "Above_85".

# In[ ]:


data["Above_85"] = np.where(((data["math score"]>=85) & (data["reading score"]>=85) & (data["writing score"]>=85)),"A","U")


# In[ ]:


ploting = sns.countplot(x='parental level of education',data = data,hue = "Above_85", palette="Blues_d")
plt.setp(ploting.get_xticklabels(), rotation=45) 
plt.xlabel("Parental Level of\nEducation")
plt.ylabel("Count")
plt.show()


# I am a student at Sakarya University because of that in this visualization we will use letter grade system of Sakarya University.  
# We separate marks by groups. And we will use lambda function.

# In[ ]:


data["total_score"] = (data["math score"]+data["reading score"]+data["writing score"])/3


# In[ ]:


def Degree(mark):
    if mark>=90:
        return "AA"
    elif 90>mark>=85:
        return "BA"
    elif 85>mark>=80:
        return "BB"
    elif 80>mark>=75:
        return "CB"
    elif 75>mark>=65:
        return "CC"
    elif 65>mark>=58:
        return "DC"
    elif 58>mark>=50:
        return "DD"
    else:
        return "FF"
    
data["Mark_Degree"] = data.apply(lambda x: Degree(x["total_score"]),axis=1)
data.head()


# In[ ]:


p = sns.countplot(x='Mark_Degree',data = data,order=['AA','BA','BB','CB','CC','DC','DD','FF'],palette="Blues_d")
plt.setp(p.get_xticklabels(), rotation=0) 
plt.xlabel("Mark Degrees")
plt.ylabel("Number of Students")
plt.show()


# <p id='8'><h3><b>3.4.Count Plot & Melting The Data</b></h3></p>
# In this visualization we will use "melt()" method to make an example of tidying data and to look it broadly. Maybe this method makes us analyse the data easier.

# In[ ]:


data_melt = data
melting = pd.melt(frame = data_melt,id_vars= ["gender","lunch"],value_vars =["math score","reading score","writing score"])
melting.drop("lunch",axis=1,inplace=True)
melting.rename(index=str,columns={"variable":"ScoreTable","value":"Values","gender":"Gender"},inplace=True)
melting.Gender = ["F" if each == "female" else "M" for each in melting.Gender]
melting.head()


# In[ ]:


p = sns.countplot(x='ScoreTable', data = melting,hue="Gender" ,palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 
plt.xlabel("Scores")
plt.ylabel("Count")
plt.show()


# <p id='9'><h3><b>3.5.Histogram Plot</b></h3></p>
# In this visualization we will look for frequencies of math, reading and writing scores.

# In[ ]:


data2["math score"].plot(kind = 'hist',bins = 50,figsize = (9,5),alpha=0.9,color="gray")
data2["reading score"].plot(kind = 'hist',bins = 50,alpha=0.7,color="blue")
data2["writing score"].plot(kind = 'hist',bins = 50,alpha=0.5,color="green")
plt.xlabel("Math, Reading & Writing Scores")
plt.legend()
plt.show()


# <p id='10'><h3><b>3.6.Strip Plot</b></h3></p>

# In[ ]:


data["Total_Score"] = data["math score"]/3 + data["writing score"]/3 + data["reading score"]/3
data.head()


# In[ ]:


sns.stripplot(x="parental level of education",y='Total_Score',data=data)
plt.xticks(rotation=45)
plt.xlabel("Parental Level of Education")
plt.ylabel("Total Score")
plt.show()


# <p id='11'><h3><b>3.7.Factor Plot</b></h3></p>

# In[ ]:


sns.factorplot(x='gender', y='Total_Score', hue='test preparation course', data=data, kind='bar')
plt.xlabel("Gender")
plt.ylabel("Total Score")
plt.show()


# <p id='12'><h2><b>4.Conclusion</b></h2></p>
# * In this dataset we looked at students' exam results, we compared math, writing and reading scores.
# * We made some visualizations using matplot and seaborn libraries and we used some of the data analyzing methods.
# * So if you have any questions or advises i will be pleased with it.

# <p id='13'><h2><b>5.References</b></h2></p>
# 
# https://www.kaggle.com/spscientist/students-performance-in-exams
# 
# https://www.kaggle.com/yunusulucay/kaggle-markdown-language-basics/edit
# 
# https://www.kaggle.com/spscientist/student-performance-in-exams
# 
# https://www.kaggle.com/venky73/predicting-student-percentage
# 
# https://www.kaggle.com/kralmachine/seaborn-tutorial-for-beginners
