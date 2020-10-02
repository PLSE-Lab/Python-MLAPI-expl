#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this kernel, I used some seaborn visualization methods to build better understanding for this data. I wrote down conclusions for the same aim. This is my first kernel, I hope you enjoy it!

# # Contents
# <font color="blue">
# 1. [Relation Between Student Scores and Other Factors](#1)
#     * [Relation Between Groups and Student Scores](#2)
#         * [Math Scores](#3)
#         * [Reading Scores](#4)
#         * [Writing Scores](#5)
#         * [Conclusion](#6)
#     * [Relation Between Gender and Student Scores](#7)
#         * [Math Scores](#8)
#         * [Reading Scores](#9)
#         * [Writing Scores](#10)
#         * [Overall Scores](#11)
#         * [Conclusion](#12)
#     * [Relation Between Parental Education Level and Student Scores](#13)
#         * [Math Scores](#14)
#         * [Reading Scores](#15)
#         * [Writing Scores](#16)
#         * [Conclusion](#17)
#     * [Relation Between Lunch Type and Student Scores](#18)
#         * [Math Scores](#19)
#         * [Reading Scores](#20)
#         * [Writing Scores](#21)
#         * [Conclusion](#22)
#     * [Relation Between Test Preparation Course and Scores of Students](#23)
#         * [Math Scores](#24)
#         * [Reading Scores](#25)
#         * [Writing Scores](#26)
#         * [Conclusion](#27)
# 2. [Correlation Between Exam Scores](#28)
#     * [General Display of Correlation Between Exam Scores by Heatmap](#29)
#     * [Math-Reading](#30)
#     * [Math-Writing](#31)
#     * [Reading-Writing](#32)
#     * [Conclusion](#33)

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")


# In[ ]:


data


# In[ ]:


data.columns


# In[ ]:


data["math_score"] = data["math score"]
del data["math score"]
data["reading_score"] = data["reading score"]
del data["reading score"]
data["writing_score"] = data["writing score"]
del data["writing score"]


# In[ ]:


data.head()


# <a id="1" ></a><br>
# # Relation Between Student Scores and Other Factors

# <a id="2" ></a><br>
# # Relation Between Groups and Student Scores
# * Math Scores
# * Reading Scores
# * Writing Scores
# * Conclusion

# <a id="3" ></a><br>
# ## Math Scores

# In[ ]:


data.math_score=data.math_score.astype(float)
grouplist=list(data["race/ethnicity"].unique())
groupscore=[]
for i in grouplist:
    x = data[data["race/ethnicity"]==i]
    mathav = sum(x.math_score)/len(x)
    groupscore.append(mathav)
mathdata = pd.DataFrame({"group":grouplist, "groupscore":groupscore})
new_index=(mathdata.groupscore.sort_values(ascending=False)).index.values
sorted_math=mathdata.reindex(new_index)

plt.figure(figsize=(10,8))
sns.barplot(x=sorted_math.group, y=sorted_math.groupscore)
plt.xticks(rotation=0)
plt.xlabel("Groups")
plt.ylabel("Average")
plt.title("Averages of Math Scores of Groups")
plt.show()


# According to plot above, the ranking of groups' scores on math is: E - D - C - B - A

# <a id="4" ></a><br>
# ## Reading Scores

# In[ ]:


# Preparation of data
data.reading_score=data.reading_score.astype(float)
grouplist=list(data["race/ethnicity"].unique())
groupscore=[]
for i in grouplist:
    x = data[data["race/ethnicity"]==i]
    readingav = sum(x.reading_score)/len(x)
    groupscore.append(readingav)
readingdata = pd.DataFrame({"group":grouplist,"groupscore":groupscore})
new_index=(readingdata.groupscore.sort_values(ascending=False)).index.values
sorted_reading=readingdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_reading.group, y=sorted_reading.groupscore)
plt.xlabel("Groups")
plt.ylabel("Average")
plt.title("Averages of Reading Scores of Groups")
plt.show()


# According to plot above, the ranking of groups' scores on reading is: E - D - C - B - A

# <a id="5" ></a><br>
# ## Writing Scores

# In[ ]:


# Preparation of data
data.writing_score=data.writing_score.astype(float)
grouplist=list(data["race/ethnicity"].unique())
groupscore=[]
for i in grouplist:
    x=data[data["race/ethnicity"]==i]
    writingav=sum(x.writing_score)/len(x)
    groupscore.append(writingav)
writingdata=pd.DataFrame({"group":grouplist, "groupscore":groupscore})
new_index=(writingdata.groupscore.sort_values(ascending=False)).index.values
sorted_writing=writingdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_writing.group, y=sorted_writing.groupscore)
plt.xlabel("Groups")
plt.ylabel("Average")
plt.title("Averages of Writing Scores of Groups")
plt.show()


# According to plot above, the ranking of groups' scores on writing is: E - D - C - B - A

# <a id="6" ></a><br>
# ## Conclusion

# According to all plots, the ranking of groups' general scores is: E - D - C - B - A
# 
# The most successful group is Group A
# 
# The less successful group is Group B

# In[ ]:


data.head()


# <a id="7" ></a><br>
# # Relation Between Gender and Student Scores
# * Math Scores
# * Reading Scores
# * Writing Scores
# * Overall Scores
# * Conclusion

# <a id="8" ></a><br>
# ## Math Scores

# In[ ]:


# Preparation of data
data.math_score=data.math_score.astype(float)
genderlist=list(data.gender.unique())
genderscore=[]
for i in genderlist:
    x = data[data.gender==i]
    mathav=sum(x.math_score)/len(x)
    genderscore.append(mathav)
mathdata=pd.DataFrame({"gender":genderlist, "genderscore":genderscore})
new_index=(mathdata.genderscore.sort_values(ascending=False)).index.values
sorted_math=mathdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_math.gender, y=sorted_math.genderscore)
plt.xlabel("Genders")
plt.ylabel("Average")
plt.title("Averages of Math Scores of Genders")
plt.show()


# According to plot above, male scores are higher than female scores in math.

# <a id="9" ></a><br>
# ## Reading Scores

# In[ ]:


# Preparation of data
data.reading_score=data.reading_score.astype(float)
genderlist=list(data.gender.unique())
genderscore=[]
for i in genderlist:
    x=data[data.gender==i]
    readingav=sum(x.reading_score)/len(x)
    genderscore.append(readingav)
readingdata=pd.DataFrame({"gender":genderlist, "genderscore":genderscore})
new_index=(readingdata.genderscore.sort_values(ascending=False)).index.values
sorted_reading=readingdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_reading.gender, y=sorted_reading.genderscore)
plt.xlabel("Genders")
plt.ylabel("Average")
plt.title("Averages of Reading Scores of Genders")
plt.show()


# According to plot above, female scores are higher than male scores in reading.

# <a id="10" ></a><br>
# ## Writing Scores

# In[ ]:


# Preparation of data
data.writing_score=data.writing_score.astype(float)
genderlist=list(data.gender.unique())
genderscore=[]
for i in genderlist:
    x = data[data.gender==i]
    writingav=sum(x.writing_score)/len(x)
    genderscore.append(writingav)
writingdata=pd.DataFrame({"gender":genderlist, "genderscore":genderscore})
new_index=(writingdata.genderscore.sort_values(ascending=False)).index.values
sorted_writing=writingdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_writing.gender, y=sorted_writing.genderscore)
plt.xlabel("Genders")
plt.ylabel("Average")
plt.title("Averages of Writing Scores of Genders")
plt.show()


# According to plot above, female scores are higher than male scores in writing.

# <a id="11" ></a><br>
# ## Overall Scores

# In[ ]:


# Preparation of data
data.writing_score=data.writing_score.astype(float)
data.reading_score=data.reading_score.astype(float)
data.math_score=data.math_score.astype(float)
genderlist=list(data.gender.unique())
genderscore=[]
for i in genderlist:
    x=data[data.gender==i]
    overallav= ((sum(x.math_score)+sum(x.reading_score)+sum(x.writing_score))/len(x))/3 # I divided them equally because there is no information about their percentages
    genderscore.append(overallav)
overalldata=pd.DataFrame({"gender":genderlist,"genderscore":genderscore})
new_index=(overalldata.genderscore.sort_values(ascending=False)).index.values
sorted_overall=overalldata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_overall.gender, y=sorted_overall.genderscore)
plt.xlabel("Genders")
plt.ylabel("Average")
plt.title("Averages of Overall Scores of Genders")
plt.show()


# According to plot above, female scores are higher than male scores in overall evaluation.(Because of the lack of information, score weights are determined as equal: Math -> 33.3%, Reading -> 33.3%, Writing -> 33.3%)

# ## Conclusion
# Males are better than females in math, females are better than males in language exams(reading & writing).

# In[ ]:


data.head()


# In[ ]:


data["parental level of education"].value_counts()


# In[ ]:


data["parent_education"] = data["parental level of education"]
del data["parental level of education"]


# In[ ]:


data.head()


# <a id="12" ></a><br>
# # Relation Between Parental Education Level and Student Scores

# * Parental Eduation Level
# * Math Scores
# * Reading Scores
# * Writing Scores
# * Conclusion

# <a id="13" ></a><br>
# ## Parental Education Level

# In[ ]:


plt.figure(figsize=(14,9))
sns.countplot(data.parent_education)
plt.xlabel("Education Level")
plt.ylabel("Frequency")
plt.title("Frequency of Education Levels of Parents")
plt.show()


# In[ ]:


labels=data.parent_education.value_counts().index
colors=["blue","orange","green","red","purple","grey"]
sizes=data.parent_education.value_counts().values

plt.figure(figsize=(8,8))
plt.pie(sizes, colors=colors, labels=labels, autopct="%1.1f%%")
plt.title("The Education Levels of Parents")
plt.show()


# <a id="14" ></a><br>
# ## Math Scores

# In[ ]:


# Preparation of data
data.math_score=data.math_score.astype(float)
parentlist=list(data.parent_education.unique())
parentscore=[]
for i in parentlist:
    x=data[data.parent_education==i]
    mathav=sum(x.math_score)/len(x)
    parentscore.append(mathav)
mathdata=pd.DataFrame({"education_level":parentlist, "level_score":parentscore})
new_index=(mathdata.level_score.sort_values(ascending=False)).index.values
sorted_math=mathdata.reindex(new_index)

# Visualization
plt.figure(figsize=(14,9))
sns.barplot(x=sorted_math.education_level, y=sorted_math.level_score)
plt.xlabel("Education Level")
plt.ylabel("Average Score")
plt.title("Relation Between Parental Education Level and Math Exam Scores of Students")
plt.show()


# <a id="15" ></a><br>
# ## Reading Scores

# In[ ]:


# Preparation of data
data.reading_score=data.reading_score.astype(float)
parentlist=list(data.parent_education.unique())
parentscore=[]
for i in parentlist:
    x=data[data.parent_education==i]
    readingav=sum(x.reading_score)/len(x)
    parentscore.append(readingav)
readingdata=pd.DataFrame({"education_level":parentlist, "level_score":parentscore})
new_index=(readingdata.level_score.sort_values(ascending=False)).index.values
sorted_reading=readingdata.reindex(new_index)

# Visualization
plt.figure(figsize=(14,9))
sns.barplot(x=sorted_reading.education_level, y=sorted_reading.level_score)
plt.xlabel("Education Level")
plt.ylabel("Average Score")
plt.title("Relation Between Parental Education Level and Reading Exam Scores of Students")
plt.show()


# <a id="16" ></a><br>
# ## Writing Scores

# In[ ]:


# Preparation of data
data.writing_score=data.writing_score.astype(float)
parentlist=list(data.parent_education.unique())
parentscore=[]
for i in parentlist:
    x=data[data.parent_education==i]
    writingav=sum(x.writing_score)/len(x)
    parentscore.append(writingav)
writingdata=pd.DataFrame({"education_level":parentlist, "level_score":parentscore})
new_index=(writingdata.level_score.sort_values(ascending=False)).index.values
sorted_writing=writingdata.reindex(new_index)

# Visualization
plt.figure(figsize=(14,9))
sns.barplot(x=sorted_writing.education_level, y=sorted_writing.level_score)
plt.xlabel("Education Level")
plt.ylabel("Average Score")
plt.title("Relation Between Parental Education Level and Writing Exam Scores of Students")
plt.show()


# <a id="17" ></a><br>
# ## Conclusion
# All the plots above have shown that there is a clear correlation between parental education level and exam scores of students.

# <a id="18" ></a><br>
# # Relation Between Lunch Type and Student Scores
# * Math Scores
# * Reading Scores
# * Writing Scores
# * Conclusion

# <a id="19" ></a><br>
# ## Math Scores

# In[ ]:


# Preparation of data
data.math_score=data.math_score.astype(float)
lunchlist=list(data.lunch.unique())
lunchscore=[]
for i in lunchlist:
    x=data[data.lunch==i]
    mathav=sum(x.math_score)/len(x)
    lunchscore.append(mathav)
mathdata=pd.DataFrame({"lunch":lunchlist, "lunchscore":lunchscore})
new_index=(mathdata.lunchscore.sort_values(ascending=False)).index.values
sorted_math=mathdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_math.lunch, y=sorted_math.lunchscore)
plt.xlabel("Lunch Type")
plt.ylabel("Average Score")
plt.title("Relation Between Lunch Type and Math Exam Scores of Students")
plt.show()


# <a id="20" ></a><br>
# ## Reading Scores

# In[ ]:


# Preparation of data
data.reading_score=data.reading_score.astype(float)
lunchlist=list(data.lunch.unique())
lunchscore=[]
for i in lunchlist:
    x=data[data.lunch==i]
    readingav=sum(x.reading_score)/len(x)
    lunchscore.append(readingav)
readingdata=pd.DataFrame({"lunch":lunchlist, "lunchscore":lunchscore})
new_index=(readingdata.lunchscore.sort_values(ascending=False)).index.values
sorted_reading=readingdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_reading.lunch, y=sorted_reading.lunchscore)
plt.xlabel("Lunch Type")
plt.ylabel("Average Score")
plt.title("Relation Between Lunch Type and Reading Exam Scores of Students")
plt.show()


# <a id="21" ></a><br>
# ## Writing Scores

# In[ ]:


# Preparation of data
data.writing_score=data.writing_score.astype(float)
lunchlist=list(data.lunch.unique())
lunchscore=[]
for i in lunchlist:
    x=data[data.lunch==i]
    writingav=sum(x.writing_score)/len(x)
    lunchscore.append(writingav)
writingdata=pd.DataFrame({"lunch":lunchlist, "lunchscore":lunchscore})
new_index=(writingdata.lunchscore.sort_values(ascending=False)).index.values
sorted_writing=writingdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_writing.lunch, y=sorted_writing.lunchscore)
plt.xlabel("Lunch Type")
plt.ylabel("Average Score")
plt.title("Relation Between Lunch Type and Writing Exam Scores of Students")
plt.show()


# <a id="22" ></a><br>
# ## Conclusion
# The plots above have shown that students who take standard lunch get better results than ones who get free/reduced lunch.

# In[ ]:


data.head()


# In[ ]:


data["test_prep"]=data["test preparation course"]
del data["test preparation course"]


# In[ ]:


data.head()


# In[ ]:


data.test_prep.value_counts()


# <a id="23" ></a><br>
# # Relation Between Test Preparation Course and Scores of Students
# * Math Scores
# * Reading Scores
# * Writing Scores
# * Conclusion

# <a id="24" ></a><br>
# ## Math Scores

# In[ ]:


# Preparation of data
data.math_score=data.math_score.astype(float)
preplist=list(data.test_prep.unique())
prepscore=[]
for i in preplist:
    x=data[data.test_prep==i]
    mathav=sum(x.math_score)/len(x)
    prepscore.append(mathav)
mathdata=pd.DataFrame({"prep":preplist, "score":prepscore})
new_index=(mathdata.score.sort_values(ascending=False)).index.values
sorted_math=mathdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_math.prep, y=sorted_math.score)
plt.xlabel("Preparation Course")
plt.ylabel("Average")
plt.title("Relation Between Test Preparation Course and Math Exam Scores")
plt.show()


# <a id="25" ></a><br>
# ## Reading Scores

# In[ ]:


# Preparation of data
data.reading_score=data.reading_score.astype(float)
preplist=list(data.test_prep.unique())
prepscore=[]
for i in preplist:
    x=data[data.test_prep==i]
    readingav=sum(x.reading_score)/len(x)
    prepscore.append(readingav)
readingdata=pd.DataFrame({"prep":preplist, "score":prepscore})
new_index=(readingdata.score.sort_values(ascending=False)).index.values
sorted_reading=readingdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_reading.prep, y=sorted_reading.score)
plt.xlabel("Preparation Course")
plt.ylabel("Average")
plt.title("Relation Between Test Preparation Course and Reading Exam Scores")
plt.show()


# <a id="26" ></a><br>
# ## Writing Scores

# In[ ]:


# Preparation of data
data.writing_score=data.writing_score.astype(float)
preplist=list(data.test_prep.unique())
prepscore=[]
for i in preplist:
    x=data[data.test_prep==i]
    writingav=sum(x.writing_score)/len(x)
    prepscore.append(writingav)
writingdata=pd.DataFrame({"prep":preplist, "score":prepscore})
new_index=(writingdata.score.sort_values(ascending=False)).index.values
sorted_writing=writingdata.reindex(new_index)

# Visualization
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_writing.prep, y=sorted_writing.score)
plt.xlabel("Preparation Course")
plt.ylabel("Average")
plt.title("Relation Between Test Preparation Course and Writing Exam Scores")
plt.show()


# <a id="27" ></a><br>
# ## Conclusion
# The plots above have clearly shown that the students who completed test preparation course did better in all exams.

# <a id="28" ></a><br>
# # Correlation Between Exam Scores

# <a id="29" ></a><br>
# ## General Display of Correlation Between Exam Scores by Heatmap

# In[ ]:


data.corr()


# In[ ]:


f, ax=plt.subplots(figsize=(9,9))
sns.heatmap(data.corr(),vmax=1, vmin=0, annot=True, linewidths=0.5, linecolor="red", fmt=" .1f", ax=ax)
plt.show()


# There are obvious correlations between all features.

# <a id="30" ></a><br>
# ## Math-Reading

# In[ ]:


data["ID"]=data.index


# ### Point Plot

# In[ ]:


df=data.iloc[:100,:]
f,ax=plt.subplots(figsize=(20,10))
sns.pointplot(data=df, x="ID", y="math_score", color="blue", alpha=0.8)
sns.pointplot(data=df, x="ID", y="reading_score", color="red", alpha=0.8)
plt.text(85,8, "Math Score", color="black", fontsize=20)
plt.text(82,8, "----", color="blue", fontsize=20)
plt.text(85,2, "Reading Score", color="black", fontsize=20)
plt.text(82,2, "----", color="red", fontsize=20)
plt.xlabel("Students", fontsize=20)
plt.ylabel("Score", fontsize=20)
plt.title("Correlation Between Math and Reading Scores")
plt.show()


# ### Joint Plot

# In[ ]:


sns.jointplot(data.reading_score, data.math_score, kind="kde", size=8)
plt.show()


# ### Lm Plot

# In[ ]:


sns.lmplot(data=data.iloc[:200,:], x="reading_score", y="math_score")
plt.xlabel("Reading Score")
plt.ylabel("Math Score")
plt.title("Correlation Between Math and Reading Scores")
plt.show()


# ### Kde Plot

# In[ ]:


sns.kdeplot(data.reading_score, data.math_score, shade=True, color="blue", cut=5)
plt.show()


# ### Conclusion
# According to charts above, there is highly considerable correlation between math and reading exam scores.

# <a id="31" ></a><br>
# ## Math-Writing

# ### Point Plot

# In[ ]:


df=data.iloc[:100,:]
f, ax=plt.subplots(figsize=(20,10))
sns.pointplot(data=df, x="ID", y="math_score", color="blue", alpha=0.8)
sns.pointplot(data=df, x="ID", y="writing_score", color="orange", alpha=0.8)
plt.text(85,8, "Math Score", color="black", fontsize=20)
plt.text(82,8, "----", color="blue", fontsize=20)
plt.text(85,2, "Writing Score", color="black", fontsize=20)
plt.text(82,2, "----", color="orange", fontsize=20)
plt.xlabel("Students", fontsize=20)
plt.ylabel("Score", fontsize=20)
plt.title("Correlation Between Math and Writing Scores")
plt.show()


# ### Joint Plot

# In[ ]:


sns.jointplot(data.math_score, data.writing_score, kind="kde", size=8, color="orange")
plt.show()


# ### Lm Plot

# In[ ]:


sns.lmplot(data=data.iloc[:200,:], x="math_score", y="writing_score")
plt.xlabel("Math Score")
plt.ylabel("Writing Score")
plt.title("Correlation Between Math and Writing Scores")
plt.show()


# ### Kde Plot

# In[ ]:


sns.kdeplot(data.math_score, data.writing_score, shade=True, color="orange", cut=5)
plt.show()


# ### Conclusion
# According to charts above, there is highly considerable correlation between math and writing exam scores.

# <a id="32" ></a><br>
# ## Reading-Writing

# ### Point Plot

# In[ ]:


df=data.iloc[:100,:]
f,ax=plt.subplots(figsize=(20,10))
sns.pointplot(data=df, x="ID", y="reading_score", color="red", alpha=0.8)
sns.pointplot(data=df, x="ID", y="writing_score", color="orange", alpha=0.8)
plt.text(85,14, "Reading Score", color="black", fontsize=20)
plt.text(82,14, "----", color="red", fontsize=20)
plt.text(85,8, "Writing Score", color="black", fontsize=20)
plt.text(82,8, "----", color="orange", fontsize=20)
plt.xlabel("Students")
plt.ylabel("Score")
plt.title("Correlation Between Reading and Writing Scores")
plt.show()


# ### Joint Plot

# In[ ]:


sns.jointplot(data.writing_score, data.reading_score, kind="kde", size=8, color="red")
plt.show()


# ### Lm Plot

# In[ ]:


sns.lmplot(data=data.iloc[:200,:], x="writing_score", y="reading_score")
plt.xlabel("Writing Score")
plt.ylabel("Reading Score")
plt.title("Correlation Between Reading and Writing Scores")
plt.show()


# ### Kde Plot

# In[ ]:


sns.kdeplot(data.writing_score, data.reading_score, shade=True, color="red", cut=5)
plt.xlabel("Writing Score")
plt.ylabel("Reading Score")
plt.title("Correlation Between Reading and Writing Scores")
plt.show()


# ### Conclusion
# According to charts above, there is highly considerable correlation between reading and writing exam scores.

# <a id="33" ></a><br>
# ## Conclusion
# As we see at charts above there are clear correlations between exam results, especially between reading and writing)
