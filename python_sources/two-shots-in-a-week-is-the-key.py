#!/usr/bin/env python
# coding: utf-8

# **Student Alcohol Consumpiton Data** is used to practice the seaborn library. 
# 
# The data was obtained in a survey of students of math in secondary school. It contains a lot of interesting social, gender and study information about students. 
# 
# If you have any helpful advice or developer criticism please get in touch with me.
# 
# Gmail=hdogukanince@gmail.com
# 
# About Data; 
# 
# 1. school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# 2. sex - student's sex (binary: 'F' - female or 'M' - male)
# 3. age - student's age (numeric: from 15 to 22)
# 4. Dalc - workday alcohol consumption 
# 5. Walc - weekend alcohol consumption 
# 6. health - current health status 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/student-mat.csv')


# > First 5 index of tha data.

# In[ ]:


data.head()


# > Last five of the data.

# In[ ]:


data.tail()


# > Columns of the data.

# In[ ]:


data.columns


# In[ ]:


data.shape


# > Checking for the Walc column whether there is any non-null data or not.

# In[ ]:


assert data["Walc"].notnull().all()


# > To count how many values are in the guardian column.

# In[ ]:


data["guardian"].value_counts(dropna=False)


# > To see the total alcohol consumption of students and add a new column, whose name is alcohol level and which shows us the level of alcohol consumption.

# In[ ]:


data["total_alcohol"] = data["Dalc"] + data["Walc"]
data["total_alcohol"] = data["total_alcohol"].astype(int)

threshold = sum(data.total_alcohol)/len(data.total_alcohol)
data["alcohol_level"] = ["high" if i > threshold else "low" for i in data.total_alcohol]
print(data.loc[:3, "alcohol_level"])


# > To see the default graphic size

# In[ ]:


fig_size=plt.rcParams["figure.figsize"]
print("Current Size:" , fig_size)


# > **1) HeatMap**

# In[ ]:


plt.figure(figsize = (13,13))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cbar=True)
plt.show()


# > **2) CountPlot**

# In[ ]:


sns.countplot(data.sex)
plt.title("Gender", color="blue")


# > **3) PieChart**

# In[ ]:


labels=data.age.value_counts().index
colors=["lime","orange","blue","yellow","purple","red","black","grey"]
explode=[0,0,0,0,0,0,0,0]
sizes=data.age.value_counts().values
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, colors=colors, explode=explode,autopct='%2.2f%%')
plt.title("Age of Empires -Just Kidding- Age of Students", color="lime",fontsize=15)
plt.show()


# > **4) PointPlot**

# In[ ]:


school_list = (data["school"].unique())
final_grade = []
for i in school_list:
    x = data[data["school"]==i]
    general_final_grade = sum(x.G3)/len(x)
    final_grade.append(general_final_grade)
    
df = pd.DataFrame({"school_list":school_list, "final_grade":final_grade})
new_data = (df["final_grade"].sort_values(ascending=True)).index.values
sorted_data = df.reindex(new_data)

sns.pointplot(x='school_list',y='final_grade',data=sorted_data,color='lime',alpha=0.8)
plt.xlabel("Schols")
plt.ylabel("Final Grade")
plt.title("Avarage Success Rate for Schools")
plt.show()


# > **5) BoxPlot**

# In[ ]:


sns.set(style="whitegrid")
sns.boxplot(x="school",y="total_alcohol",hue="alcohol_level", data=data)
plt.show()


# In[ ]:


data.loc[:, ["Dalc", "Walc", "total_alcohol"]].plot()
plt.show()


# > **6) BarPlot**

# In[ ]:


list = []
for i in range(11):
    list.append(len(data[data.total_alcohol == i]))
ax = sns.barplot(x = [0,1,2,3,4,5,6,7,8,9,10], y = list)
plt.ylabel('Number of Students')
plt.xlabel('Total alcohol consumption')
plt.show()


# > **7) PairPlot**

# In[ ]:


g1= data["G1"]
g2= data["G2"]
g3= data["G3"]
totAlc= data["total_alcohol"]
newdf = pd.concat([g1,g2,g3,totAlc],axis=1)
sns.pairplot(data=newdf)
plt.show()


# > **7) JointPlot**

# In[ ]:


sns.jointplot(data.G3, data.total_alcohol, kind="kde", height=5)
plt.show()


# > **8) LmPlot**

# In[ ]:


sns.lmplot(x="total_alcohol", y="G3", data=data)
plt.show()


# > Great bucket of thanks to Datai Team --->[https://www.kaggle.com/kanncaa1](http://)
# 
# > ***Thank you for your time to read.***
# 
