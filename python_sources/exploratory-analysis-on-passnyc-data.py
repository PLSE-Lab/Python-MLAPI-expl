#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Importing CSV into a data frame and plotting the scatter plot

# Q. Is school income dependent on economic need?                                                                       
# Ans. The below scatter plot clearly shows direct (inverse) relationship between school income and economic need. Except for a few outlier scenarios, higher the economic need index, lower is the school income estimate. (corelation = -.89)

# In[ ]:


filename = "../input/2016 School Explorer.csv"
data1 = pd.read_csv(filename)

df = pd.DataFrame(data1, columns=["Economic Need Index", "School Income Estimate"])
plt.figure(figsize=(10,10))
df = df.dropna()
df['School Income Estimate'] = df['School Income Estimate'].replace('[\$,]', '', regex=True).astype(float)
income_estimate= df['School Income Estimate']

max_val = max(income_estimate)

plt.yticks(np.arange(0, max_val, 3000))
plt.ylabel("Student Income Estimate")
plt.xlabel("Economic need index")

plt.scatter(df['Economic Need Index'],df['School Income Estimate'])
plt.show()


# Q. Which are the cities with high number of schools?                                                                                     
# A. The below count plot shows "Brooklyn city" having highest number of schools (>400). After Brooklyn, Bronx and New York are the cities with high number of schools (> 200)

# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(y="City", data=data1)


# Q. What is the general trend among the schools with respect to lowest grade and highest grade serviced?                                                                                                                                                       
# A. Analyzing the below heat map, it can be observed that majority of schools (about 50%) service between Pre-KinderGarden/Kindergarden and 5th grade. Also, many schools (about 20%) service from 6th grade to 8th grade.
# 

# In[ ]:


plt.figure(figsize=(10,10))
grade_scale = data1[['SED Code', 'Grade Low', 'Grade High']].pivot_table(values= 'SED Code',
                            index = 'Grade Low', columns = 'Grade High' , aggfunc = np.size
                                                                         ,fill_value = 0)
sns.heatmap(grade_scale, annot = True ,fmt ='d' )


# Q. Compare average ELA proficiency among the schools with average Math proficiency.                          
# 
# 
# 
# A. Looking at the below KDE plot, it appears that average Math proficiency among schools is higher than average ELA proficiency. Majority of schools have average ELA proficiency between 2 and 3 while average Math proficiency is between 2 and 3.5

# In[ ]:


sns.kdeplot(data1['Average ELA Proficiency'] , shade = True ,color ='g')
plt.subplot(111)

sns.kdeplot(data1['Average Math Proficiency'] , shade = True ,color ='b')
plt.subplot(111)


#     Q. Is there any relationship between Collaborative Teachers, Supportive Environment and Trust?     
#     A. Below pairplot shows the relationship among the three. There is a direct relation among them. Higher the Collarborative Teachers, higher is the Supportive Environment and higher is the Trust and vice versa.

# In[ ]:


plt.figure(figsize=(7,7))
df1 = pd.DataFrame(data1, columns=["Collaborative Teachers %", "Supportive Environment %","Trust %"])
df1 = df1.dropna()
df1
df1['Collaborative Teachers %']=df1['Collaborative Teachers %'].str.rstrip('%').astype(float)
df1['Trust %']=df1['Trust %'].str.rstrip('%').astype(float)
df1['Supportive Environment %']=df1['Supportive Environment %'].str.rstrip('%').astype(float)
sns.pairplot(df1)


# Q. Are the students underachieving or meeting their targets? Similarly, how is the trust rating among the schools?
# 
# A.i) Looking at the below count plots of Student Achievement Rating and Trust Rating, majority school's (about 50%) students are meeting their targets. Only a very small fraction of students not meeting their targets.                       
# ii) If we observe the countplot of Trust rating, most of the schools (>75%) are either meeting their trust targets or exceeding them. Only a very small fraction of schools are not meeting their trust targets.

# In[ ]:


plt.figure(figsize=(8,8))
plt.subplot(211)
sns.countplot(x= "Student Achievement Rating" , data =data1 ,palette= "Blues")
plt.subplot(212)
sns.countplot(x= "Trust Rating" , data =data1 ,palette= "Blues")


# Q. Is chronic absence among schools related to the community?
# 
# A. If we observe below joint plots, more the number of asian and white community people in the school, less is the chronic absence in such schools. Schools with high number of black and hispanic community people are found to be having high chronic absence.
# 

# In[ ]:


df2 = pd.DataFrame(data1, columns=["Percent Black","Percent of Students Chronically Absent","Percent Asian","Percent Hispanic","Percent White","Percent ELL"])
df2 = df2.dropna()
df2['Percent Black']=df2['Percent Black'].str.rstrip('%').astype(float)
df2['Percent Hispanic']=df2['Percent Hispanic'].str.rstrip('%').astype(float)
df2['Percent ELL']=df2['Percent ELL'].str.rstrip('%').astype(float)
df2['Percent White']=df2['Percent White'].str.rstrip('%').astype(float)
df2['Percent Asian']=df2['Percent Asian'].str.rstrip('%').astype(float)
df2['Percent of Students Chronically Absent']=df2['Percent of Students Chronically Absent'].str.rstrip('%').astype(float)

sns.jointplot(y="Percent of Students Chronically Absent", x="Percent Black", data=df2)
sns.jointplot(y="Percent of Students Chronically Absent", x="Percent Asian", data=df2)
sns.jointplot(y="Percent of Students Chronically Absent", x="Percent White", data=df2)
sns.jointplot(y="Percent of Students Chronically Absent", x="Percent ELL", data=df2)
sns.jointplot(y="Percent of Students Chronically Absent", x="Percent Hispanic", data=df2)

