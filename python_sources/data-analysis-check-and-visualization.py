#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #visualization
import seaborn as sns #visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#import some packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Import my data from csv file
df = pd.read_csv('../input/heart.csv')
df.head(13)


# 

# **Attribute Information:**
# 
#         age 
#         sex 0:female 1:male
#         chest pain type (4 values)
#         resting blood pressure
#         serum cholestoral in mg/dl
#         fasting blood sugar > 120 mg/dl
#         resting electrocardiographic results (values 0,1,2)
#         maximum heart rate achieved
#         exercise induced angina
#         oldpeak = ST depression induced by exercise relative to rest
#         the slope of the peak exercise ST segment
#         number of major vessels (0-3) colored by flourosopy
#         thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
#         target 0:false 1:true (related to have or not heart disease) 

# # I WILL CHECK IF MY DATA IS CLEAN 

# In[ ]:


# i confirm that do not exist NaN in my dataframe if exists appear error with message
assert df.notnull().all().all(),"exists some NaN"


# In[ ]:


#I will check if the sex column has only with 0,1 values
#first way 

check1=(all((df.sex ==1)| (df.sex==0)))
print("The column has only the values that i want? ",check1)


# In[ ]:


#I will check if the sex column has only with 0,1 values
#second way 
check2=df.sex.unique()# find the unique values of sex column
if ((check2[0]==1)|(check2[0]==0)|(check2[1]==1)|(check2[1]==0)):
    print("The columns contains only 0,1 values.")


# In[ ]:


#i will check the column of age contains only integers from 0 to 120
check_age=[]#define an empty variable
for age in df.age:
    if (age>0)&(age<120):
        check_age.append(1)
    else:
        check_age.append("PROBLEM MUST BE SOLVED")
if all (x ==1 for x in check_age):
    print("Clear the age column")


# In[ ]:


#I will try to check for many columns with same in puts(function)
#0,1 values take the columns:sex fbs exang target
#returning YES must include only 0,1 values into the column,a lot of times exist values that dont related to my data for this reason i must check them!
def check3(col_name):
    uni=col_name.unique()
    if ((uni[0] == 1) | (uni[0] == 0) | (uni[1] == 1) | (uni[1] == 0)):
        return "YES"


# In[ ]:


# i run the function for the columns that must have values 0,1 to check if is TRUE
#the columns that must have 0,1 are sex,target,fbs,exang
print("Is the sex column clear? ",check3(df.sex))
print("Is the target column clear? ",check3(df.target))
print("Is the fbs column clear?",check3(df.fbs))
print("Is the exang column clear ?",check3(df.exang))


# In[ ]:


# i will examinize some others columns with different way, with assert
#if the columns take some useless values the python appears a message 
assert ((df.thal==0) | (df.thal==1)|(df.thal==2)|(df.thal==3)).all(),"check this column"
assert ((df.cp==0) | (df.cp==1)|(df.cp==2)|(df.cp==3)).all(),"check this column"
assert ((df.restecg==0) | (df.restecg==1)|(df.restecg==2)).all(),"check this column"
assert ((df.slope==0) | (df.slope==1)|(df.slope==2)).all(),"check this column"


# **MY DATA IS CLEAR AND READY FOR ANALYSIS**

# In[ ]:


# informations about my DataFrame
df.info()


# In[ ]:


#columns names of my Dataframe
df.columns


# In[ ]:


#observe some interest things from statistics
df.describe()


# In[ ]:


#i see the types of my columns
print(df.dtypes)


# In[ ]:


#count of target per gender
sns.countplot(df.target,hue=df.sex, palette="Set3")
plt.title("Count of Diseases and not")
plt.xlabel("target")
plt.ylabel("Count")
plt.show()


# In[ ]:


#i want to group the cholesterol to three parts.Low,natural,high
conditions=[(df.chol<200),
            (df.chol<239),
            (df.chol<1000)]
choices=['low','natural','High']

df.chol=np.select(conditions, choices)
df.chol.tail(5)


# In[ ]:


#"chol" must become categorical variable

df.chol=df["chol"].astype("category")
df.dtypes #Observe that tranform to categorical columns


# In[ ]:


#relationship between  age and level of cholesterin

sns.stripplot(x="chol", y="age",order=["low","natural","High"],data=df)
plt.ylabel("age")
plt.xlabel("levels of cholesterol")
plt.title('relation between age and type of cholesterol')
plt.show()
#with hue seperate famales=0 and males=1
sns.stripplot(x="age",y="chol",data=df,order=["High","natural","low"],hue='sex')
plt.ylabel('levels of cholesterol')
plt.xlabel("age")
plt.title("relation between age and type of chol by gender")
plt.show()


# In[ ]:


#I use swarmplot to observe how many for each age and level of chol will have target=1(disease) or target=0(not disease)
sns.swarmplot(x="chol",y="age",data=df,hue="target")
plt.title("cholesterol - age frequency with hue=target ")
plt.show()
#for this plot we can figure out that the most of people have high levels of cholesterin ,then natural and low
#as well people with high chol and age between 40-55 huge percent have heart disease
#and the people 55 and above with high chol dont have heart disease.For me is a little strangle this output


# In[ ]:


# I will find the probability of the three stages of cholesterin

(df.chol.value_counts(normalize=True)).plot(kind='barh',color='red')
plt.ylabel('level of cholisteroin')
plt.xlabel('PDF')
plt.title('participants distribution of cholesterol')# i see that about the 50% have high holisterin
plt.show()
#observe that the 50% of the participants to survey have high levels of cholesterol


# In[ ]:


#creation one pivot table(because have duplicate data) with values the mean of age for each "fbs" and "sex"
print(df.fbs.value_counts())
piv=df.pivot_table(index="fbs",columns="sex",values="age",aggfunc="mean")
piv


# In[ ]:


#the plot of the pivot table from above
piv.plot(kind="bar",rot=0)
plt.ylabel("age mean")
plt.xlabel(" no                     fbs                   yes")
plt.title("mean of age per gender that have or not fbs",color="tomato")
plt.show()


# In[ ]:


#create a pie chart with percentage of exercise or not("exang")
print(df.exang.value_counts())
labels=["excercise","no excercise"]
colors=["yellow","red"]
values=[df.exang.value_counts()[1],df.exang.value_counts()[0]]
plt.pie(values,labels=labels,colors=colors,autopct="%.lf%%")
plt.show()


# In[ ]:


#i will show the relation between chol and exercise(1:yes  0:no)
cross=pd.crosstab(df.chol,df.exang,normalize=True)#the result is probability
print(cross)
#Now is clear that if someone work out  have less probability to have high level chol
#low levels of holesterol must be more for people who work out , but dont happend ,the main reason is that only the 33% have exercised


# In[ ]:


#plot bar  the above
cross.plot(kind="bar",rot=0,color=["red","black"])
plt.show()


# In[ ]:


#corelation between all
sns.heatmap(df.corr())
plt.show()


# In[ ]:


#the same but with correlation values and other background
zf,ax = plt.subplots(figsize=(18,18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, cmap="BuPu", ax=ax)
plt.show()


# In[ ]:


# see the scatter plot  and observe that "thalach" is higher for younger people
df.plot(x="age",y="thalach",kind="scatter", c="black", marker='^',alpha= 0.9)
plt.show()


# In[ ]:


#I will find for each cp all the others column's mean
df_m=df.groupby("cp").mean()
df_m


# In[ ]:


#i will  keep only some of the columns 
df_m=df_m[["age","trestbps"]]
df_m


# In[ ]:


#frequency of all my data
df.hist(bins=50, figsize=(12,14), color= "green", grid=False)
plt.show()


# In[ ]:


#i will create a Data_Frame that have the mean of "trestbps" per "age"
df_new=df.groupby("age").mean()
df_new.tail(10)


# In[ ]:


# i will keep only the columns that i am intersted in
df_new=df_new[["trestbps","thalach"]]
df_new.head(4)


# In[ ]:


# plot now.. i will plot and the mean of "trestbps" for each age
#1
df_new.plot(kind="line",use_index=True,y="trestbps",fontsize=15,c="tomato")
plt.ylabel("trestbps")
plt.xlabel("age")
plt.title("mean of trestbps by age")
# i will plot and the mean of "thalach" for each age
df_new.plot(kind="line",use_index=True,y="thalach",c="yellow")
plt.legend(loc="upper right") 
plt.ylabel("trestbps")
plt.xlabel("age")
plt.title("mean of trestbps by age")
plt.show()

#It is clear for the first plot that when a person become older the trestbps(something like blood pressure) increase ,in age approximately 68 years old havw one huge reduce 


# In[ ]:


#takes the data for the people that work out ..exang=1 (true)
#I examine the people who exercised and their oldpeakand slope
df_ex=df[df.exang==1]
df_ex.oldpeak.head(5)
sns.stripplot(x="slope",y="oldpeak",data=df_ex)
plt.title("related to person who exercise oldpeak -slope",color="blue")
plt.xlabel("low                                      medium                                         high")
plt.ylabel("oldpeak(depression)")
sns.despine(bottom = True)#remove the x-axis
plt.show()

#persons who exercised hard(slope high) dont have depression


# In[ ]:


#Age  for different types of chest pain BOXPLOT
plt.figure(figsize = (10,5))
plot = sns.boxplot(x = "cp", y = "age", data = df)
plot.set_title("Age  for different types of chest pain")


# In[ ]:


#regression "thalach" and "age"
g = sns.lmplot(x="age", y="thalach", hue="sex", data=df)
sns.despine(left= True,bottom = True)#without axis


# In[ ]:


#regression for all types of cholesterol
sns.lmplot(x="age", y="thalach", hue="sex", col="chol",data=df, height=6, aspect=.4, x_jitter=.1)


# In[ ]:




