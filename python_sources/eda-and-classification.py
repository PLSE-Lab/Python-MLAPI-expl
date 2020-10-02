#!/usr/bin/env python
# coding: utf-8

# <img src='https://lh3.googleusercontent.com/proxy/01gmxojDBtlEdhXze1ednoxHwaehrotjwAsORg8jFS3_x0q8zEDgBbKvm86pOEybGJvymIHTVC0D0m2F8HS45hLdOiQQFCxxQlzt6Ts0HzE54EPj'>

# Welcome to my notebook!!!. First we will try to find out insights from the data and figure out which factors are resposnsible for a student to get placed. We will find out answers of many questions:-
# *  Does Gender affect for getting placed?
# *  Which stream students are getting more placed and which stream students are mostly not placed?
# *  Does marks really matter?
# *  Students from which degree and specialisation are getting placed
# 
# And we will fit logistic regression and random forest to data to predict whether a student will get placed or not.

# # Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


df.drop(['sl_no'],axis=1,inplace=True)


# In[ ]:


df['status'].values[df['status']=='Not Placed'] = 0 
df['status'].values[df['status']=='Placed'] = 1
df.status = df.status.astype('int')


# In[ ]:


df.head()


#    #  Drawing insights

# In[ ]:


sns.heatmap(df.isnull(), cbar=False) #finding columns having nan values


# In[ ]:


df['salary'] = df['salary'].replace(np.nan, 0) #Replace Nan with 0


# In[ ]:


df.describe()


# In[ ]:


sns.pairplot(df,kind='reg')


# In[ ]:


plt.figure(figsize=(14,12))
sns.heatmap(df.corr(), linewidth=0.2, cmap="YlGnBu", annot=True)


# Surprisingly MBA percenrage and E-test precentage are the least significant variable that affect the placement outcome.

# Lets explore more!!

# In[ ]:


sns.countplot(df['status'])


# About 2 times more students got placed as compared to not placed.

# # Significance of Gender

# Lets See whether Gender plays any significant role or not

# In[ ]:


sns.countplot(df['gender']) #to see the composition of gender
plt.show()


# In[ ]:


sns.countplot(df['status'],hue=df['gender'])
plt.show()


# In[ ]:


Boys_placed=100
Total_Boys=140
Boys_placed_prop=Boys_placed/Total_Boys


Girls_placed=50
Total_Girls=70
Girls_placed_prop=Girls_placed/Total_Girls

print('Proportion of boys got placed: ') , 
print(Boys_placed_prop)

print('\nProportion of girls got placed: ') , 
print(Girls_placed_prop)


# 1. From the above plots we see that Boys are getting more placements and the ratio of boys to girls(placements) is about 100:50.
# 2. Total number girls not placed are 30 and Total number of boys not placed are 40
# 3. But proportion of getting placed is same for both, i.e 71% . Therefore **Gender DOESN'T MATTER** for getting placed. 
# 

# # Significance of Marks

# In[ ]:


fig,axes = plt.subplots(3,2, figsize=(20,12))
sns.barplot(x='status', y='ssc_p', data=df, ax=axes[0][0])
sns.barplot(x='status', y='hsc_p', data=df, ax=axes[0][1])
sns.barplot(x='status', y='degree_p',data=df, ax=axes[1][0])
sns.barplot(x='status', y='etest_p',data=df, ax=axes[1][1])
sns.barplot(x='status', y='mba_p', data=df, ax=axes[2][0])
fig.delaxes(ax = axes[2][1]) 


# In[ ]:


sns.catplot(x="status", y="ssc_p", data=df,kind="swarm")
sns.catplot(x="status", y="hsc_p", data=df,kind="swarm")
sns.catplot(x="status", y="degree_p", data=df,kind="swarm")
sns.catplot(x="status", y="etest_p", data=df,kind="swarm")
sns.catplot(x="status", y="mba_p", data=df,kind="swarm")
plt.show()


# 1. The students who have scored less than 60 percent in 10th or 12th or degree are mostly not getting placed. So Yes, Percentage matter for placement as we have seen from the HeatMap in previous section.
# 2. But, Higher Percentage necessarily doesn't guarantee a Placement.
# 3. etest and MBA percentage doesn't matter.
# 

# # Significance of work experience

# In[ ]:


df.groupby(['workex','status']).count()['salary']


# In[ ]:


sns.countplot(df['workex']) #to see the composition of work experience
plt.show()


# In[ ]:


sns.countplot(df['status'],hue=df['workex'])
plt.show()


# In[ ]:


Y_placed=64
Total_Y=74
Y_placed_prop=Y_placed/Total_Y


N_placed=84
Total_N=141
N_placed_prop=N_placed/Total_N

print('Proportion of student with work experience got placed: ') , 
print(Y_placed_prop)

print('\nProportion of students with No work experience got placed: ') , 
print(N_placed_prop)


# From above we can infere that:- 
#  * Students having work experience are more likely to get placed (86%).
#  * 59% of studnets having no work experience got selected.  

# **Therefore having work experience will help you for getting placed**

# # Signifiance of Specialization

# In[ ]:


df.groupby(['specialisation','status']).count()['salary']


# In[ ]:


sns.countplot(df['specialisation']) #to see the composition of work experience
plt.show()


# In[ ]:


sns.countplot(df['status'],hue=df['specialisation'])
plt.show()


# In[ ]:


MH_placed=53
Total_MH=95
MH_placed_prop=MH_placed/Total_MH


MF_placed=95
Total_MF=120
MF_placed_prop=MF_placed/Total_MF

print('Proportion of student from Market and HR got placed: ') , 
print(MH_placed_prop)

print('\nProportion of students from Market and finance got placed: ') , 
print(MF_placed_prop)


# From above plots and calculation 
# * Marketing and Finance Specialization is Most Demanded by Corporate.
# 

# # Does school Board really matters?

# **SSC board**

# In[ ]:


df.groupby(['ssc_b','status']).count()['salary']


# In[ ]:


sns.countplot(df['ssc_b'])
plt.show()


# In[ ]:


sns.countplot(df['status'],hue=df['ssc_b'])
plt.show()


# In[ ]:


print('Proportion of student having central board in SSC got placed: ') , 
print(78/(78+38))

print('\nProportion of students having other board in SSC got placed: ') , 
print(70/(70+29))


# **HSC board**

# In[ ]:


df.groupby(['hsc_b','status']).count()['salary']


# In[ ]:


sns.countplot(df['hsc_b'])
plt.show()


# In[ ]:


sns.countplot(df['status'],hue=df['hsc_b'])
plt.show()


# In[ ]:


print('Proportion of student having central board in HSC got placed: ') , 
print(57/(57+27))

print('\nProportion of students having other board in HSC got placed: ') , 
print(91/(91+40))


# We got the following insights from the above graphs and calculations:-
#  
# 
# 
# 

#  **Percentage of students got placed from:**
# * Central board in **SSC** : 67%
# * Other board in **SSC** : 71%
# * Central board in **HSC** : 67%
# * Other board in **HSC** : 69%
# 

# We can see here that there is not significant difference between selecting board in SSC or HSC will help you getting placed. So it doesn't matter whether you you did your education form central board or other board.

# # Does Stream of HSC matters?

# In[ ]:


df.groupby(['hsc_s','status']).count()['salary']


# In[ ]:


sns.countplot(df['status'],hue=df['hsc_s'])
plt.show()


# In[ ]:


print('Proportion of commerce student got placed: ') , 
print(79/(79+34))

print('\nProportion of science students got placed: ') , 
print(63/(63+28))


# Its better to opt for Commerce or Science,but commerce stream will be benefcial as it will help a student to build strong foundation in Business studies. Students from Arts are very few therefore we are ignoring them. 

# # Degree Type

# In[ ]:


df.groupby(['degree_t','status']).count()['salary']


# In[ ]:


sns.countplot(df['status'],hue=df['degree_t'])
plt.show()


# In[ ]:


print('Proportion of Comm&Mgmt student got placed: ') , 
print(102/(43+102))

print('\nProportion of Sci&Tech students got placed: ') , 
print(41/(41+18))


# Same as that of HSC stream.
# Its better to opt for Comm&Mgmt or Sci&Tech,but comm&Mgmt stream will be benefcial. Students from others are very few therefore we are ignoring them.

# # So, who are the Ideal Students who are more likely to get placed?

# * Class 10 and class 12 percentage should be greater than 60% from any board
# * You should have opted for commerce in HSC (science will also work)
# * Degree from Commerce and managemnet will help you (Science and Tech will also work)
# * Having work experience is like strawberry on cake. Students having work experience are more likley to get placed
# 
# 

# # ML algorithm

# **Coverting categorical columns to dummy variables**

# In[ ]:


def cat_to_num(data_x,col):
    dummy = pd.get_dummies(data_x[col])
    del dummy[dummy.columns[-1]]#To avoid dummy variable trap
    data_x= pd.concat([data_x,dummy],axis =1)
    return data_x


# **Selecting relevant columns which determines whether a student is placed or not**

# In[ ]:


df.columns


# In[ ]:


df_x=df[[ 'ssc_p', 'hsc_p',  'hsc_s', 'degree_p',
       'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']]


# In[ ]:


for i in df_x.columns:
    if df_x[i].dtype ==object:
        print(i)
        df_x =cat_to_num(df_x,i)


# In[ ]:


df_x.drop(['workex','specialisation','hsc_s','degree_t'],inplace =True,axis =1)


# In[ ]:


y = df['status']
X = df_x


# **Splitting into training set and test test**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# **Fitting logistic regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[ ]:


y_predict = model.predict(X_test)

model.score(X_test,y_test)


# **Fitting random forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)


# In[ ]:


y_predict = model.predict(X_test)

model.score(X_test, y_test)


# # THANK YOU!!!
