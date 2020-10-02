#!/usr/bin/env python
# coding: utf-8

# #### What is Regression?
# 
# Regression analysis is a form of preditive modelling technique which investigates the relationship between a dependant and independant variable.  
# 
# Uses:
# ->Determining the strength of predictors
# ->Forecasting an effect
# ->Trend Forecasting
# 
#     Concept                     : The data is modelled using a straight line
#     Used with                   : Continuous variable
#     Output/Prediction           : Value of the Variable
#     Accuracy/Goodness of fit    : measured by loss, R Squared, Adjusted R Squared. 

# #### What is Linear Regression? 
# A statistical model that attempts to show the linear relationship between the variables with linear equation.
# 

# #### Why Linear Regression? 
# 
# As the Linear Regression is used to predict the Value of the Continuous Variable(Charges), we use it here. 
# Such as - sales made on a day, Temperature of a city.
# And the linear Regression is not good for discrete variable(Output)

# #### DATA SET INFORMATION:::
# This is "Sample Insurance Claim Prediction Dataset" which based on "[Medical Cost Personal Datasets][1]" to update sample value on top.
# 
# age : age of policyholder
# 
# sex: gender of policy holder (female=0, male=1)
# 
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 25 steps: average walking steps per day of policyholder 
# 
# children: number of children / dependents of policyholder 
# 
# smoker: smoking state of policyholder (non-smoke=0;smoker=1) 
# 
# region: the residential area of policyholder in the US (northeast=0, northwest=1, southeast=2, southwest=3)
# 
# charges: individual medical costs billed by health insurance 
# 
# insuranceclaim: yes=1, no=0

# R Square Value: 
#     This Value tells how close the data are to the fitted regression line. 
#     Coefficient of determination or Coefficient of multiple determination
# 
# The Value should be close to 1(Regression line).
# If the value is 1 - it means all the data points lies on the regression line. 
# 
# Note: It is also possible to have a low R Squared value for a good Model & high R Squared value for a line that does not fit at all. 

# ### Lets Start Now... 
# Import all the libraries required.
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math as m
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr,spearmanr


# Import the DataSet from the path. 

# In[ ]:


df=pd.read_csv('../input/insurance.csv')


# In[ ]:


df.columns


# ### Exploratory Data Analysis(EDA)

# #### UniVariate Analysis

# 1.AGE

# In[ ]:


sns.distplot(df.age)
df.age.describe()


# 'Age' is a uniform distribution

# 2.SEX

# In[ ]:


df.sex.value_counts()


# In[ ]:


df.sex.value_counts()/len(df.sex)


# 3.BMI

# In[ ]:


sns.distplot(df.bmi)
df.bmi.describe()


# 'BMI' is a unifirm distribution

# 4.CHILDREN

# In[ ]:


df.children.value_counts()/len(df)


# 5.SMOKER

# In[ ]:


df.smoker.value_counts()/len(df)


# 6.REGION

# In[ ]:


df.region.value_counts()/len(df)


# 7.CHARGES

# In[ ]:


sns.distplot(df.charges)


# As we can see here, charges are divided into 1 part till 30000, and into another part after 30000

# In[ ]:


sns.distplot(df[df.smoker=='no'].charges,label='Non Smoker')
sns.distplot(df[(df.smoker=='yes')&(df.charges<30000)].charges,label='Poor Smoker')
sns.distplot(df[(df.smoker=='yes')&(df.charges>30000)].charges,label='Rich Smoker')

plt.legend()


# In the above plot, 
# 'Non Smoker' describes about the charges for who doesn't Smoke at all.
# 
# 'Poor Smoker' descirbes about who smokes and thier charges are average. 
# 
# 'Rich Smoker' descirbes about who smokes and their charges are High. 

# In[ ]:


sns.distplot(df[df.smoker=='no'].charges,label='Non Smoker')
plt.legend()


# In[ ]:


sns.distplot(np.log10(df[df.smoker=='no'].charges))


# Taking log will maximize the bins clearly. 

# #### MultiVariate Analysis

# In[ ]:


#H0: Less Smoking, more Children

df.groupby('smoker')['children'].value_counts()/df.groupby('smoker')['children'].count()


# No Smoking ppl has No children(0.431391)
# & Smoking ppl also has No children(0.419708)

# In[ ]:


#H0: Older people(Age above 60), Smoking habbit, Children

df[df.age>=60].groupby('smoker')['children'].value_counts()#/df[df.age>=60].groupby('smoker')['children'].count()


# In[ ]:


#H0: We loose BMI by Smoking

sns.barplot(df.smoker,df.bmi)


# This looks strange as both smoker and Non Smoker has the same BMI

# In[ ]:


#Who smokes more, male or Female

df.groupby('smoker')['sex'].value_counts()/df.groupby('smoker')['sex'].count()


# Male smokes more than Female. 

# In[ ]:


# How Age impacts the BMI
#H0: As the Age increases, BMI Increases,You gain weight.
# Age correlated to weight.

sns.scatterplot(df.age,df.bmi)


# In[ ]:


sns.scatterplot(df.bmi,df.charges)


# In[ ]:


df.groupby('region')['charges'].mean()


# Compartively in NorthEast region , the charges are high

# In[ ]:


df.groupby('region')['smoker'].value_counts()/df.groupby('region')['smoker'].count()


# OMG. In all the regions, the non smoker has been charged high than the smokers. so this shows that the Region is useless for predicting the charges

# In[ ]:


df.groupby('region')['sex'].value_counts()/df.groupby('region')['sex'].count()


# In all the 4 regions, male and females are almost of equal numbers

# In[ ]:


# No relation between Bmi and children

sns.barplot(df.children,df.bmi)


# In[ ]:


# Female will have less BMI, and with LESS BMI they pay less

sns.barplot(df.sex,df.bmi)
df.groupby('sex')['bmi'].mean()


# Male and Female has the same BMI more or less

# In[ ]:


# lets check the charges of Female who has BMI less than 30 and more than 30

print('Avg charge of a female who has BMI less than 30 is :'+str(df[(df.bmi<=30)&(df.sex=='female')]['charges'].mean()))
print('Avg charge of a female who has BMI less than 30 is :'+str(df[(df.bmi>30)&(df.sex=='female')]['charges'].mean()))


# In[ ]:


df.groupby('region')['bmi'].mean()


# In[ ]:


df.groupby(['smoker','children'])['charges'].mean()


# In[ ]:


df[(df.children>=4)&(df.smoker=='yes')]


# In[ ]:


sns.scatterplot(df.age,df.charges)


# ## Data preparation

# In[ ]:


X=df[['age','sex','bmi','smoker','children','region']].copy()
X.sex=X.sex.map(lambda x:1 if x=='male' else 0)
X.smoker=X.smoker.map(lambda x:1 if x=='yes' else 0)
X=X.drop(['region'],axis=1)
lr=LinearRegression()
y=df.charges
lr.fit(X,y)
y_hat=lr.predict(X)
residuals=y-y_hat


# In[ ]:


print('MSE:', mean_squared_error(y,y_hat))
print('RMSE:',mean_squared_error(y,y_hat)**0.5)
print('MAE:',mean_absolute_error(y,y_hat))
print('R2 Score:',r2_score(y,y_hat))


# In[ ]:


def adj_r2(y,y_hat,p):
    r2=r2_score(y,y_hat)
    n=len(y)
    return 1 - (1-r2)*(n-1)/(n-p-1)


# In[ ]:


adj_r2(y,y_hat,X.shape[1])


# In[ ]:


adj_r2(y,y_hat,X.shape[1]+4)


# #The Adjusted R2 value looks not bad.. we can see how can this be tuned further for better score. 

# ## Things We Missed. 

# In[ ]:


sns.pairplot(df,hue='smoker')


# In[ ]:


from scipy.stats import pearsonr
sns.lmplot(data=df[df.smoker=='yes'],x='bmi',y='charges')
print('pearsonr:',pearsonr(df[df.smoker=='yes'].bmi,df[df.smoker=='yes'].charges)[0])


# In[ ]:


sns.lmplot(data=df[df.smoker=="no"],x="bmi",y="charges")


# Will add a new column called 'Obese' and will see how the data looks like. 

# In[ ]:


df["obese"]=df.bmi.map(lambda x:"obese" if x>30 else "fit")


# In[ ]:


sns.pairplot(df,hue="obese")


# Will add another new column called 'fit_fat' and will label as fit smoker, fat smoker, fit and fat

# In[ ]:


def obese_smoker(obese,smoker):
    if obese=="obese" and smoker=="yes":
        return "fat_smoker"
    elif obese=="fit" and smoker=="yes":
        return "fit_smoker"
    elif obese=="obese" and smoker=='no':
        return "fat"
    else:
        return "fit"  


# In[ ]:


df["fit_fat"]=df[["obese","smoker"]].apply(lambda x: obese_smoker(x["obese"],x["smoker"]),axis=1)


# In[ ]:


df.head(5)


# In[ ]:


X=df[['age', 'bmi', 'children', "fit_fat"]].copy()
X=X.join(pd.get_dummies(X.fit_fat)).drop(["fit_fat"],axis=1)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)
y_hat=lr.predict(X)
residuals=y-y_hat

print("MSE:",mean_squared_error(y,y_hat))
print("RMSE:",mean_squared_error(y,y_hat)**0.5)
print("MAE:",mean_absolute_error(y,y_hat))
print("R squared:",r2_score(y,y_hat))


# In[ ]:


adj_r2(y,y_hat,X.shape[1])


# In[ ]:


for i,j in zip(X.columns,lr.coef_):
    print(i,"*",j,"+")
print(lr.intercept_)


# So this model looks better than the before

# COMMENTS/SUGGESTIONS are always welcome. 
# 
# Will discuss More of it.. 
# 
# Happy Learning

# In[ ]:




