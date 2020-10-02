#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv(r'/kaggle/input/insurance/insurance.csv')


# **Checking data types of the columns**

# In[ ]:


df.dtypes


# In[ ]:


df['age'].isna().value_counts()
df.columns


# In[ ]:


def checkNull (data) :
    columns = data.columns
    for column in columns :
        print('****Checking null values for '+ str(column) + ' ****')
        print(data[column].isna().value_counts())
        print('****Checking competed*****')
    


# Checking null values in the dataframe

# In[ ]:


checkNull(df)


# In[ ]:


corrMatrix = df.corr()


# In[ ]:


fig = plt.figure(figsize = (8,6))
ax= fig.add_subplot(111)
sns.heatmap(corrMatrix,annot=True,ax=ax)


# In[ ]:


df[['sex','charges']].plot(kind = 'scatter',x='sex',y='charges')


# Finding out diffrent regions

# In[ ]:


df['region'].value_counts()


# Checking the if different regions have any affect on the charges or not

# In[ ]:


df[df['region'] == 'southeast'][['region','charges']].plot(kind='hist',title = 'southeast')


# In[ ]:


df[df['region'] == 'southwest'][['region','charges']].plot(kind='hist',title = 'southwest')


# In[ ]:


df[df['region'] == 'northwest'][['region','charges']].plot(kind='hist',title = 'northwest')


# In[ ]:


df[df['region'] == 'northeast'][['region','charges']].plot(kind='hist',title = 'northeast')


# As above seen the distribution is similar in all the regions .So it doesnot have a much of affect on charges. 

# Now lets check wheather smoking has some affect on charges or not !

# In[ ]:


df_smoking = df[['smoker','charges']]
#df_smoker = df_smoking[df_smoking['smoker'] == 'yes']
df_smoking['smoker'].value_counts()


# There are total 274 smokers 

# In[ ]:


df_smoking.replace({'yes' : 1,'no': 0},inplace = True)


# In[ ]:


fig1 = plt.figure(figsize = (12,6))
fig1.subplots_adjust(wspace = 0.5)
ax1 = fig1.add_subplot(121)
sns.boxplot(x='smoker',y='charges',data=df_smoking,palette='rainbow',ax =ax1,showfliers=False)
ax1.set_xticklabels(['Non-Smoker','Somker'])
ax1.set_xlabel('')
ax1.set_ylabel('Insurance Charges')

ax1 = fig1.add_subplot(122) 
sns.violinplot(data =df_smoking, x='smoker', y='charges',ax= ax1)
ax1.set_xticklabels(['Non-Smoker','Somker'])
ax1.set_xlabel('')
ax1.set_ylabel('Insurance Charges')


# In above box plot and vilion plot we can see that the Average Insurance Amount paid to a non smoker is around $8000 to $9000, where as for smokers the Average Insurance Amount paid is far greater then the non smoker which is nearly $35000.
# So the Smoker feature is very useful for predicting the insurance charges.

# We can also see that the there is a posotive correlation between smoking and charges.

# In[ ]:


mat = df_smoking.corr()
sns.heatmap(mat,annot=True)


# Now lets see If Gender as any impact on insurance amount paid

# In[ ]:


df_gen = df[['sex','charges']]
fig2 = plt.figure(figsize = (15,6))
fig2.subplots_adjust(wspace = 0.5)
ax2 = fig2.add_subplot(121) 
sns.boxplot(data= df_gen,x = 'sex',y='charges',showfliers=False,ax=ax2)
ax2 = fig2.add_subplot(122)
sns.violinplot(data =df_gen, x='sex', y='charges',ax=ax2)


# Both males and females have same average Insurance charges ,but for the females the threshold for the 3rd quartile nearly 16000 dollars and for males it is nearly 20000 dollars.I think gender will not have that much of a impact on charges as the smoking habit,but we still consider it as a feature for the model

# In[ ]:


#df_gen.replace({'male': 1,'female':0},inplace = True)
df_gen.corr()


# Correlation is also not that much strong.

# Now let prepare the data for our model.By converting the categorical data to numerical.

# In[ ]:


df_model =                  df.replace({
                                        'southeast':0,
                                        'northwest':1,
                                        'southwest':2,
                                        'northeast':3,
                                        'male': 1,
                                        'female':0,
                                        'yes' : 1,
                                        'no': 0
                                         })
df_model.head()


# lets just have a quick look into the heatmap.Here SEX,CHILDREN,REGION have very weak correlation. 
# Where values nearer to +1 or -1 have strong correlation.

# In[ ]:


figg = plt.figure(figsize = (10,8))
ax3 = figg.add_subplot(111) 
sns.heatmap(df_model.corr(),annot=True,ax=ax3)


# Let's get started with the preparing the test and train data. We will be taking 80% of data for tarining and 20% for with testing. 

# In[ ]:


total_data_count = df_new_model.shape[0]
train_data_index = round(total_data_count*0.8) -1
train_data = df_new_model.loc[:train_data_index,:]
test_data_index = train_data_index +1
test_data = df_new_model.loc[test_data_index:,:]


# Now we splited our training and testing data.Lets prepare the model.

# In[ ]:


test_data.shape
train_data.shape


# In[ ]:


from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#reg = linear_model.LinearRegression() 
#x = df_new_model[['age','sex','bmi','children','smoker','region']]
#x = df_model[['smoker']]
#y = df_new_model[['charges']]
#reg.fit(x,y)
#print('The coefficients are ',reg.coef_)
#print('The intercept is ',reg.intercept_)


# Now that we trained our model.Now lets test it. 

# In[ ]:


y_hat = reg.predict(test_data[['age','sex','bmi','children','smoker','region']])
print ('R2 score is ',r2_score(test_data['charges'],y_hat))
print('RMSE score is ',math.sqrt(mean_squared_error(test_data['charges'],y_hat)))


# R2 score is average only but,RMSE is pretty high which is not right.
# I have to find why ?

# Previously I chnaged the smoker column data for 'Yes' to 1 and 'No' to 0, similarly also chaanged Gender column data for 'Male' to 1 and 'Female' to 0. This is popularly called as label encoding which has some diadvantages.

# Now lets use the One Hot Encoding to the change the catgeorical columns into numerical columns using pandas get_dummies method.

# In[ ]:


df_new_model = pd.get_dummies(df,columns=['smoker','sex','region'],prefix =['smoker','sex','region'])
df_new_model.head()
#df_new_model.columns


# lets quickly check for the correlation.

# In[ ]:


fig_new = plt.figure(figsize = (10,8))
ax_new = fig_new.add_subplot(111) 
sns.heatmap(df_new_model.corr(),annot=True,ax=ax_new)


# Now you see that we have converted the categorical columns to numerical columns using one hot encoding.Though our number of columns have increased.Lets build our model again and see if we can better score.

# In[ ]:


total_data_count = df_new_model.shape[0]
train_data_index = round(total_data_count*0.8) -1
train_data = df_new_model.loc[:train_data_index,:]
test_data_index = train_data_index +1
test_data = df_new_model.loc[test_data_index:,:]


# In[ ]:


from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
reg = linear_model.LinearRegression() 
x = df_new_model[['age', 'bmi', 'children','smoker_no', 'smoker_yes',
       'sex_female', 'sex_male', 'region_northeast', 'region_northwest',
       'region_southeast', 'region_southwest']]
y = df_new_model[['charges']]
reg.fit(x,y)
print('The coefficients are ',reg.coef_)
print('The intercept is ',reg.intercept_)


# In[ ]:


y_hat = reg.predict(test_data[['age', 'bmi', 'children','smoker_no', 'smoker_yes',
       'sex_female', 'sex_male', 'region_northeast', 'region_northwest',
       'region_southeast', 'region_southwest']])
print ('R2 score is ',r2_score(test_data['charges'],y_hat))
print('RMSE score is ',math.sqrt(mean_squared_error(test_data['charges'],y_hat)))


# Still no significant improvement huh !

# Lets give a try using ploynomial regression.

# We will be using multiple ploynomial regression model to check which degree polynomial gives best result.Here I will be using only few features

# In[ ]:


total_data_count = df_new_model[['smoker_yes']].shape[0]
train_data_index = round(total_data_count*0.8) -1
train_data = df_new_model.loc[:train_data_index,:]
test_data_index = train_data_index +1
test_data = df_new_model.loc[test_data_index:,:]


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def ployFuntion (degrees,train_data,test_data) :
    for degree in degrees :
        poly = PolynomialFeatures(degree)
        train_x_poly = poly.fit_transform(train_data[['age','bmi','smoker_no', 'smoker_yes']])
        test_x_poly = poly.fit_transform(test_data[['age','bmi','smoker_no', 'smoker_yes']])
        reg = linear_model.LinearRegression()
        reg.fit(train_x_poly,train_data['charges'])
        y_hat = reg.predict(test_x_poly)
        print('Degree of polynomial model used :',degree)
        print ('R2 score is ',r2_score(test_data['charges'],y_hat))
        print('RMSE score is ',math.sqrt(mean_squared_error(test_data['charges'],y_hat)))
        print ('MAE score is ',mean_absolute_error(test_data['charges'],y_hat))
        print ('*****************************')


# In[ ]:


ployFuntion(degrees = list(range(2,12)),train_data = train_data,test_data =test_data)


# Thank you all for giving time to my notebook.You guys can give suggestions in comments on how further I can increase the accuracy of the model.

# I got a suggestion that if I scale the features to smiliar range then it can do some improvement.
# Lets scale our variables using StandardScalar. What it does is 
# Let is consider feature X  (1,2,3,4,5...)
# After applying StandardScalar
# every element of x will be converted using 
# new_x(i th element) = (x (i th) - mean)/standard deviation
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
stan = StandardScaler()
std_charges = stan.fit_transform(df_new_model[['charges']])


# In[ ]:


from sklearn.preprocessing import StandardScaler
stan = StandardScaler()
std_bmi = stan.fit_transform(df_new_model[['bmi']])


# In[ ]:


from sklearn.preprocessing import StandardScaler
stan = StandardScaler()
std_age = stan.fit_transform(df_new_model[['age']])


# In[ ]:


df_new_model['age'] = std_age
df_new_model['bmi'] = std_bmi
df_new_model['charges'] = std_charges


# So using both polynomial and Multilinear regression we can see that 
# for Polynomial regression at dergee of freedom '9'
# R2 score is  0.7185110926492733
# RMSE score is  0.5437599132780697
# for Multilinear regression is 
# R2 score is  0.7603054934267751
# RMSE score is  0.5017710322215123
# 
# which is pretty good.
