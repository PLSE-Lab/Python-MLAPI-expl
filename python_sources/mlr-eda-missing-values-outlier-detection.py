#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Hello Everyone!I am a Beginner in this Field ! I am used Python Language to Predict the Dependent Variable(Salary) to see which varialbes have most relationship will dependent variable(Multiple linear Regression)
# Use various Machine learning too and Libraries!
# - Numpy ,Pandas , matplotlib ,seaborn!
# - Missing Values
# - Detecting Outliers
# - Multiple Linear Regression (Backward Elimination)
# - Assumption on Linear Regression
# - Prediction on Test dataset 

# # Import our Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sn
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import  variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error


# # Importing our Dataset

# In[ ]:


data = pd.read_csv(r"/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")


# In[ ]:


data.head() # Reture 1st (5) Rows


# In[ ]:


data.tail()  # Return Last (5) Rows


# # Data Types and Shape of the Dataset

# In[ ]:


data.shape


# In[ ]:


data.info() # To showcast which variable contain which type of Datatype


# In[ ]:


data.columns


# In[ ]:


data.describe()


# # Checking the Missing Values

# In[ ]:


missing_value_presentage =round(data.isnull().sum()*100/len(data),2).reset_index()
missing_value_presentage.columns = ['column_name','missing_value_presentage']
missing_value_presentage = missing_value_presentage.sort_values('missing_value_presentage',ascending =False)
missing_value_presentage


# # As we all can see only the Salary varaible contain the Missing values : 31.16 % with is not less then the 50%. So we can replace the missing values by (Mean)

# In[ ]:


data['salary'].fillna(data['salary'].mean(),inplace =True)


# # Once again Run the Same Code

# In[ ]:


missing_value_presentage =round(data.isnull().sum()*100/len(data),2).reset_index()
missing_value_presentage.columns = ['column_name','missing_value_presentage']
missing_value_presentage = missing_value_presentage.sort_values('missing_value_presentage',ascending =False)
missing_value_presentage


# # Find which varaibles contain whic type of data type?

# In[ ]:


data_num =data[data.select_dtypes(include=[np.number]).columns.tolist()]


# In[ ]:


data_num.describe()


# In[ ]:


# take out our String columns
data_str = data[data.select_dtypes(exclude=[np.number]).columns.tolist()]


# In[ ]:


data_str.describe(include ='all')


# # For the Object variables U First have to make all the variables into the Numeric format using the  ( LabelEncoder)

# In[ ]:


from sklearn.preprocessing import LabelEncoder
# Label Encoder Perform on all the Varaible which contains the (Object Datatype)


# In[ ]:


data_str.gender =(LabelEncoder().fit_transform(data_str.gender))
data_str.ssc_b =(LabelEncoder().fit_transform(data_str.ssc_b))
data_str.hsc_b =(LabelEncoder().fit_transform(data_str.hsc_b))
data_str.hsc_s =(LabelEncoder().fit_transform(data_str.hsc_s))
data_str.degree_t =(LabelEncoder().fit_transform(data_str.degree_t))
data_str.workex =(LabelEncoder().fit_transform(data_str.workex))
data_str.specialisation=(LabelEncoder().fit_transform(data_str.specialisation))
data_str.status =(LabelEncoder().fit_transform(data_str.status))


# In[ ]:


dataset = pd.concat([data_str,data_num],axis = 1)


# In[ ]:


dataset.head()


# # Detection of Outlier (Boxplot &Barplot)

# In[ ]:


#  Outliers Detection
cont_vars =list(dataset.columns)[:]
def outlier_visual(dataset):
    plt.figure(figsize=(16,35))
    i = 0
    for col in cont_vars:
        i+= 1
        plt.subplot(15,2,i)
        plt.boxplot(dataset[col])
        plt.title('{} boxplot'.format(col))
        i += 1
        plt.subplot(15,2,i)
        plt.hist(dataset[col])
        plt.title('{} histogram'.format(col))
    plt.show()
outlier_visual(dataset)


# # AS u can see Only Two columns contain the Outliers (hsc_p & Salary)

# We are going to remove the outliers using the IQR

# In[ ]:


############################# hsc_p #########################
q1 = dataset['hsc_p'].quantile(0.25)
q3 = dataset['hsc_p'].quantile(0.75)
iqr = q3-q1 #Interquartile range
low  = q1-1.5*iqr #acceptable range
high = q3+1.5*iqr #acceptable range
low,high


# In[ ]:


dataset['hsc_p']=np.where(dataset['hsc_p'] > high,high,dataset['hsc_p']) # upper limit
dataset['hsc_p']=np.where(dataset['hsc_p'] < low,low,dataset['hsc_p']) # low limit


# In[ ]:


############################# Salary ########################################
q1 = dataset['salary'].quantile(0.25)
q3 = dataset['salary'].quantile(0.75)
iqr = q3-q1 #Interquartile range
low  = q1-1.5*iqr #acceptable range
high = q3+1.5*iqr #acceptable range
low,high


# In[ ]:


dataset['salary']=np.where(dataset['salary'] > high,high,dataset['salary']) # upper limit


# In[ ]:


outlier_visual(dataset)


# # Data Visualization

# # Who is getting more placements girls or boys?

# In[ ]:


def plot(dataset,x,y):
    plt.Figure(figsize =(10,10))
    sns.boxplot(x = data[x],y= data[y])
    g = sns.FacetGrid(data, row = y)
    g = g.map(plt.hist,x)
    plt.show()


# In[ ]:


plot(dataset,"salary","gender")


# - The Range of salary is high for boys with the median of 2.5 Lakhs per annum
# - The Median salary for girls is 2.1 Lakhs per annum
# - The highest package is offered to a boy which is nearly 10 Lakhs per annum
# - The highest package offered for girls is 7 Lakhs per annum
# - Total number girls not placed are 30 and Total number of boys not placed are 40

# In[ ]:


sns.countplot(dataset['status'],hue=dataset['gender'])


#  As u can see that  Boys have got the more number of placement !! Ratio is also more great 100:50

# # To get placed in a company with high package which board should I choose (Central or State board) in 10th?

# In[ ]:


plot(dataset,"salary","ssc_b")


# - The Range of salary is high for central board students with the median of 2.5 Lakhs per annum
# - The Median salary for other board students is 2.3 Lakhs per annum
# - The highest package is offered to a central board student which is nearly 10 Lakhs per annum and as per our previous finding the student is a boy
# - The highest package offered for other board students is 5 Lakhs per annum
# - Total number central board students not placed are 27 and Total number of other board student not placed are 37

# In[ ]:


sns.countplot(dataset['status'],hue=dataset['ssc_b'])


# # To get placed in a company with high package which board should I choose (Central or State board) in 12th?

# In[ ]:


plot(dataset,"salary","hsc_b")


# # Who is mostly not getting placed?

# In[ ]:


sns.catplot(x="status", y="ssc_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="hsc_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="degree_p", data=data,kind="swarm",hue='gender')


# The students who have scored less than 60 percent in 10th or 12th or degree are mostly not getting placed because they don't even have basic eligibility(more than 60 percent in 10th,12th and degree)

# # Which stream students are getting more placed and which stream students are mostly not placed?

# In[ ]:


sns.violinplot(x="degree_t", y="salary", data=data)
sns.stripplot(x="degree_t", y="salary", data=data,hue='status')


# The stream in which the students mostly get placed are Commucation and management , also science and technology students are mostly getting placed and other stream students are not getting that much placements due to less number of students...

# # Data Partition

# In[ ]:


X = dataset.drop('salary', axis = 1)
Y = dataset[['salary']]
# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)


# In[ ]:


# Correlation Matrix 
dataset_exp =pd.concat([y_train,X_train],axis = 1)


# In[ ]:


# calculate the correlation matrix
corr = dataset_exp.corr()
display(corr)
# plot the correlation heatmap
sns.heatmap(corr,annot =True)


# # Model Building

# In[ ]:


import statsmodels.api as sm
X_1 = sm.add_constant(X_train)
# create a OLS model
model = sm.OLS(y_train, X_1).fit()
print(model.summary())


#  # Multicolinearity (with the help of VIF)

# In[ ]:


X1=dataset.drop(['salary'],axis=1)
# the VFI does expect a constant term in the data, 
#so we need to add one using the add_constant method
#X1 = sm.add_constant(econ_df_before)
series_before = pd.Series([variance_inflation_factor(X1.values, i) 
                           for i in range(X1.shape[1])], index=X1.columns)
series_before


# #  Backward Elimination Method[****](http://)

# In[ ]:


X = dataset.drop('salary', axis = 1)
y = dataset['salary']


# In[ ]:


#Backward Elimination
cols = list(X.columns) # all column present  in x  
pmax = 1
while (len(cols)>0):  # count of variable should be greater than zero
    p= []
    X_1 = X[cols]  # all column we are assign in x_1
    X_1 = sm.add_constant(X_1)# adding a column with value 1
    final_model = sm.OLS(y,X_1).fit() # Regression model
    p = pd.Series(final_model.pvalues.values[1:],index = cols) 
    # to get p-values for all variable only     
    pmax = max(p)  # select a max P-value 
    feature_with_p_max = p.idxmax()   
    # idmax is used to display the variable name which has max P-value
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# In[ ]:


final_model.summary()


# # Vif of Model

# In[ ]:


X1=dataset.loc[:,['gender', 'degree_t', 'status', 'etest_p', 'mba_p']]
series_before = pd.Series([variance_inflation_factor(X1.values, i) 
                           for i in range(X1.shape[1])], index=X1.columns)
series_before


# # Assumption of Model
# 
# * Linearity
# * Normality
# * Homoscedasicity
# * Model Error has to be independently identificaly Distibuted

# # Normality

# In[ ]:


import pylab
# check for the normality of the residuals
sm.qqplot(final_model.resid, line='s')
pylab.show()


#  # Homoscedasicity

# In[ ]:


Data=pd.concat([X_train,y_train],axis=1)


# In[ ]:


Data['Fitted_value']=final_model.fittedvalues
Data['Residual']=final_model.resid


# In[ ]:


p = Data.plot.scatter(x='Fitted_value',y='Residual')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')
plt.show()


# # Model Error are IID

# In[ ]:


Data['Residual'].plot.hist()


# # Prediction on Test Data (unseen data)

# In[ ]:


X_test2 = X_test[['gender', 'degree_t', 'status', 'etest_p', 'mba_p']]


# In[ ]:


X_test2 = sm.add_constant(X_test2)


# In[ ]:


X_test2.head()


# In[ ]:


y_predict=final_model.predict(X_test2)


# In[ ]:


test=pd.concat([X_test,y_test],axis=1)


# In[ ]:


test['Predicted']=y_predict


# In[ ]:


test.head()


# # Performance on Test Data set

# In[ ]:


import math
# calculate the mean squared error
model_mse = mean_squared_error(test['salary'], test['Predicted'])
# calculate the mean absolute error
model_mae = mean_absolute_error(test['salary'], test['Predicted'])
# calulcate the root mean squared error
model_rmse = math.sqrt(model_mse)
# display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


#  # Performance on Training Data set

# In[ ]:


import math
# calculate the mean squared error
model_mse = mean_squared_error(Data['salary'], Data['Fitted_value'])
# calculate the mean absolute error
model_mae = mean_absolute_error(Data['salary'], Data['Fitted_value'])
# calulcate the root mean squared error
model_rmse = math.sqrt(model_mse)
# display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[ ]:




