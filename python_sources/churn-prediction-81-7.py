#!/usr/bin/env python
# coding: utf-8

# ### INTRODUCTION
# 
# #### Predicting the customer churn mainly through logistic regression
# #### churn class was classified into two categories  No(0) and Yes(1)
# #### Steps taken in preprocessing includes Data cleaning, Standardization etc
# #### Other models where used to compare accuracy
# 
# ### SIDE NOTE
# #### You can leave your question about any unclear part in the comment section
# #### Any correction will be highly welcomed

# ### LOADING THE DATAFRAME

# In[ ]:


#Import the neccesary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sns.set()

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# In[ ]:


path = '/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(path)


# In[ ]:


df.head()


# In[ ]:


df.describe(include = 'all')


# #### DATA CLEANING

# In[ ]:


df.info()


# #### This data is clean but on further analysis TotalCharges includes some empty value which we will replace with 0

# In[ ]:


#Value count of the column
df['TotalCharges'].value_counts()


# In[ ]:


#Replacing the empty value with zero 
df['TotalCharges'].replace(' ', 0, inplace = True)


# In[ ]:


df[df['TotalCharges'].apply (lambda x: x== ' ')]


# In[ ]:


#Changing the column datatype to float
df['TotalCharges'] = df['TotalCharges'].astype('float')


# ### DATASET ANALYSIS AND OUTLIERS REMOVAL
# 
# #### we will plot the distribution of all the numeric variables in other to be able to identify outliers and any other abnormalities
# #### Outliers will be dealt with by removing either top 1% or the bottom 1%
# 

# In[ ]:


sns.distplot(df['tenure']) #This distribution plot appears to be normal with no outlier


# In[ ]:


sns.distplot(df['MonthlyCharges']) #This distribution plot appears to be normal with no outlier


# In[ ]:


sns.distplot(df['TotalCharges'])#This distribution plot appears to be having a few outliers. Let's explore it further


# In[ ]:


df.describe()


# In[ ]:


#Selecting Totalcharges above 8500 to see if they are outliers
df[df['TotalCharges'].apply (lambda x: x > 8500)]
#Upon further exploration they are not outliers


# ### CHECKING OLS ASSUMPTIONS
# 
# #### Let's check that our dataset are not violating any of this assumptions which includes:
# #### 1. No Endogeneity
# #### 2. Normality and Homoscedasticity
# #### 3.No Autocorrelation
# #### 4.NO multicollinearity: making sure our independents variables are not strongly related(correlated) with each other
# 
# ####  We are not violating  assumptions 1 through 3 but for NO multicollinearity we need to check

# In[ ]:


#Getting Variables in our dataframe
df.columns.values


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# the target column (in this case 'churn') should not be included in variables
#Categorical variables already turned into dummy indicator may or maynot be added if any
variable = df[['tenure', 'MonthlyCharges','TotalCharges',]]
X = add_constant(variable)
vif = pd.DataFrame()
vif['VIF']  = [variance_inflation_factor(X.values, i) for i in range (X.shape[1])]
vif['features'] = X.columns

vif
#Using 10 as the minimum vif values i.e any independent variable 10 and above will have to be dropped
#From the results all independent variable are below 10


# ### Standardization
# 
# #### Standardizing helps to give our independent varibles a more standard and relatable numeric scale, it also helps in improving model accuracy
# #### We are going to standardize only our numerical variables then use new columns to hold the resulting values

# In[ ]:


#Selecting the variable
scale_int = df[['MonthlyCharges']]

scaler = StandardScaler()#Selecting the standardscaler
scaler.fit(scale_int)#fitting our independent variables


# In[ ]:


df['scaled_monthly']= scaler.transform(scale_int)#scaling


# In[ ]:


scale_int = df[['tenure']] #Selecting the variable

scaler = StandardScaler()#Selecting the standardscaler
scaler.fit(scale_int)#fitting our independent variables


# In[ ]:


df['scaled_tenure']= scaler.transform(scale_int)#scaling


# In[ ]:


scale_int = df[['tenure']] #Selecting the variable

scaler = StandardScaler()#Selecting the standardscaler
scaler.fit(scale_int)#fitting our independent variables


# In[ ]:


df['scaled_charges']= scaler.transform(scale_int)#scaling


# In[ ]:


df.describe()# Checking our scaled results


# In[ ]:


df.describe(include = 'all')


# In[ ]:


#Dropping columns not needed
df.drop(['tenure','MonthlyCharges','customerID', 'TotalCharges'], axis = 1, inplace = True)


# ### Dummy Variables
# #### churn is a categorical variable so we need  to turn it into a dummy indicator before we can perform our regression
# #### For other categorical variable we will use get_dummies

# In[ ]:


#Turning Churn to a dummy indicator with 1 standing yes and 0 standing for no
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})


# In[ ]:


#Variables in our dataframe
df.columns.values


# In[ ]:


#new dataframe with dummies
df_dummies = pd.get_dummies(df, drop_first = True)

df_dummies


# ### LOGISTIC REGRESSION

# In[ ]:


#Declaring independent variable i.e x
#Declaring Target variable i.e y
x = df_dummies.drop('Churn', axis = 1)
y = df_dummies['Churn']


# In[ ]:


#Splitting our data into train and test dataframe
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 24)


# In[ ]:


reg = LogisticRegression() #Selecting the model
reg.fit(x_train, y_train) #training the model with x_train and y_train


# In[ ]:


#Predicting with our already trained model using x_test
y_hat = reg.predict(x_test)


# In[ ]:


#Getting the accuracy of our model
acc = metrics.accuracy_score(y_hat, y_test)
acc


# In[ ]:


#The intercept for our regression
reg.intercept_


# In[ ]:


#Coefficient for all our variables
reg.coef_


# ### CONFUSION MATRIX

# In[ ]:


cm = confusion_matrix(y_hat,y_test)
cm


# In[ ]:


# Format for easier understanding
cm_df = pd.DataFrame(cm)
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df


# #### Our model predicted '0' correctly 954 times while  predicting '0' incorrectly 154 times
# #### Also it predicted  '1'  correctly 198 times while predicting '1' incorrectly  103

# ### OTHER MODELS

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier # for K nearest neighbours
from sklearn import svm #for Support Vector Machine (SVM) 


# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y1 = dt.predict(x_test)
acc1 = metrics.accuracy_score(y1, y_test)
acc1


# In[ ]:


kk = KNeighborsClassifier()
kk.fit(x_train,y_train)
y2 = kk.predict(x_test)
acc2 = metrics.accuracy_score(y2, y_test)
acc2


# In[ ]:


sv = svm.SVC()
sv.fit(x_train,y_train)
y3 = sv.predict(x_test)
acc3 = metrics.accuracy_score(y3, y_test)
acc3


# #### After comparison with some other model  logistic regression gave us the best accuracy with ~81.8% followed closely by svm model with ~81.7%

# ###  CONCLUSION
# #### Let's try to make a table and with weight(BIAS) and odds 

# In[ ]:


result = pd.DataFrame(data = x.columns.values, columns = ['features'] )
result['weight'] = np.transpose(reg.coef_)
result['odds'] = np.exp(np.transpose(reg.coef_))

result


# In[ ]:




