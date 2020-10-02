#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# In[ ]:


import os
print(os.listdir("../input/bank-loan-modelling"))


# Importing the data set and renaming the columns.

# In[ ]:


df = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx', 'Data')
df.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]


# 

# In[ ]:


df.head(4) #to check first 4 rows of data set.


# In[ ]:


df.info()


# To see all the summary of data let us make one DataFrame that shows feature,dtype,Null value count,Unique count, Unique item.

# In[ ]:


listitem=[]
for col in df.columns:
    listitem.append([col,df[col].dtypes,df[col].isna().sum(),round((df[col].isna().sum()/len(df[col]))*100,2),df[col].nunique(),df[col].unique()])
dfdesc=pd.DataFrame(columns=['features','dtype','Null value count','Null value percentage','Unique count','Unique items'],data=listitem)
dfdesc


# In[ ]:


df.shape #to check no of rows and column


# In[ ]:


pd.set_option("display.float","{:.2f}".format)
df.describe()


# # **Exploratory Data Analysis.**

# Exploratory Data Analysis is a approach to analyzing the data set to summarize their main characteristics  often with visual method . Here Visualizing means we will plot different charts and graph to get valuable information about  the data set. Every Machine project starts with Eda and Exploratory data analysis is the most important part of Machine learning project.

# In[ ]:


df.PersonalLoan.value_counts()


# In[ ]:


plt.figure(figsize=(5,5))
df.PersonalLoan.value_counts().plot(kind="bar",color=['salmon','lightblue'])


# Personal Loan feature which is target variable has imblance data set which have more count of persoanl loan 0 than personal loan 1.  i.e. 9:1.
# We Use Over sampling in Feature Engineering to make it balance data set.

# In[ ]:


categorical_val=[]
continuous_val=[]
for column in df.columns:
    print('=================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continuous_val.append(column)


# In[ ]:


categorical_val


# In[ ]:


continuous_val


# In[ ]:


plt.figure(figsize=(17,17))
for i , column in enumerate(categorical_val,1):
    plt.subplot(3,3,i)
    df[df["PersonalLoan"]==0][column].hist(bins=35,color='red',label='Have Personal Loan = No')
    df[df["PersonalLoan"]==1][column].hist(bins=35,color='Blue',label="Have Personal Loan = Yes")
    plt.legend()
    plt.xlabel(column)


# Form the above histogram chart we can see that.
# 1. Family size of 3 and 4 members are tending to take Personal Loan.
# 2. Customer that belong to Education category 2 and 3 i.e. Graduate and Professional have taken more Persoanl Loan then the Undergraduate class.
# 3. Customer who does not have Security Account have taken Personal Loan .
# 4. Customer who does not CDAcount in this higher number of customer don't have Personal Loan . We can see that customer who have CDAcount most of them had taken Personal Loan. Here CDAccount means Certificate of Deposit.
# 5. Customer how use Internet Bank service also have higher count of Personal Loan then those who does not use Online Service.
# 6. Customer who don't have excess to Credit Card for Universal Bank are more likely to apply for PersonaL Loan.
# 
# 

# In[ ]:


df[continuous_val].plot(kind='box',subplots=True, layout=(3,3), fontsize=10, figsize=(14,14))


# Income, CCAvg , Mortgage have Outlier we will deal with this in Feature Engineering.

# In[ ]:


sns.pairplot(data=df)


# 

# From the pair plot we can see that.
# 1. Age and Experience both have high correlation which each other. 
# 2. Income,CCAvg,Mortage show positive skewness.

# In[ ]:


plt.figure(figsize=(7,7))
plt.scatter(x='Age',y='Experience',data=df)


# As we can see Age and Experience both have high correlation between each other. We have to remove any one of them in feature Engineering.

# In[ ]:


df["Age"].value_counts().plot.bar(figsize=(20,6))


# 1. 1. # **Feature Enginerring**

# Feature engineering is the process of using domain knowledge to extract features from raw data set. Having a good knowledge of feature engineering  will help us get more accurate representation of underlying structure of the data and help us to improve the performance of machine learning algorithm. In Feature Engineering we have several topics like how to deal with missing values, how to deal with outliers , how do we convert categorical variables to numerical values so that model can read values easily.

# In[ ]:


df.describe()['Experience']


# The data contain some negative Experience data point as we see min is -3.
# Let us see the count of negative data points.

# In[ ]:


df[df['Experience']<0].count()


# In[ ]:


# Let us replace all the negative Experience data points by absolute value.
df['Experience']=df['Experience'].apply(abs)


# In[ ]:


df[df['Experience']<0].count()


# Now we don't have any negative data points in Experience.

# Now we had outlier in our data set.To treat them we will be replacing all those data points whole value less than equal to LL=(Q1-1.5*IQR) and greater than equal to UL=(Q3+1.5*IQR) by LL and UL.This is known as Capping Method

# In[ ]:


Outlier = ['Income', 'CCAvg', 'Mortgage']
Q1=df[Outlier].quantile(0.25)
Q3=df[Outlier].quantile(0.75)
IQR=Q3-Q1
LL,UL = Q1-(IQR*1.5),Q3+(IQR*1.5)

for i in Outlier:
    df[i][df[i]>UL[i]]=UL[i];df[i][df[i]<LL[i]]=LL[i] 


# In[ ]:


df[continuous_val].plot(kind='box',subplots=True, layout=(3,3), fontsize=10, figsize=(14,14))


# Now we can see that now we do not have any outlier in data set.

# In[ ]:


corr=df.corr()
fig,ax=plt.subplots(figsize=(12,12))
ax=sns.heatmap(corr,annot=True,square=True,fmt=".2f",cmap="YlGnBu")


# From look at above Heatmap
# We can consider to remove Experience,ID & ZIPCODE.

# In[ ]:


categorical_val.remove('PersonalLoan')
print(categorical_val)


# As we can see Family and Education are Ordinal Variables so we do label endcoing.

# In[ ]:


dataset = df.copy() # Let us create new dataset


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
label1=encode.fit_transform(df['Family'])
label2=encode.fit_transform(df['Education'])
dataset['Family']=label1
dataset['Education']=label2


# In[ ]:


dataset


# In[ ]:


dataset.drop(['ID','Experience','ZIPCode'],axis=1,inplace=True)


# In[ ]:


from sklearn.preprocessing import StandardScaler
ssc=StandardScaler()
col_to_scale=['Age','Income','CCAvg','Mortgage']
dataset[col_to_scale] = ssc.fit_transform(dataset[col_to_scale])


#  We have to make dataset balance. We will be using SMOTHE Method 

# In[ ]:


X = dataset.drop('PersonalLoan', axis=1)
y = dataset.PersonalLoan


# In[ ]:


PersonalLoan1=dataset[dataset['PersonalLoan']==1]
PersonalLoan0=dataset[dataset['PersonalLoan']==0]
print(PersonalLoan1.shape,PersonalLoan0.shape)


# In[ ]:


## RandomOverSampler to handle imbalanced data
from imblearn.over_sampling import RandomOverSampler
os =  RandomOverSampler(random_state=101)
X_ros,y_ros=os.fit_sample(X,y)


# In[ ]:


from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_ros)))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size= 0.3, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_ros, y_ros)
y_pred=log_reg.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[ ]:


from sklearn.metrics import (accuracy_score , f1_score , precision_score , recall_score)
print("Accuracy:" , accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test , y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1:", f1_score(y_test,y_pred))


# Let us try to use Catboost Classifier Algorithm

# In[ ]:


data = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx', 'Data')
data.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]


# In[ ]:


data.head()


# In[ ]:


categorical_val=[]
continuous_val=[]
for column in data.columns:
    print('=================')
    print(f"{column} : {data[column].unique()}")
    if len(data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continuous_val.append(column)


# In[ ]:


data[continuous_val].plot(kind='box',subplots=True, layout=(3,3), fontsize=10, figsize=(14,14))


# In[ ]:


# Capping Method

Outlier = ['Income', 'CCAvg', 'Mortgage']
Q1=data[Outlier].quantile(0.25)
Q3=data[Outlier].quantile(0.75)
IQR=Q3-Q1
LL,UL = Q1-(IQR*1.5),Q3+(IQR*1.5)

for i in Outlier:
    data[i][data[i]>UL[i]]=UL[i];data[i][data[i]<LL[i]]=LL[i] 


# In[ ]:


data[continuous_val].plot(kind='box',subplots=True, layout=(3,3), fontsize=10, figsize=(14,14))


# In[ ]:


# Standardizing the variables
from sklearn.preprocessing import StandardScaler
ssc=StandardScaler()
col_to_scale=['Age','Income','CCAvg','Mortgage']
data[col_to_scale] = ssc.fit_transform(data[col_to_scale])


# In[ ]:


data.drop(['ID','Experience','ZIPCode'],axis=1,inplace=True)
data


# In[ ]:


cat=['Family','Education','SecuritiesAccount','CDAccount','Online','CreditCard']


# In[ ]:


target_col='PersonalLoan'
X= df.loc[:,df.columns!=target_col]
y=df.loc[:,target_col]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


features=list(X_train.columns)


# In[ ]:


# Importing Library
get_ipython().system('pip install catboost')
from catboost import CatBoostClassifier


# In[ ]:


model_cb=CatBoostClassifier(iterations=1000, learning_rate=0.01, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=42) # use_best_model params will make the model prevent overfitting


# In[ ]:


model_cb.fit(X_train,y_train,cat_features=cat,eval_set=(X_test,y_test))


# In[ ]:


y_pred1 =model_cb.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test , y_pred1))


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test , y_pred1))


# In[ ]:


print("Accuracy:",accuracy_score(y_test , y_pred1))
print("Precision:",precision_score(y_test,y_pred1))
print("Recall:",recall_score(y_test,y_pred1))
print("F1-score:",f1_score(y_test,y_pred1))


# Conclusion
# As we can see by Using CatBoost Classifier we have gained 99% Accuracy. I think Catboost Classifier give us maximum accuracy.
# 

# I will be doing the Model Deployment Soon using Flask and Heroku.
# If you think there needs some impovement let me know in comment section.
