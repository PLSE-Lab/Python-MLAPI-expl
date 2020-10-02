#!/usr/bin/env python
# coding: utf-8

# # TASK #1: UNDERSTAND THE PROBLEM STATEMENT AND BUSINESS CASE

# 
# <table>
#   <tr><td>
#     <img src="https://drive.google.com/uc?id=1yJKgmHrRFnBk987HJfeDrMcTEXtk0z7W"
#          alt="Fashion MNIST sprite"  width="1000">
#   </td></tr>
#   <tr><td align="center">
#     <b>Figure 1. Employee Retention Prediction
#   </td></tr>
# </table>
# 

# ![alt text](https://drive.google.com/uc?id=10NJUOTWOBzp2MNkgcPpCF0fLtdoN_jKj)

# ![alt text](https://drive.google.com/uc?id=1evbDHoW2t0emxkbQd8yevYFZ5woJKRPY)

# ![alt text](https://drive.google.com/uc?id=1Mk2H7VYfv6ijUS9XqEdBQV6_LaHiyvkJ)

# # TASK #2: IMPORT LIBRARIES AND DATASETS

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# You have to include the full link to the csv file containing your dataset
dataset = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
dataset.sample(5)


# In[ ]:


dataset.info()


# ### We also now know which features are catogerical.

# In[ ]:


dataset.describe()


# ### .describe() gives us a lot of information. For example now I know the average age group of employess in this company which is around 37.

# # TASK #3: VISUALIZE DATASET

# In[ ]:


# Let's replace 'Attrition' , 'overtime' , 'Over18' column with integers before performing any visualizations 
dataset['Attrition'] = dataset['Attrition'].apply(lambda x:1 if x == 'Yes' else 0)
dataset['OverTime'] = dataset['OverTime'].apply(lambda x:1 if x == 'Yes' else 0)
dataset['Over18'] = dataset['Over18'].apply(lambda x:1 if x == 'Y' else 0)
dataset.head()


# In[ ]:


# Let's see if we have any missing data.
sns.heatmap(dataset.isnull(),cmap = 'Blues', cbar = False, yticklabels = False)


# ### Luckily we do not have any missing values. :)

# ### Now lets plot a histogram of all the features together and analyse some important features

# In[ ]:


dataset.hist(bins=30,figsize=(20,20),color='g')


# ### We get really good analyses for some important features such as Age, percentagehike, totalworkingyears, Monthlyincome, attrition. All this by a simple .hist command

# In[ ]:


# Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are tail heavly
# It makes sense to drop 'EmployeeCount' and 'Standardhours' since they do not change from one employee to the other


# In[ ]:


# It makes sense to drop 'EmployeeCount' , 'Standardhours' and 'Over18' since they do not change from one employee to the other
# Let's drop 'EmployeeNumber' as well
# use inplace = True to change the values in memory.

dataset.drop(['EmployeeCount','StandardHours','Over18','EmployeeNumber'],axis = 1, inplace = True)


# In[ ]:


# Let's see how many employees left the company! 
left_df = dataset[dataset['Attrition'] == 1]
stayed_df = dataset[dataset['Attrition'] == 0]


# In[ ]:


print('1. Total = {} '.format(len(dataset)))
print('2. Number of employees left the company = {}'.format(len(left_df)))
print('3. Percentage of employees left the company = {}'.format((len(left_df)/len(dataset))*100))
print('4. Number of employees who stayed in the company = {}'.format(len(stayed_df)))
print('5. Percentage of employees stayed the company = {}'.format((len(stayed_df)/len(dataset))*100))


# #Count the number of employees who stayed and left
# #It seems that we are dealing with an imbalanced dataset 
# 

# In[ ]:


left_df.describe()

#  Let's compare the mean and std of the employees who stayed and left 
# 'age': mean age of the employees who stayed is higher compared to who left
# 'DailyRate': Rate of employees who stayed is higher
# 'DistanceFromHome': Employees who stayed live closer to home 
# 'EnvironmentSatisfaction' & 'JobSatisfaction': Employees who stayed are generally more satisifed with their jobs
# 'StockOptionLevel': Employees who stayed tend to have higher stock option level


# In[ ]:


stayed_df.describe()


# In[ ]:


correlations = dataset.corr()
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(correlations, annot=True)


# #### Job level is strongly correlated with total working years
# #### Monthly income is strongly correlated with Job level
# #### Monthly income is strongly correlated with total working hours
# #### Age is stongly correlated with monthly income
# #### Also we can see that overtime has a strong affect on Attrition
# 

# ### Now lets see what age group tent to leave the company more.

# In[ ]:


plt.figure(figsize=(25,12))
sns.countplot(x = 'Age', hue = 'Attrition', data = dataset)


# #### We can see that the age group between 28 to 31 have left the most

# ### Lets explore more

# In[ ]:


plt.figure(figsize=(20,20))
plt.subplot(511)
sns.countplot(x = 'JobRole',hue = 'Attrition', data=dataset)
plt.subplot(512)
sns.countplot(x = 'MaritalStatus',hue = 'Attrition', data=dataset)
plt.subplot(513)
sns.countplot(x = 'JobInvolvement',hue = 'Attrition', data=dataset)
plt.subplot(514)
sns.countplot(x = 'JobLevel',hue = 'Attrition', data=dataset)
plt.subplot(515)
sns.countplot(x = 'OverTime',hue = 'Attrition', data=dataset)


# #### Sales Representitives tend to leave compared to any other job 
# #### Single employees tend to leave compared to married and divorced
# #### Less involved employees tend to leave the company 
# #### Less experienced (low job level) tend to leave the company 

# ### Lets do some more visualizations, but now for continuous values

# In[ ]:


plt.figure(figsize = (12,8))
sns.kdeplot(left_df['DistanceFromHome'], label = 'Employees who left', color = 'r', shade = True)
sns.kdeplot(stayed_df['DistanceFromHome'],label='Employees who stayed',color = 'b',shade=True)
plt.xlabel('Distance from home')


# ### As we can see, as the distance increases the employees tend to leave more as compared to who stayed.

# In[ ]:


plt.figure(figsize=(12,8))
sns.kdeplot(left_df['YearsWithCurrManager'],shade=True,color='r',label='Employes who left')
sns.kdeplot(stayed_df['YearsWithCurrManager'],shade=True,color='b',label='Employes who stayed')

plt.xlabel('Number of years with the current manager')
plt.title('Number of years with the current manager v/s Atrition')


# #### As seen that in early time with the manager, emloyees tend to leave more than staying but with time passing by the employees tend to stay.

# In[ ]:


plt.figure(figsize=(12,8))
sns.kdeplot(left_df['TotalWorkingYears'],label='Employees who left',shade = True, color = 'r')
sns.kdeplot(stayed_df['TotalWorkingYears'],label='Employees who stayed',shade = True, color = 'b')

plt.xlabel('Number of total working years')
plt.title('Number of total working years v/s Attrition')


# #### Interesting to see the trend that a lot of employees leave between 7 to 10 years of working

# ### Lets create some boxplot for more visualizations

# In[ ]:


# Let's see the Gender vs. Monthly Income
sns.boxplot(x='MonthlyIncome',y='Gender',data=dataset)


# ### Nice to see the gender equality here. Infact female tend to get more salaries here.

# In[ ]:


# Let's see the Jod role vs. Monthly Income
plt.figure(figsize=(10,8))
sns.boxplot(x='MonthlyIncome',y='JobRole',data=dataset)


# #### No doubt why we saw sales representatives leaving the job a lot in my earlier visualizations.

# # TASK #4: CREATE TESTING AND TRAINING DATASET & PERFORM DATA CLEANING

# #### Handling the catogerical variables.
# First we select them, then we transform them.

# In[ ]:


cat_var = [key for key in dict(dataset.dtypes)
             if dict(dataset.dtypes)[key] in ['object'] ] 
cat_var


# In[ ]:


X_cat = dataset[['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus']]
X_cat.head()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
X_cat


# In[ ]:


X_cat = pd.DataFrame(X_cat)
X_cat.head()


# #### Selecting all the numerical values

# In[ ]:


numeric_var = [key for key in dict(dataset.dtypes)
                   if dict(dataset.dtypes)[key]
                       in ['float64','float32','int32','int64']]
numeric_var


# In[ ]:


X_numerical = dataset[['Age','Attrition','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]


# In[ ]:


X_all = pd.concat([X_cat,X_numerical],axis=1)
X_all.head()


# In[ ]:


# I will now drop the target variable 'Attrition'
X_all.drop('Attrition',axis=1,inplace=True)
X_all.shape


# ### Now I will be scaling down all the values so that we can feed it to our ML/DL models

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X_all)
scaled_data


# In[ ]:


y = dataset['Attrition']
y.shape


# # TASK #5: UNDERSTAND THE INTUITION BEHIND LOGISTIC REGRESSION CLASSIFIERS, ARTIFICIAL NEURAL NETWORKS, AND RANDOM FOREST CLASSIFIER

# ![alt text](https://drive.google.com/uc?id=19DpnhFkfsNEDPlH1dkfdr1zO36vRcBit)

# ![alt text](https://drive.google.com/uc?id=1J03xZf6OiYtGV3IgJBUURBWyScpvaAbU)

# ![alt text](https://drive.google.com/uc?id=1WNsznVn7je5r9HGnSLLdABICxrIv2Mrs)

# ![alt text](https://drive.google.com/uc?id=1bX5uGmy5vbYTlp7m4tw_V2kTNzAHpHLp)

# ![alt text](https://drive.google.com/uc?id=1ztrMNehNYWMw6NwhOOC9BDBdnoNirpqZ)

# # TASK #6: UNDERSTAND HOW TO ASSESS CLASSIFICATION MODELS 

# ![alt text](https://drive.google.com/uc?id=1OZLbKm1AJSyvoBgfvlfcLIWZxLOvzOWq)

# ![alt text](https://drive.google.com/uc?id=11pNdVw4oWeNOWrkadrrxon7FU4qO5m6U)

# ![alt text](https://drive.google.com/uc?id=1Bk1xFW2tGBdwg-njOhw79MxtYBQnK-6x)

# ![alt text](https://drive.google.com/uc?id=19cXoBqSiqbEGNofnD603bz3xEAsX28hy)

# # TASK #7: TRAIN AND EVALUATE A LOGISTIC REGRESSION CLASSIFIER

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(scaled_data,y,test_size = 0.25, random_state=43)


# In[ ]:


model_LR = LogisticRegression()
model_LR.fit(X_train,y_train)
LR_pred = model_LR.predict(X_test)
print('The accuracy score for Logistic Regression is: {}'.format(100*accuracy_score(LR_pred,y_test)))


# In[ ]:


cm = confusion_matrix(LR_pred,y_test)
sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(LR_pred,y_test))


# ### As seen I got a good recall score with Logistic but my precision score for those who will leave is not good.
# 

# # TASK #8: TRAIN AND EVALUATE A RANDOM FOREST CLASSIFIER

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier()


# In[ ]:


model_RF.fit(X_train,y_train)
RF_pred = model_RF.predict(X_test)
print('The accuracy score for Random Forest is: {}'.format(100*accuracy_score(RF_pred,y_test)))


# In[ ]:


# Testing Set Performance
cm = confusion_matrix(RF_pred,y_test)
sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(RF_pred,y_test))


# ### With Random Forest my precission score for employees who will leave is not so good.

# # TASK #9: TRAIN AND EVALUATE A DEEP LEARNING MODEL 

# In[ ]:


import tensorflow as tf


# In[ ]:


model_NN = tf.keras.models.Sequential()
model_NN.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(50, )))
model_NN.add(tf.keras.layers.Dense(units=500, activation='relu'))
model_NN.add(tf.keras.layers.Dense(units=500, activation='relu'))
model_NN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[ ]:


model_NN.summary()


# In[ ]:


model_NN.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


scaled_df = pd.DataFrame(scaled_data,columns=X_all.columns)
scaled_df.head()


# In[ ]:


X_train_new,X_test_new,y_train_new,y_test_new = train_test_split(scaled_df,y,test_size = 0.25, random_state=43)


# In[ ]:


epochs_hist = model_NN.fit(X_train_new, y_train_new, epochs = 30, batch_size = 50)


# In[ ]:


y_pred = model_NN.predict(X_test)
y_pred = (y_pred > 0.5)


# In[ ]:


plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])


# In[ ]:


plt.plot(epochs_hist.history['accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(['Training Accuracy'])


# In[ ]:


# Testing Set Performance
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)


# In[ ]:


print(classification_report(y_test, y_pred))


# ## Conclusion
# 
# #### In this notebook I did some EDA and visualized the data with the help of plt and sns. I used Logistic Regression, Random Forest & later built an ANN for predictions. Each of them had good accuracy. 
# ### But This is a mistake I see a lot of people doing.
# #### Hence I want to address here. A good accuracy score is not enough to evaluate the performance of your model. It can actually mislead you sometimes. A real world model should have a good precision & recall score too. Hence its always a good practice to draw a confusion matrix and a classification report to get a better understanding of your model's performance. 
# ### So how can my models perform better?
# #### Well there are a lot of things that can be done to make the models performe better. One more technique can be to handle the imbalance data. The target 'Attrition' was really imbalanced. The data can be balanced by using suppose SMOTE. 
# #### One can use RandomisedSearchCV to find the best params of ML models. The ANN model can also be hyper tuned. There are a number of ways. 
# ## I would appreciate anyone to copy my notebook can make the predictions better and let me know the results. 

# In[ ]:


#from imblearn.over_sampling import SMOTE
#oversampler = SMOTE(random_state=0)
#smote_train, smote_target = oversampler.fit_sample(X_train_new, y_train_new)
#epochs_hist = model_NN.fit(smote_train, smote_target, epochs = 10, batch_size = 50)


# In[ ]:




