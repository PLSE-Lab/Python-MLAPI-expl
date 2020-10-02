#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv',index_col='customerID')


# In[ ]:


df.head(15)


# In[ ]:


df.info()


# **Here we can see that Total Charges is an object variable.
# Let's Change it to float**

# In[ ]:


# We need to convert the Total Charges from object type to Numeric
df['TotalCharges'] = df['TotalCharges'].replace(r'\s+', np.nan, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])


# In[ ]:


df.info()


# # First Hypothesis Using Examples 

# **Let's Check for any Data Point **
# 
# **For Second Data Point**
# 
# **Our Calculation**  = 56.95 * 34(tenure) = 1936.3
# 
# **Given Total Charges** = 1889.50 
# 
# **Difference** = 46.8 
# 
# **Let's Take Another Example**
# 
# **For Fourth Data Point**
# 
# **Our Calculation**  = 42.30 * 45(tenure) = 1903.5
# 
# **Given Total Charges** = 1840.75
# 
# **Difference** = 62.75
# 
# **Here we can See we Have some difference this might be some kind of additional Tax and this must vary 
# from monthly plan to plan.**
# 
# **So According to Our Hypothesis Total-Charges should be Monthly Charges * Tenure + Additional Charges(Tax). 
# We will see Later if this Hypothesis is Correct or Not.**

# ** Now Let's Explore the Dataset**

# In[ ]:


df.Partner.value_counts(normalize=True).plot(kind='bar');


# In[ ]:


df.SeniorCitizen.value_counts(normalize=True).plot(kind='bar');


# In[ ]:


df.gender.value_counts(normalize=True).plot(kind='bar');


# In[ ]:


df.tenure.value_counts(normalize=True).plot(kind='bar',figsize=(16,7));


# In[ ]:


df.PhoneService.value_counts(normalize=True).plot(kind='bar');


# In[ ]:


df.MultipleLines.value_counts(normalize=True).plot(kind='bar');


# In[ ]:


df.InternetService.value_counts(normalize=True).plot(kind='bar');


# In[ ]:


df.Contract.value_counts(normalize=True).plot(kind='bar');


# In[ ]:


df.PaymentMethod.value_counts(normalize=True).plot(kind='bar');


# **We will Visualize other variables as we will perform our Analysis.**

# **Now Let's Plot variables with respect to Our Target Variable.**

# In[ ]:


# First let's see Our Target Variable
df.Churn.value_counts(normalize=True).plot(kind='bar');


# In[ ]:


# Now Let's Start Comparing.
# Gender Vs Churn
print(pd.crosstab(df.gender,df.Churn,margins=True))
pd.crosstab(df.gender,df.Churn,margins=True).plot(kind='bar',figsize=(7,5));


# In[ ]:


print('Percent of Females that Left the Company {0}'.format((939/1869)*100))
print('Percent of Males that Left the Company {0}'.format((930/1869)*100))     


# **We can See that Gender Does'nt Play an important Role in Predicting Our Target Variable.**

# In[ ]:


# Contract Vs Churn
print(pd.crosstab(df.Contract,df.Churn,margins=True))
pd.crosstab(df.Contract,df.Churn,margins=True).plot(kind='bar',figsize=(7,5));


# In[ ]:


print('Percent of Month-to-Month Contract People that Left the Company {0}'.format((1655/1869)*100))
print('Percent of One-Year Contract People that Left the Company {0}'.format((166/1869)*100)) 
print('Percent of Two-Year Contract People that Left the Company {0}'.format((48/1869)*100))     


# **Most of the People that Left were the Ones who had Month-to-Month  Contract.**

# In[ ]:


# Internet Service Vs Churn
print(pd.crosstab(df.InternetService,df.Churn,margins=True))
pd.crosstab(df.InternetService,df.Churn,margins=True).plot(kind='bar',figsize=(7,5));


# In[ ]:


print('Percent of DSL Internet-Service People that Left the Company {0}'.format((459/1869)*100))
print('Percent of Fiber Optic Internet-Service People that Left the Company {0}'.format((1297/1869)*100)) 
print('Percent of No Internet-Service People that Left the Company {0}'.format((113/1869)*100))     


# **Most of the people That Left had Fiber Optic Internet-Service.**

# In[ ]:


# Tenure Median Vs Churn
print(pd.crosstab(df.tenure.median(),df.Churn))
pd.crosstab(df.tenure.median(),df.Churn).plot(kind='bar',figsize=(7,5));


# In[ ]:


# Partner Vs Dependents
print(pd.crosstab(df.Partner,df.Dependents,margins=True))
pd.crosstab(df.Partner,df.Dependents,margins=True).plot(kind='bar',figsize=(5,5));


# In[ ]:


print('Percent of Partner that had Dependents {0}'.format((1749/2110)*100))
print('Percent of Non-Partner that had Dependents {0}'.format((361/2110)*100))     


# **We can See Partners had a much larger percent of Dependents than Non-Partner this tells us that Most Partners might be Married.**

# In[ ]:


# Partner Vs Churn
print(pd.crosstab(df.Partner,df.Churn,margins=True))
pd.crosstab(df.Partner,df.Churn,margins=True).plot(kind='bar',figsize=(5,5));


# In[ ]:


plt.figure(figsize=(17,8))
sns.countplot(x=df['tenure'],hue=df.Partner);


# **Most of the People that Were Partner will Stay Longer with The Company. So Being a Partner is a Plus-Point For the Company as they will Stay Longer with Them.**

# In[ ]:


# Partner Vs Churn
print(pd.crosstab(df.Partner,df.Churn,margins=True))
pd.crosstab(df.Partner,df.Churn,normalize=True).plot(kind='bar');


# In[ ]:


# Senior Citizen Vs Churn
print(pd.crosstab(df.SeniorCitizen,df.Churn,margins=True))
pd.crosstab(df.SeniorCitizen,df.Churn,normalize=True).plot(kind='bar');


# **Let's Check for Outliers in Monthly Charges And Total Charges Using Box Plots**

# In[ ]:


df.boxplot('MonthlyCharges');


# In[ ]:


df.boxplot('TotalCharges');


# **Both Monthly Charges and Total Charges don't have any Outliers so we don't have to Get into Extracting Information from Outliers.**

# In[ ]:


df.describe()


# # Correlation Matrix

# In[ ]:


# Let's Check the Correaltion Matrix in Seaborn
sns.heatmap(df.corr(),xticklabels=df.corr().columns.values,yticklabels=df.corr().columns.values,annot=True);


# **Here We can See Tenure and Total Charges are correlated and also Monthly charges and Total Charges are also correlated with each other. So this is proving our first Hypothesis right of Considering Total charges = Monthly charges * Tenure + Additional Tax that We had Taken Above.
# **

# # Data Munging Process

# In[ ]:


# Checking For NULL 
df.isnull().sum()


# **We can See here that We have 11 Null Values in Total Charges so  let's try to fill them..**

# In[ ]:


df.head(15)


# In[ ]:


fill = df.MonthlyCharges * df.tenure


# In[ ]:


df.TotalCharges.fillna(fill,inplace=True)


# In[ ]:


df.isnull().sum()


# **No Null Values are there Now..**

# # When Churn = 'Yes'

# In[ ]:


df.loc[(df.Churn == 'Yes'),'MonthlyCharges'].median()


# In[ ]:


df.loc[(df.Churn == 'Yes'),'TotalCharges'].median()


# In[ ]:


df.loc[(df.Churn == 'Yes'),'tenure'].median()


# In[ ]:


df.loc[(df.Churn == 'Yes'),'PaymentMethod'].value_counts(normalize = True)


# **Most of the People that Left are the Ones who had Payment Method as Electronic Check so Let's Make a Seperate Variable for it so that The Model can Easily Predict our Target Variable.**

# In[ ]:


df['Is_Electronic_check'] = np.where(df['PaymentMethod'] == 'Electronic check',1,0)


# In[ ]:


df.loc[(df.Churn == 'Yes'),'PaperlessBilling'].value_counts(normalize = True)


# In[ ]:


df.loc[(df.Churn == 'Yes'),'DeviceProtection'].value_counts(normalize = True)


# In[ ]:


df.loc[(df.Churn == 'Yes'),'OnlineBackup'].value_counts(normalize = True)


# In[ ]:


df.loc[(df.Churn == 'Yes'),'TechSupport'].value_counts(normalize = True)


# In[ ]:


df.loc[(df.Churn == 'Yes'),'OnlineSecurity'].value_counts(normalize = True)


# **We can See that People That Left the Company did'nt use Services Like Online Security , Device Protection , Tech Support and Online Backup quite often. Hence for Our Prediction these variables will not be of much Importance. We will Drop them in the End.**

# In[ ]:


df= pd.get_dummies(df,columns=['Partner','Dependents',
       'PhoneService', 'MultipleLines','StreamingTV',
       'StreamingMovies','Contract','PaperlessBilling','InternetService'],drop_first=True)


# **We have Encoded the Categorical Variables with Numeric using get dummies Property which will make it easy for the Machine to Make Correct Prediction.**

# In[ ]:


df.info()


# **Now Let's Drop the variables that are not Important For us according to our Analysis.**

# In[ ]:


df.drop(['StreamingTV_No internet service','StreamingMovies_No internet service'],axis=1,inplace=True)


# In[ ]:


df.drop('gender',axis=1,inplace=True)


# In[ ]:


df.drop(['tenure','MonthlyCharges'],axis=1,inplace=True)


# In[ ]:


df.drop(['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','PaymentMethod'],axis=1,inplace=True)


# **Let's Convert Our Target Variable 'Churn' for Yes or No to 1 or 0. **

# In[ ]:


df = pd.get_dummies(df,columns=['Churn'],drop_first=True)


# In[ ]:


df.info()


# **Now We have only 16 variables that we think are important for Our Prediction. So let's Start our Modelling Part.**

# # Modelling Part

# In[ ]:


X = df.drop('Churn_Yes',axis=1).as_matrix().astype('float')
y = df['Churn_Yes'].ravel()


# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# **Let's start with Logistic Regression Model because we know Our Target Variable has a Binary Outcome.**

# In[ ]:


# Import Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[ ]:


# create model
model_lr_1 = LogisticRegression(random_state=0)


# In[ ]:


# train model
model_lr_1.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,classification_report


# In[ ]:


# performance metrics
# accuracy
print ('accuracy for logistic regression - version 1 : {0:.2f}'.format(accuracy_score(y_test, model_lr_1.predict(X_test))))
# confusion matrix
print ('confusion matrix for logistic regression - version 1: \n {0}'.format(confusion_matrix(y_test, model_lr_1.predict(X_test))))
# precision 
print ('precision for logistic regression - version 1 : {0:.2f}'.format(precision_score(y_test, model_lr_1.predict(X_test))))
# precision 
print ('recall for logistic regression - version 1 : {0:.2f}'.format(recall_score(y_test, model_lr_1.predict(X_test))))


# **We have got a Great Accuracy of Approx  80% with Our Logistic Regression Model. Let's Try another Model which is XG Boost Classifier.**

# In[ ]:


print(classification_report(y_test,model_lr_1.predict(X_test)))


# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# In[ ]:


# performance metrics
# accuracy
print ('accuracy for xgboost- version 1 : {0:.2f}'.format(accuracy_score(y_test, classifier.predict(X_test))))
# confusion matrix
print ('confusion matrix for xgboost - version 1: \n {0}'.format(confusion_matrix(y_test, classifier.predict(X_test))))
# precision 
print ('precision for xgboost - version 1 : {0:.2f}'.format(precision_score(y_test, classifier.predict(X_test))))
# precision 
print ('recall for xgboost - version 1 : {0:.2f}'.format(recall_score(y_test, classifier.predict(X_test))))


# In[ ]:


print(classification_report(y_test,classifier.predict(X_test)))


# **Here we can see that XG Boost gives us the best results. We can also perform Parameter tuning using Cross Validation for Improving our Model Accuracy but I will keep that For you all.**

# **Thanks For Reading the Whole Kernel if you have any suggestions or changes that I can Make in this Kindly tell me in the Comment Section. I'll be glad to Here them.**

# In[ ]:




