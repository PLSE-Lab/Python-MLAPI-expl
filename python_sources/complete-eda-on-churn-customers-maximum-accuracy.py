#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing useful libararies
import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# Read the Data

# In[ ]:


df = pd.read_csv(r'../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# Have a look the data, how its looks like.

# In[ ]:


df.head()


# In[ ]:


df.isna().sum()
df.columns
df.isnull().sum()
df.shape


# ### This dataset contains no 'null' or 'na' values.
# ### Shape is (7043, 21).
# ### Have 21 different columns.
# ### Have 3 Contineous varibales and 18 categorical variables.
# ### Our Target varible is Churn, also a categorical variable.

# * Dropping unnecessary column.

# In[ ]:


df.drop(['customerID'],axis=1,inplace=True)


# In[ ]:


df.nunique()


# * Easily see the number of categories each varible has.

# In[ ]:


for i in df.columns:
    print(df[i].value_counts())


# ### **Observation:**
# * there are some " "(Blank Space) in Total  Charges Column. 
# * So, we need to replace that values with null values.

# In[ ]:


df.isin([" "]).sum()


# In[ ]:


df['TotalCharges'] = df['TotalCharges'].replace([" "], np.nan)


# * Replace the null values with Nan values.**

# In[ ]:


df.isna().sum().sum()


# you can the count of nan values.

# In[ ]:


df.dropna(inplace = True)


# ** Dropped NA Values.**[](http://)

# ### Setting the correct data Types of colums, then its easy to manipulate.

# In[ ]:


df['TotalCharges']= df['TotalCharges'].astype(float)


# In[ ]:


df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)


# In[ ]:


df['SeniorCitizen'] = df['SeniorCitizen'].replace({1:'Yes', 0:'No'})


# ### Seprating the categorical and Numerical variables.

# In[ ]:


cat = [i for i in df.columns if df[i].dtypes == 'O']
num = [i for i in df.columns if df[i].dtypes != 'O']


# Replacing 'No Phone Service' or 'No Internet Service' with 'No'.
# *Because it does not make any sense out of it. Still we will do some EDA on this.

# In[ ]:


df.replace(['No phone service'], ['No'], inplace = True)


# In[ ]:


df.replace({'No internet service':'No'}, inplace = True)


# In[ ]:


df[cat].nunique().plot(kind='barh')


# ### **Observation:**
# - The paymentMethod, Contract and Internet Service has 3 or more categories.
# - Most of the variables has only 2 categories.

# In[ ]:


fig, ax =plt.subplots(1,2, figsize=(15,5))
plt.figure(figsize=(5,5))
sea.countplot(x ='StreamingTV', hue = 'Churn' ,data =df, ax= ax[0])
sea.countplot(x ='PaymentMethod', hue = 'Churn' ,data =df, ax= ax[1])
fig.show()


# ### **Observation:**
# - Those customer who have subscribe **Streaming TV** service are less likey to churn as compared to those have subscribed it.
# - Those customer who are using **Electronic check** payment method are more likey to churn.

# In[ ]:


fig, ax =plt.subplots(1,2, figsize=(15,5))
plt.figure(figsize=(5,5))
sea.countplot(x ='PaperlessBilling', hue = 'Churn' ,data =df, ax= ax[0])
sea.countplot(x ='Contract', hue = 'Churn' ,data =df ,ax = ax[1])
fig.show()


# ### **Observation:**
# - Those customers who use paperless billing are more likely to churn.
# - Those cusotmers who are using Month- to Month Contract are more churning. 

# In[ ]:


fig, ax =plt.subplots(1,2, figsize=(15,5))
plt.figure(figsize=(5,5))
sea.countplot(x ='InternetService', hue = 'Churn' ,data =df, ax= ax[0])
sea.countplot(x ='gender', hue = 'Churn' ,data =df, ax = ax[1])
fig.show()


# ### **Observation:**
# - Fiber Optic Internet service users are more churning.

# In[ ]:


plt.figure(figsize=(10,5))
ax = sea.countplot(x="Churn", hue="Contract", data=df);
ax.set_title('Contract Type vs Churn')


# ## Some Analysis on Contineous varibles.

# In[ ]:


df[num].head()


# have a look at numerical varible

# In[ ]:


plt.figure(figsize=(8,5))
plt.title("Monthly C,harges VS Total Charges")
plt.scatter(x = df.MonthlyCharges, y = df.TotalCharges)
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
plt.show()


# ### **Observation:**
# - The relationship Between the MontyCharges and TotalCharges is postive. 

# In[ ]:


plt.figure(figsize=(5,5))
df[['MonthlyCharges','tenure']].head(35).plot(kind='line')
plt.title("Monthly Charges VS Tenure")
plt.xlabel('Monthly Charges')
plt.ylabel('Tenure')
plt.show()


# ### **Observation:**
# - 

# In[ ]:


sea.countplot(x= 'Churn' ,data=df, hue='SeniorCitizen')


# ### **Observation:**
# - Senior Citizen cutomers are less and they are very less likely to churn.

# In[ ]:


df.Contract.value_counts().plot(kind='pie', legend= True)


# ### **Observation:**
# - The Month-to-Month Contract customers are more in numbers.

# In[ ]:


sea.distplot(df["tenure"], color="b")


# ### **Observation:**
# - The tenure of maximum customer are belong to 0-20.

# In[ ]:


sea.distplot(df["MonthlyCharges"], color="r")


# ### **Observation:**
# - The variation between the MontlyCharges.

# In[ ]:


sea.distplot(df["TotalCharges"], color="g")


# ### **Observation:**
# - The maxmimum total charges is vary between 0 to 2000.

# In[ ]:


df['Count_OnlineServices'] = (df[['OnlineSecurity', 'DeviceProtection', 'StreamingMovies', 'TechSupport','StreamingTV', 'OnlineBackup']] == 'Yes').sum(axis=1)
plt.figure(figsize=(10,5))
sea.countplot(x= 'Count_OnlineServices', hue= 'Churn', data =df)


# ### **Observation:**
# - Those cutomers, who is subscribing various  services are less likely to churn.

# In[ ]:


ax = sea.boxplot(x='Churn', y = 'tenure', data=df)
ax.set_title('Churn vs Tenure', fontsize=20)


# ### **Observation:**
# - The less the tenure more likely to churn.

# In[ ]:


sea.violinplot(x="MultipleLines", y="tenure", hue="Churn", kind="violin",
                 split=True, palette="pastel", data=df)


# ### **Observation:**
# - Those who are not using Multiple lines Service tend to churn in first of their months.

# In[ ]:


cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df1 = pd.melt(df[df["InternetService"] != "No"][cols]).rename({'value': 'Has service'}, axis=1)
plt.figure(figsize=(10, 5))
ax = sea.countplot(data=df1, x='variable', hue='Has service')
ax.set(xlabel='Additional service', ylabel='Num of customers')
plt.show()


# ### **Observation:**
# - The Most used service is Streaming Movies and Streaming TV.
# - the OnlineSecurity and Techsupport serices are less acquired by customers.

# In[ ]:


plt.figure(figsize = (12,5))
sea.countplot(x= 'Churn' ,data=df, hue='PaymentMethod')


# In[ ]:


plt.figure(figsize = (12,5))
sea.boxplot(x="Contract", y="MonthlyCharges", hue="Churn", data=df)


# ## Data Preprocessing for Machine Learning Equations

# In[ ]:


y = df.Churn


# * seprate the target column

# In[ ]:


df.drop('Churn', axis =1, inplace= True)


# * drop that target column from the dataframe

# In[ ]:


y = pd.DataFrame(y)
y['Churn'].replace(to_replace='Yes', value=1, inplace=True)
y['Churn'].replace(to_replace='No',  value=0, inplace=True)


# * replace their values from Yes to 1 and No to 0. 

# In[ ]:


df_temp =df


# In[ ]:


df = pd.get_dummies(df)


# * Creating Dummy variables for out all categorical columns and drop all other columns

# In[ ]:


df.shape


# Shape of our prepared data frame.

# In[ ]:


scaler = MinMaxScaler()
df[num] = scaler.fit_transform(df[num])


# * Doing Min MAx scalling for numerical varibles. As they are high in values so, we need to take them at same page. 

# In[ ]:


df.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0)


# * Dividing the data into Training and Testing.

# ## LOgistic Regression Model

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
model_lr = LogisticRegression()
result = model_lr.fit(X_train, y_train)
prediction_test = model_lr.predict(X_test)
metrics.accuracy_score(y_test, prediction_test)


# In[ ]:


disp = plot_roc_curve(model_lr, X_test, y_test)


# ## Random Forest Classifier with 1000 Decision Tress.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=1000)
model_rf.fit(X_train, y_train)
prediction_test = model_rf.predict(X_test)
metrics.accuracy_score(y_test, prediction_test)


# In[ ]:


disp = plot_roc_curve(model_rf, X_test, y_test)


# In[ ]:


importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=df.columns.values)
weights.sort_values()[-20:].plot(kind = 'barh')


# **You can see the Important features for model prediction according to Random Forest algorithm. **

# ## Support vector Machine Algorithm.

# In[ ]:


from sklearn.svm import SVC
model_svm = SVC(kernel='linear') 
model_svm.fit(X_train,y_train)
preds = model_svm.predict(X_test)
metrics.accuracy_score(y_test, preds)


# In[ ]:


disp = plot_roc_curve(model_svm, X_test, y_test)


# ## Extream Boost Classifier with Learning rate is 0.25

# In[ ]:


import xgboost as xgb
model_gb=xgb.XGBClassifier(learning_rate=0.25,max_depth=4)
model_gb.fit(X_train, y_train)
model_gb.score(X_test,y_test)


# In[ ]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],   
}


# ## RandomizedSearch for tuning and finding the best parameters for Extream boosting algorithm.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
random_search=RandomizedSearchCV(model_gb,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_train, y_train)


# In[ ]:


random_search.best_params_


# **We found out the best paramerter impleting again in Extream boosting. **

# In[ ]:


import xgboost as xgb
model_gb=xgb.XGBClassifier(learning_rate=0.05,max_depth=3)
model_gb.fit(X_train, y_train)
model_gb.score(X_test,y_test)


# In[ ]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier


# ## Building ensemble learning model with Stacking Classifier
# * For stage one, I choose Logistic Regression, KNearestNeigbour, DecisionTreeCassifier, and NaiveBayes.
# * For stage two, default model is Logistic Regression.

# In[ ]:


level0 = list()
level0.append(('lr', LogisticRegression()))
level0.append(('knn', KNeighborsClassifier()))
level0.append(('cart', DecisionTreeClassifier()))
level0.append(('svm', SVC()))
level0.append(('bayes', GaussianNB()))
# define meta learner model
level1 = LogisticRegression()
# define the stacking ensemble
modelx = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
modelx.fit(X_test, y_test)


# In[ ]:


preds = modelx.predict(X_test)
metrics.accuracy_score(y_test, preds)


# # **If you like this Notebook, Do Upvote this**
# # Thanks
