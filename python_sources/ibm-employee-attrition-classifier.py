#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


# In[4]:


#printing the first 5 rows
df=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()


# In[5]:


#checking for null values in the dataset
df.isnull().sum()
#No null values in the dataset


# In[6]:


#datatypes of all the fields in the dataset
df.dtypes


# In[7]:


correlation_df = df.corr()
# The below correlation coefficients is NaN for Employee Count and Standard Hours Fields
#This may be because of the zero variance in those fields
employee_count_var = df["EmployeeCount"].var() #this is 0
standard_hours_var = df["StandardHours"].var() #this is 0


# In[8]:


#Hence we drop these 2 rows
new_df = df.drop(["EmployeeCount","StandardHours"],axis = 1)


# In[9]:


new_df.head()


# In[10]:


correlation_new_df = new_df.corr()
correlation_new_df


# In[11]:


#The above given matrix can also be drawn on MatplotLib for better visual Interpretation

sns.set()
f, ax = plt.subplots(figsize=(10, 8))
corr = new_df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()


# Now from the above plot, we can see that YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion and YearsWtihCurrManager
# are closely related, Hence we drop those columns.
# Also, we'll drop the following rows based on their High Correlation coefficient from the plot:
# 1. TotalWorkingYears
# 2. PercentSalaryHike
# 3. PerformanceRating
# 4. NumCompaniesWorked
# 5. MonthlyIncome
# 
# 
# 

# In[12]:


#After removing the strongly correlated variables
df_numerical = new_df[['Age','DailyRate','DistanceFromHome','Education',
                       'EnvironmentSatisfaction', 'HourlyRate',                     
                       'JobInvolvement', 'JobLevel','MonthlyRate',
                       'JobSatisfaction',
                       'RelationshipSatisfaction', 
                       'StockOptionLevel',
                        'TrainingTimesLastYear','WorkLifeBalance']].copy()
df_numerical.head()


# Now we can scale our numerical features either by StandardScaler or by using math operations
# 

# In[13]:


df_numerical = abs(df_numerical - df_numerical.mean())/df_numerical.std()  
df_numerical.head()


# As we have got our numerical data set up for modelling, We need to do the same for the categorical data.
# We create dummies for the categorical data and remove any redundant rows.

# In[14]:


df_categorical = new_df[['Attrition', 'BusinessTravel','Department',
                       'EducationField','Gender','JobRole',
                       'MaritalStatus',
                       'Over18', 'OverTime']].copy()
df_categorical.head()


# In[15]:


df_categorical["Over18"].value_counts()
#Since all values are Y, we can drop this column


# In[16]:


df_categorical = df_categorical.drop(["Over18"],axis = 1)


# In[17]:


# We now Label Encode the Attrition data 
lbl = LabelEncoder()
lbl.fit(['Yes','No'])
df_categorical["Attrition"] = lbl.transform(df_categorical["Attrition"])
df_categorical.head()


# In[18]:


# We create dummies for the remaining categorical variables

df_categorical = pd.get_dummies(df_categorical)
df_categorical.head()


# In[19]:


#Now we finally join both the numerical and categorical dataframes for model evaluation

final_df = pd.concat([df_numerical,df_categorical], axis= 1)
final_df.head()


# Now we finally have to split our data into training and test set. First, we allot the input and the output variables
# 

# In[20]:


X = final_df.drop(['Attrition'],axis= 1)
y = final_df["Attrition"]


# In[21]:




X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 4)


# Now we have a wide variety of classification algorithms to choose from, and we'll select the one with the highest model accuracy. We start with Logistic Regression
# 
# 

# In[22]:


lr = LogisticRegression(solver = 'liblinear',random_state = 0) #Since this a small dataset, we use liblinear solver and Regularization strength as
# default i.e C = 1.0
lr.fit(X_train,y_train)


# In[23]:


y_pred_lr = lr.predict(X_test)


# In[24]:


accuracy_score_lr = accuracy_score(y_pred_lr,y_test)
accuracy_score_lr 
#Logistic Regression shows 85.7 percent accuracy


# 
# Now we use DecisionTreeClassifier for classifying our input data

# In[25]:


dtree = DecisionTreeClassifier(criterion='entropy',max_depth = 4,random_state = 0)


# In[26]:


dtree.fit(X_train,y_train)


# In[27]:


y_pred_dtree = dtree.predict(X_test)


# In[28]:


accuracy_score_dtree = accuracy_score(y_pred_dtree,y_test)
accuracy_score_dtree


# Now we use RandomForestClassifier

# In[29]:


rf = RandomForestClassifier(criterion = 'gini',random_state = 0)
rf.fit(X_train,y_train)


# In[30]:


y_pred_rf = rf.predict(X_test)
accuracy_score_rf = accuracy_score(y_pred_rf,y_test)
accuracy_score_rf


#  Support vector Machines

# In[31]:


sv = svm.SVC(kernel= 'linear',gamma =2)
sv.fit(X_train,y_train)


# In[32]:


y_pred_svm = sv.predict(X_test)
accuracy_score_svm = accuracy_score(y_pred_svm,y_test)
accuracy_score_svm


# KNN Algorithm

# In[33]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)


# In[34]:


y_pred_knn = knn.predict(X_test)
accuracy_score_knn = accuracy_score(y_pred_knn,y_test)
accuracy_score_knn


# In[35]:


scores = [accuracy_score_lr,accuracy_score_dtree,accuracy_score_rf,accuracy_score_svm,accuracy_score_knn]
scores = [i*100 for i in scores]
algorithm  = ['Logistic Regression','Decision Tree','Random Forest','SVM', 'K-Means']
index = np.arange(len(algorithm))
plt.bar(index, scores)
plt.xlabel('Algorithm', fontsize=10)
plt.ylabel('Accuracy Score', fontsize=5)
plt.xticks(index, algorithm, fontsize=10, rotation=30)
plt.title('Accuracy scores for each classification algorithm')
plt.ylim(80,100)
plt.show()    


# Based on the above graph, Random Forest and SVM shows the highest accuracy score.
# 
# We now calculate feature importances for RandomForest Classifier

# In[36]:


feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')
plt.show()


# Findings : Monthly Rate affects the Attrition the most

# In[ ]:





# In[ ]:





# In[ ]:




