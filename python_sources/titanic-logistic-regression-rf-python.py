#!/usr/bin/env python
# coding: utf-8

# **Libraries**

# In[93]:


# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[94]:


df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')
df_full = df_train.append(df_test)
pred_passid = df_test.PassengerId


# In[95]:


print (df_train.shape)
print (df_test.shape)
print (df_full.shape)


# In[96]:


df_full.info()


# In[97]:


df_full.isna().sum()


# ## Missing Values

# As you can see, we should have 1309 records. Columns with the missing values are **Age**, **Embarked**, **Fare** and **Cabin**.

# In[98]:


# Age Distribution
print ("The mean of age is: %.1f" % df_full['Age'].mean())
print ("The median of age: %.1f" % df_full['Age'].median())


# In[99]:


# Age
sns.distplot(df_full['Age'], bins=15);


# The **Age** distribution tells us that it skewes slightly to the right. Median will be used for the missing Values.

# In[100]:


# Since Age skewes to the right, we will use median for the NA values.
df_full['Age']=df_full['Age'].fillna(df_full['Age'].median())
df_full['Age'].describe()


# Similar to **Age**, let's swap all NAs in **Fare** with median.

# In[101]:


df_full['Fare']=df_full['Fare'].fillna(df_full['Fare'].median())


# Next, we need to impute for **Embarked** field. Since it's a categorical variable, I'm going to replace with it's most often used value. In this case, S appears the most often.

# In[102]:


df_full.Embarked.value_counts()


# In[103]:


df_full.loc[df_full['Embarked'].isna(),'Embarked']='S'


# Lastly, **Cabin** will be dropped due to a vast number of missing values

# In[104]:


df_full.drop('Cabin', axis =1, inplace = True)
df_full.head()


# In[105]:


df_full.isna().sum()


# # Dataset Exploration

# In[106]:


# Imbalanced Classes
df_full['Survived'].value_counts(normalize=True)


# In[107]:


#Sex
sns.countplot(x='Sex',hue='Survived',data=df_full);


# In[108]:


# Fare vs. Survived
plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_full["Fare"][df_full.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(df_full["Fare"][df_full.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
ax.set(xlabel='Fare')
plt.xlim(-10,85)
plt.show()


# In[109]:


# Pclass vs Survived
sns.countplot(x='Pclass',hue='Survived',data=df_full)
plt.show()


# # Feature Extraction
# 
# 1. Title & Name length
# 2. Family size

# In[110]:


df_full['Title']=df_full['Name'].str.split(', ', expand=True)[1].str.split('. ',expand=True)[0]
df_full['Title'].value_counts()


# In[111]:


Weird_Title = ['Rev','Mlle','Col','Marjor','Capt','Jonkheer','Mme','th','Lady','Major', 'Dr', 'Dona','Don']
df_full[df_full['Title'].isin(Weird_Title)].sort_values(by=['Sex','Title'], ascending = True)


# In[112]:


df_full['Title']=df_full['Title'].replace(['Lady', 'Mlle','Mme','th','Ms', 'Dona'], 'Miss')
df_full['Title']=df_full['Title'].replace(['Rev','Col','Marjor','Capt','Jonkheer','Don','th', 'Sir','Major'], 'Mr')
df_full['Title'].value_counts()


# # Family Size
# 
# Family size = # of Sibling + # of Parent + Attendent(him/herself)

# In[113]:


df_full['Fam_num']=df_full['SibSp']+df_full['Parch']+1
df_full.head()


# # Feature Selection

# In[114]:


# To remove 
df_full.drop(['PassengerId' ,'Name', 'Ticket'],axis=1, inplace=True)


# # Dataset Splitting
# 
# 1. Splitting cleaned dataset back into training and test datasets.
# 2. **PassengerId**, **Name** and **Ticket** will be dropped before splitting.

# In[115]:


df_train_cleaned = df_full.iloc[0:891,:]
df_test_cleaned = df_full.iloc[891:,:]


# In[116]:


X = df_train_cleaned.drop(['Survived'], axis=1)
y = df_train_cleaned['Survived']


# In[117]:


X.describe()


# In[118]:


X.describe(exclude='number')


# # Dealing with categorical variables
# 
# Logistic Regression cannot directly work with variables with categorical values
# 
# Two general approaches:
# 
# 1. Get_dummies
# 2. OneHotEncoder
# 
# For the sake of ease, I'm going to utilize **get_dummies** function built in *pandas* library.
# 
# Note: To avoid the perfect Multicollinerity, the first columns produced by the function will be removed.

# In[119]:


X=pd.get_dummies(X, drop_first=True)
X.head()


# In[120]:


#np.corrcoef(X['Sex_female'],X['Sex_male'])
sns.heatmap(X.corr(), annot=True,fmt=".1f");


# **Observation:**
# 
# Highly correlated variable matches are
# 
# 1. Title_Mr & Sex_male: 0.9
# 2. Title_Miss & Sex_male:-0.7

# # Final Dataset

# In[121]:


X.info()


# # Splitting Datasets into Training and Valudation

# In[122]:


from sklearn.model_selection import train_test_split


# In[123]:


X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state=17)


# In[124]:


from sklearn.linear_model import LogisticRegression


# In[125]:


lr = LogisticRegression(random_state=17, class_weight='balanced')
lr.fit(X_train,y_train)


# In[126]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[127]:


# Without 'balanced': 0.6902
# with: 0.723; 0.746268656716418
# with Sex and Embarked: 0.7649
print(accuracy_score(y_valid, lr.predict(X_valid)))
print(confusion_matrix(y_valid,lr.predict(X_valid)))


# **Probability and Feature Importance**

# In[128]:


prob = lr.predict_proba(X_train)
prob_df = pd.DataFrame({'prob_no': prob[:,0],
                       'prob_yes': prob[:,1],
                       'actual': y_train}, index=X_train.index)
prob_df.head()


# In[129]:


pd.DataFrame({'features': X_valid.columns,
              'coef': lr.coef_.flatten().tolist(),
              'abs_coef': np.abs(lr.coef_.flatten().tolist())}).sort_values(by='abs_coef', ascending=False)


# # Random Forest

# In[130]:


from sklearn.ensemble import RandomForestClassifier


# In[131]:


rf = RandomForestClassifier(random_state=17, class_weight='balanced')
rf.fit(X_train,y_train)


# **Feature Importance for Random Forest**

# In[132]:


pd.DataFrame({'Feature': X_train.columns,
             'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)


# In[133]:


print ('Accuracy (Test): %.3f' % accuracy_score(y_valid,rf.predict(X_valid)))
print (confusion_matrix(y_valid, rf.predict(X_valid)))


# In[134]:


from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train,y_train)


# In[135]:


print ('Accuracy (Test): %.3f' % accuracy_score(y_valid,
                                                xgb_model.predict(X_valid)))


# In[137]:


print (accuracy_score(y_valid,
                      xgb_model.predict(X_valid)))


# # Hyperparameter Tuning - Logistic Regression & Random Forest

# In[138]:


from sklearn.model_selection import GridSearchCV


# In[143]:


# Set up the hyperparamter grid
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}

lr_cv = GridSearchCV(lr, parameters, scoring='accuracy', cv=5)
lr_cv.fit(X_train,y_train)


# In[144]:


lr_cv.best_score_, lr_cv.best_params_


# In[145]:


accuracy_score(y_valid, lr_cv.predict(X_valid))


# In[147]:


parameters_rf = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]}

rf_cv=GridSearchCV(rf, parameters_rf, scoring='accuracy',cv=5, n_jobs=-1)
rf_cv.fit(X_train,y_train)


# In[148]:


rf_cv.best_score_, rf_cv.best_params_


# In[149]:


accuracy_score(y_valid, rf_cv.predict(X_valid))


# # XGB Classifier

# In[155]:


parameters_xgb = {
    "n_estimators": [10,20,30,40,50,60,70,80,90,100],
    "learning_rate": [0.1, 0.2, 0.3,0.4,0.5]
}

xgb_cv=GridSearchCV(xgb_model, parameters_xgb, scoring = 'accuracy',cv=5, n_jobs=-1)
xgb_cv.fit(X_train,y_train)


# In[160]:


print (xgb_cv.best_score_)
print (xgb_cv.best_params_)


# In[161]:


accuracy_score(y_valid, xgb_cv.predict(X_valid))


# # Transform Test Dataset
# 
# Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S, Title_Master, Title_Miss, Title_Mr, Title_Mrs

# In[ ]:


df_test_cleaned=pd.get_dummies(df_test_cleaned, drop_first=True)
df_test_cleaned.drop(['Survived'], axis=1, inplace=True)
df_test_cleaned.head()


# # Prediction

# In[ ]:


pred = xgb_cv.predict(df_test_cleaned)
pred = pred.astype(np.int64)


# In[ ]:


output= pd.DataFrame({'PassengerId': pred_passid,
                     'Survived': pred})


# In[ ]:


output.to_csv('titanic.csv', index=False)


# **Learnings From others' Kernel**
# 
# https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python
# https://www.kaggle.com/zlatankr/titanic-random-forest-82-78
# 
# 1. Age - Median 
# 2. New feature - alone or with family; titles from the names
# 3. pd.get_dummies
# 4. Need more visualizations
# 5. pass 'scoring=roc_auc' to the model or other scoring
