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


# **Module Import Section**
# 
# 
# Here we import main two main visualization libraries for some basic EDA, and also to suppress all warnings. It is to mention suppressing warnings is just a method to show some cleaner output, but one may avoid it if they want

# In[ ]:


#importing other necessary modules for visualization and label-encoding
import seaborn as sns
import matplotlib.pyplot as plt

#warning suppressing section
import warnings
warnings.filterwarnings('ignore')


# **Data Import Section**
# 
# 

# In[ ]:


data_train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# **Introduction with the data**
# 
# It is important to know about the given features of our dataset and missing values for each feature. We also check basic statistics of the data using the describe method. 

# In[ ]:


data1 = data_train.copy()
combo = [data1, test]
print(data1.isnull().sum())
print(data1.describe())


# In[ ]:


print(test.isnull().sum())
print(test.describe())


# In[ ]:


data1.head(10)


# From the above results we found that **Age, Embarked , Cabin, Fare** contains missing values and hence must undergo some missing value treatment. Fare column contains a large number of missing values hence it is better to drop it entirely, rest are to be filled in

# In[ ]:


sns.boxplot(data1.Age)


# In[ ]:


sns.boxplot(data1.Fare)


# Here we see that age and and Fare are not evenly distributed through out the entire range but we will try to retain this structure after the missing value treatments. Also Ticket column will not contribute much since ther is no numbering or any kind of tagging cannot be assigned with that ticket number. PassengerId is dropped because we already have passenger name details for reference purpose.

# In[ ]:


for i,d in enumerate(combo):
    d.Age.fillna(d.Age.median(), inplace=True)
    d.Embarked.fillna(d.Embarked.mode()[0], inplace=True)
    d.Fare.fillna(d.Fare.median(),inplace=True)
    d = d.drop(columns=['Cabin','PassengerId','Ticket'], axis=1)              
    print(d.isnull().sum())
    print("-"*20)
    combo[i]=d


# **Basic EDA**
# 
# Since **Survived** is the target variable hence it is important know the effects of other variables on Target Variable. From a first glance at the data it is obvious to look for the answer how survival is affected by gender of the passengers

# In[ ]:


data1=combo[0]
data1.groupby('Sex')['Survived'].value_counts(normalize=True)


# Here we see a high survival rate for female passengers and a very low survival rate for male passengers so we may expect this pattern to retain over test dataset too.
# 
# Next another important factor comes into consideration is **Age** of the passenger. So we check the histogram for age of the passengers to have an idea regarding its distribution

# In[ ]:


sns.set_style('darkgrid')
sns.distplot(data1.Age,hist=True,kde=True,bins=25)


# It is to notice that there were a good number of children ( Age < 5) among passengers so , it is better to look for their survival rate and there are a few older people too ( Age > 65) but they are very less in number. Also the distribution almost following a properly distributed normal curve with slight high density in the starting region and very mildly skewed towards lower age side, but it will not make much impact. 

# In[ ]:


print("Total passengers within age group below Five:",data1[data1.Age<5]['Survived'].count())
print("Survived:",data1[(data1.Age<5)&(data1.Survived==1)]['Survived'].count())


# We notice that more than 50% of the children in the age group below Five survived, hence this pattern will retain in test dataset too
# 

# **Feature Engineering**
# 
# It is always a good idea to generate new features, sometimes it helps to improve the model greatly. But we need to avoid collinearity before fitting to the model.
# We Introduce two new features
# * 1. Family Size: since we have information about number of siblings ,spouse, parent so it is easy to calculate the family size of an aboarded individual.
# * 2. Title Tags: Since every name comes with it title tags (Mr., Mrs. etc) it may play some role with the survival values in our data

# In[ ]:


data1['title']=d['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
data1.title.value_counts()


# Well, we go for the majority and mark the rest ones with tag 'Misc.'

# In[ ]:


for i,d in enumerate(combo):
    d = d.copy()
    d['family_size']= d['Parch']+d['SibSp']+1
    #name_breakdown
    d['title']=d['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
    title_tags = d.title.value_counts() <10
    d['title']=d.title.apply(lambda x:'Misc' if title_tags.loc[x]==True else x)
    combo[i]= d


# Next We have two continuous variables which may cut into different groups for some quick insights about age and fare. 
# * 1. Farebin: It splits the Fare column into different groups which can easily identify whether a passenger is within high , low or some intermediate medium range
# * 2. Agebin: Similar to Farebin it splits the entire age into different age groups for quick identification

# In[ ]:


for i,d in enumerate(combo):
    d= d.copy()
    d['Farebin']=pd.qcut(d.Fare.astype('int'), 4)
    d['Agebin']=pd.cut(d.Age.astype('int'), 5)
    combo[i]=d


# **Treatments of Categorical Variables**
# 
# In order to deal with categorical variables we encode them with different labels with integer values. Since rest of the values are numeric it is a good idea to bring all of them under similar system. As for later purpose we may need to use some feature scaling and then all variables may be treated the same way. Though in this problem we may not require that but many other machine learning problem requires some feature scaling for better results 

# In[ ]:


#label encoding for the train and test data
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for i,d in enumerate(combo):
    d=d.copy()
    d['sex_code']=enc.fit_transform(d.Sex)
    d['embarked_code']=enc.fit_transform(d.Embarked)
    d['title_code']=enc.fit_transform(d.title)
    d['agebin_code']=enc.fit_transform(d.Agebin)
    d['farebin_code']=enc.fit_transform(d.Farebin)
    combo[i]=d


# In[ ]:


data_enc = combo[0]
print(data_enc.columns)


# **Dummy Variable Introduction**
# 
# The more features we have the more we have insights about the data, hence dummy variable introduction is a good thing to practice plus it helps us to show more details about categorical variable regarding which categorical value is contributing more to the data

# In[ ]:


#dummy variable creation
y_feature=['Survived']
X_ft=data_enc.columns
X_ft_copy = X_ft.copy()
X_ft_copy = X_ft_copy.drop(['Survived', 'Name','Farebin','Agebin'])
X_dummy = pd.get_dummies(data_enc[X_ft_copy])
print(X_ft_copy)
print(X_dummy.columns)


# **Correlation Matrix Visualization**
# 
# Since we are using many features , it is possible that there are some redundant features which are not contributing that much. It also show if there is multicollinearity between two features, which must be avoided. We use heatmap to visualize them quickly and easily

# In[ ]:


X_aug_y = X_dummy; X_aug_y['y']= data1['Survived']
plt.figure(figsize=(15,10))
sns.heatmap(X_aug_y.corr(), annot=True, cmap='coolwarm')


# As of the conclusion from heatmap, the following features are dropped because of multicollinearity and also some of them are not contributing much to towards target variable classification

# In[ ]:


to_drop = ['SibSp','Parch','Sex_female','Sex_male','Embarked_Q','embarked_code','title_Misc','y']
X_dummy = X_dummy.drop(to_drop, axis=1)
#X_dummy.columns


# **Target Variable Preparation**

# In[ ]:


y=data1['Survived']


# **Model Fitting **
# 
# We approach towards the Gradient Boosting Classifier to solve our problem. It is quite a helpful algorithm in terms of binary classification. Scikit-Learn library provides the classifier model pre-trained, yet hyperparameters are tuned to for better result and avoid overfitting
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
# 

# The following function fits the model and evaluates necessary metrics for model evaluation and also shows the important features suggested by the model as of high priority. This following tutorial gave me the idea for building this and it is quite helpful in terms of a compact model performance evaluation
# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

# In[ ]:


#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def evaluate(estimator, data, features, performCV=True, printFeatureImportance=True, cv=5):
    #Fit the algorithm on the data
    estimator.fit(data[features], y)
        
    #Predict training set:
    predictions = estimator.predict(data[features])
    predprob = estimator.predict_proba(data[features])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(estimator, data[features], y, cv=cv, scoring='roc_auc')
    
    #Print model report:
    print("Model Report")
    print("Accuracy : {}".format(accuracy_score(y,predictions)))
    print("F1 Score : {}".format(f1_score(y,predictions)))
    print("AUC Score (Train): {}".format(roc_auc_score(y, predprob)))
    
    if performCV:
        print("CV Score : Mean - {:.7f} | Std - {:.7f} | Min - {:.7f} | Max - {:.7f}"               .format(np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(estimator.feature_importances_, features).sort_values(ascending=False)
        feat_imp.plot(kind='bar',title='Feature Importances')
        plt.ylabel('Feature Importance Score')


# First we create a baseline gradient boosting model with hyperparameters not properly tuned and  prone to overfittin then we will reach upto a properly tuned model for our classification purpose. In the following steps we tune the model and evaluate its result so that eventually we reach upto a better classification model and showing important features step by step after proper tuning. Grid Search Cross Validation is somewhat a lengthy process if multiple parameters are tuned simulteneously so it had broken to several steps. 

# In[ ]:


#baseline gbm model
features = X_dummy.columns
gbm0=gbc(random_state=20)
evaluate(gbm0, X_dummy, features)


# In[ ]:


#tuning1:
gbm1=gbc(min_samples_split=10,min_samples_leaf=5,max_depth=3, max_features='sqrt', subsample=0.8, random_state=20)
param_tune1 = {'n_estimators':range(10,51,10)}
gridCV1=GridSearchCV(estimator=gbm1, param_grid=param_tune1, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
gridCV1.fit(X_dummy,y)


# After fitting gridsearch CV model properly to our data we check best parameters and corresponding scores after every step. This parameter will be used in the model. Since this is a binary classification problem hence ROC-AUC is a good evaluation metric for evaluation of classification power of the model. 

# In[ ]:


print(gridCV1.best_params_, gridCV1.best_score_)


# In[ ]:


#tuning2:
gbm2=gbc(n_estimators=40,min_samples_leaf=5,random_state=20,subsample=0.8)
param_tune2 = {'max_depth':range(2,11),
               'min_samples_split':range(5,20),
                'max_features':['auto','sqrt']}
gridCV2 = GridSearchCV(estimator=gbm2,param_grid=param_tune2, scoring='roc_auc',iid=False, n_jobs=-1,cv=5)
gridCV2.fit(X_dummy,y)
print(gridCV2.best_params_, gridCV2.best_score_)


# In[ ]:


#tuning3:
gbm3=gbc(n_estimators=40,max_depth=3,max_features='sqrt',random_state=20,subsample=0.8)
param_tune3 = {'min_samples_split':range(20,81),
                'min_samples_leaf':range(5,31)}
gridCV3 = GridSearchCV(estimator=gbm3,param_grid=param_tune3, scoring='roc_auc',iid=False, n_jobs=-1,cv=5)
gridCV3.fit(X_dummy,y)
print(gridCV3.best_params_, gridCV3.best_score_)


# In[ ]:


#evaluation based on current parameter
gbm4=gbc(n_estimators=40,min_samples_split=58, min_samples_leaf=5, max_features='sqrt',
        max_depth=3,subsample=0.8, random_state=20)
evaluate(gbm4, X_dummy, features)


# In[ ]:


gbm5 = gbc(min_samples_split=58, min_samples_leaf=5, max_features='sqrt',
          max_depth=3, subsample=0.8, random_state=20)
param_tune4 = {'learning_rate':np.arange(0.01,0.1,0.01),
              'n_estimators':range(40,400,10)}
gridCV4 = GridSearchCV(estimator=gbm5,param_grid=param_tune4, scoring='roc_auc',iid=False, n_jobs=-1,cv=5)
gridCV4.fit(X_dummy, y)
print(gridCV4.best_params_, gridCV4.best_score_)


# In[ ]:


gbm6=gbc(learning_rate=0.02,n_estimators=220, min_samples_split=58, min_samples_leaf=5, max_features='sqrt',
          max_depth=3, subsample=0.8, random_state=20)
#param_tune4={'n_estimators':range(100,1001,100)}
#gridCV4=GridSearchCV(estimator=gbm6, param_grid=param_tune4, scoring='roc_auc', iid=False, n_jobs=-1,
                    #cv=5)
#gridCV4.fit(X_dummy, y)
evaluate(gbm6, X_dummy, features)


# We have so far reached the final step towards model training. The last model will be used for our classification purpose but before that we need to provide the exact same treatment we provided for the case of train dataset, which will make our test dataset to contain the same features as of  the train dataset.
# 
# **Preparaing Test Data for Survival Prediction**
# 
# Here we will use the same dummy variable introduction technique as we used previously and remove all those features we had removed for training data case. This will enable us to use our model to predict using test data.

# In[ ]:


test1=combo[1]
#test1.head(10)
ft=test1.columns
ft_copy = ft.copy()
ft_copy = ft_copy.drop(['Name','Farebin','Agebin'])
X_test = pd.get_dummies(test1[ft_copy])


# In[ ]:


drop = ['SibSp','Parch','Sex_female','Sex_male','Embarked_Q','embarked_code','title_Misc']
X_test=X_test.drop(drop, axis=1)
print(X_dummy.columns)
print(X_test.columns)


# In[ ]:


#Prediction of Survival Labels
test['Survived'] = gbm6.predict(X_test)


# In[ ]:


data = {'PassengerId':test.PassengerId.values,
                          'Survived':test.Survived.values}
submission = pd.DataFrame(data)
submission.to_csv('submission.csv', index=False)


# **Conclusion**
# 
# Here we do some basic analysis of predicted labels of survial of passengers. During basic EDA, we found the following facts
# * 1. There is high survival rate in case of female passengers as compared to male passengers
# * 2. Children below the age of Five had a survival rate of more than 50%
# 
# So, we may expect the same pattern in case of test dataset too. The following part focuses on that

# In[ ]:


test.groupby('Sex')['Survived'].value_counts(normalize=True)


# Here we also see the same pattern in survival. A very high survival rate for female passengers and that of very low for male passengers

# In[ ]:


print("Total passengers within age group below Five:",test[test.Age<5]['Survived'].count())
print("Survived:",test[(test.Age<5)&(test.Survived==1)]['Survived'].count())


# Here we have 100% survival rate for children which is improved as compared to train set.
