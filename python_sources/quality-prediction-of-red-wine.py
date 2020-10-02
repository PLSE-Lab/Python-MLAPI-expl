#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
pd.set_option('display.max_columns',None)


# In[ ]:


df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')


# In[ ]:


df


# First and foremost will check is there any Nan values present in the Dataset.

# In[ ]:


df.isnull().sum()


# There are no Nan values present in Dataset.

# In[ ]:


df.quality.value_counts()


# As there are many labels, we will divide it into 3 labels. 

# In[ ]:


df['quality'] = np.where(df['quality']<=4,0,df['quality'])


# In[ ]:


df['quality'] = np.where((df['quality']<=6) & (df['quality']!=0 ),1,df['quality'])


# In[ ]:


df['quality'] = np.where( df['quality']>=7,2,df['quality'])


# we have converted quality variable into three labels as 0-poor,1-good,2-best.

# In[ ]:


df.quality.value_counts()


# As we can see here,Dataset is completely imbalanced.
# 
# so,we need to fix it. Otherwise your model will baised to single label.

# In[ ]:


from imblearn.combine import SMOTETomek


# In[ ]:


smk = SMOTETomek(random_state=0)


# In[ ]:


X,y=smk.fit_sample(df.drop('quality',axis=1),df['quality'])


# In[ ]:


df = pd.concat([X,y],axis=1)


# In[ ]:


df.quality.value_counts()


# Now, it is perfectly balanced dataset

# In[ ]:


df.head()


# Here,All the variables(features) are of Numerical type.

# will analyse it one by one. 

# In[ ]:


features = [feature for feature in df.columns if feature!='quality']


# In[ ]:


for feature in features:
    sns.boxplot(x=feature,data=df)
    plt.xlabel(feature)
    plt.show()


# As we can see there are number of Outliers present in each feature.
# so,here will use top encoding and bottom encoding technique to fix this.

# In[ ]:


dic = {}
for feature in features:
    IQR = df[feature].quantile(0.75) - df[feature].quantile(0.25)
    upper_bond = df[feature].quantile(0.75) + (IQR * 1.5)
    lower_bond = df[feature].quantile(0.25) - (IQR * 1.5)
    
    df[feature] = np.where(df[feature]>upper_bond,upper_bond,df[feature])
    df[feature] = np.where(df[feature]<lower_bond,lower_bond,df[feature])


# In[ ]:


for feature in features:
    sns.boxplot(x=feature,data=df)
    plt.xlabel(feature)
    plt.show()


# Now we will move to feature selection part.

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


selectk = SelectKBest(score_func=chi2,k=7)


# In[ ]:


Best = selectk.fit(df.drop('quality',axis=1),df['quality'])


# In[ ]:


Best.scores_


# These are the scores related to each feature with respect to output variable(quality).

# In[ ]:


features


# In[ ]:


dfscores = pd.DataFrame(Best.scores_)
dffeatures = pd.DataFrame(features)


# we are mapping each score with respect to each feature recpectively.

# In[ ]:


features_scores = pd.concat([dffeatures,dfscores],axis=1)


# In[ ]:


features_scores.columns = ['feature','scores']


# In[ ]:


features_scores.sort_values(by='scores',ascending=False)


# we will take top 7 features

# In[ ]:


Best_features = features_scores[features_scores['scores']>30]['feature']


# Feature Selection with the help of correlation

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


# From above we can notice that volatile acidity,citric acid,alcohol and sulphates are correlated more than fifty percent to target variable (quality).

# Now we split our dataset into train and test dataset.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(df[Best_features],df['quality'],test_size=0.2,random_state=0)


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification_report


# But Outliers do not impact much on tree based models.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


model = DecisionTreeClassifier()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


y_predict = model.predict(X_test)


# In[ ]:


y_predict_proba_train = model.predict_proba(X_train)


# In[ ]:


y_predict_proba_test = model.predict_proba(X_test)


# In[ ]:


roc_auc_score(y_train,y_predict_proba_train,multi_class='ovo')


# In[ ]:


roc_auc_score(y_test,y_predict_proba_test,multi_class='ovo')


# As we know Decision tree follows low bias and high variance. Which means for training dataset it gives high accuracy but for testing dataset it gives less accuracy.
# 
# This problem can be easily solved with the help of ensemble techniques. 
# 
# e.g - RandomForest,XGBoost.

# In[ ]:


confusion_matrix(y_test,y_predict)


# In[ ]:


accuracy_score(y_test,y_predict)


# In[ ]:


print(classification_report(y_test,y_predict))


# From above we can see that for class 1 precision and recall is falling behind

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


cross_val_score(model,df[Best_features],df['quality'],scoring='accuracy',n_jobs=-1).mean()


# with cross_val_score, Decisison Tree is giving 82% accuracy. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model = RandomForestClassifier()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


y_predict = model.predict(X_test)


# In[ ]:


y_predict_proba_train = model.predict_proba(X_train)


# In[ ]:


y_predict_proba_test = model.predict_proba(X_test)


# In[ ]:


roc_auc_score(y_train,y_predict_proba_train,multi_class='ovo')


# In[ ]:


roc_auc_score(y_test,y_predict_proba_test,multi_class='ovo')


# As we can see RandomForest fixed the problem of low bias high variance to low bias low variance.

# In[ ]:


confusion_matrix(y_test,y_predict)


# In[ ]:


accuracy_score(y_test,y_predict)


# In[ ]:


print(classification_report(y_test,y_predict))


# Precision and Recall is improved with Random Forest

# In[ ]:


cross_val_score(model,df[Best_features],df['quality'],scoring='accuracy',n_jobs=-1).mean()


# we can see Random  Forest Classifier is giving 88% with cross_val_score
# 
# Now we will check with XGBClassifier.

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


model = XGBClassifier()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


y_predict = model.predict(X_test)


# In[ ]:


y_predict_proba_train = model.predict_proba(X_train)


# In[ ]:


y_predict_proba_test = model.predict_proba(X_test)


# In[ ]:


roc_auc_score(y_train,y_predict_proba_train,multi_class='ovo')


# In[ ]:


roc_auc_score(y_test,y_predict_proba_test,multi_class='ovo')


# In[ ]:


confusion_matrix(y_test,y_predict)


# In[ ]:


accuracy_score(y_test,y_predict)


# In[ ]:


print(classification_report(y_test,y_predict))


# Precision and Recall is further improved with XGBoost

# In[ ]:


cross_val_score(model,df[Best_features],df['quality'],scoring='accuracy',n_jobs=-1).mean()


# XGBClassifier is giving 90% accuracy with cross_val_score

# we will improve model accuracy by using Hyperparameter Optimization.
# 
# Here we are using RandomizedSearchCV.

# In[ ]:


params = {
    'n_estimators' : list(np.arange(5,101,1)) ,
    'max_depth' : list(np.arange(3,16,1)) ,
    'min_child_weight' : [1,3,4,5,6,7,8] ,
    'learning_rate' : list(np.arange(0.05,0.35,0.05)) ,
    'colsample_bytree' : [0.4,0.5,0.6,0.7],
    'gamma' : [0.0,0.1,0.2,0.3,0.4]    
}


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


random_search = RandomizedSearchCV(model,param_distributions=params,n_jobs=-1,scoring='accuracy',verbose=3,cv=5)


# In[ ]:


random_search.fit(df[Best_features],df['quality'])


# In[ ]:


random_search.best_estimator_


# In[ ]:


model = XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.4, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.3, max_delta_step=0, max_depth=10,
              min_child_weight=1, monotone_constraints=None,
              n_estimators=37, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method=None, validate_parameters=False, verbosity=None)


# In[ ]:


cross_val_score(model,df[Best_features],df['quality'],scoring='accuracy',n_jobs=-1).mean()


# As we can see with the help of Hyperparameter Optimization we have improved 1% accuracy

# #### I hope you enjoyed a lot.

# #### Thank You
