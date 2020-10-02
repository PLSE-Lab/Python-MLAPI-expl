#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ed-chin-git/DS-Unit2-Kaggle-Competition/blob/master/Unit_2_Kaggle_Competition_Tanzanian_Water_Pumps.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#  # Class Challenge : Predicting Water Pump Failure in Tanzania     
# 
# Intro [Video]( https://www.youtube.com/watch?v=NsKpvxpX_eU&feature=youtu.be)
# 
# Open Classroom Review [video](https://www.youtube.com/watch?v=4B4EP6eTzLk)
# 
# Reference Articles:
# *   [Begin with baseline models](https://github.com/rrherr/baselines/blob/master/Begin%20with%20baseline%20models.ipynb)
# *   [Visiting: Categorical Features and Encoding in Decision Trees](https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931)

# ## IMPORTS and EXPORTS

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999


# In[ ]:


def create_submission(pred_array,X_test_df,dest_url):
  pred_df=pd.DataFrame(pred_array,columns=['status_group'])
  pump_ids=pd.DataFrame(X_test_df.id,columns=['id'])
  pump_ids = pump_ids.astype('int32')
  submit_df=pd.concat([pump_ids, pred_df], axis=1)
  submit_df.to_csv(dest_url, index=False, header=['id','status_group'])
  return


# ## Load DATA

# In[ ]:


X_train_url = '../input/train_features.csv'
y_train_url = '../input/train_labels.csv'
X_test_url = '../input/test_features.csv'
y_test_url = '../input/sample_submission.csv'

train_features = pd.read_csv(X_train_url)
train_labels = pd.read_csv(y_train_url)
test_features = pd.read_csv(X_test_url)
sample_sub =pd.read_csv(y_test_url)


# In[ ]:


print(train_features.shape)
train_features.head()


# In[ ]:


train_labels.head()


# ###  Quick Viz inspired by Sir Ryan Herr

# In[ ]:


df_viz=train_features.copy()
df_viz[df_viz['longitude']>0] [df_viz['latitude']<0][df_viz['construction_year']>0].plot(kind='scatter', x="longitude", y="latitude", alpha=0.4,
s=df_viz["population"]/10, label="population", figsize=(14,10),
c="construction_year", cmap=plt.get_cmap("Blues"), colorbar=True,
sharex=False);
plt.title("Waterpump Locations in Tanzania", 
         fontsize =16, fontweight='bold')
plt.legend;


# ## SIMPLE BASELINE  submission using Majority Class
# 
# 
# This gets a [**Kaggle Score**](https://www.kaggle.com/c/ds1-predictive-modeling-challenge/submissions?sortBy=date&group=all&page=1) of : ** .53754**

# In[ ]:


train_labels.head()


# **Look at Percentage of value counts**

# In[ ]:


train_labels.status_group.value_counts(normalize=True)


# **Establish Mode as Majority Class**

# In[ ]:


majority_class = train_labels.status_group.mode()[0]


# **Create All Majority-class Predictions**

# In[ ]:


submit_df=sample_sub  ## make a copy of the sample submission df
submit_df.status_group.replace(majority_class) ## replace predicted label with majority class


# ## Create baseline submission

# In[ ]:


submission_url = '../output/submission.csv'
submit_df.to_csv(submission_url, index=False, header=['id','status_group'])


# ## Data Prep
#  
#  Make sure any changes made to the Training-dataset are also made to the Test-dataset

# ###  Combine the features and labels for data wrangling

# In[ ]:


df_train=pd.merge(train_features,train_labels,how='left', on=['id'])
df_test=test_features.copy()

df_train.head()


# ## Null Values

# In[ ]:


#label NaNs as unknown for One Hot Encoding as their own feature
df_train.funder.fillna('unknown', inplace=True)
df_train.permit.fillna('unknown', inplace=True)
df_train.installer.fillna('unknown', inplace=True)
df_train.subvillage.fillna('unknown', inplace=True)
df_train.scheme_name.fillna('unknown', inplace=True)
df_train.public_meeting.fillna('unknown', inplace=True)
df_train.scheme_management.fillna('unknown', inplace=True)

df_test.funder.fillna('unknown', inplace=True)
df_test.permit.fillna('unknown', inplace=True)
df_test.installer.fillna('unknown', inplace=True)
df_test.subvillage.fillna('unknown', inplace=True)
df_test.scheme_name.fillna('unknown', inplace=True)
df_test.public_meeting.fillna('unknown', inplace=True)
df_test.scheme_management.fillna('unknown', inplace=True)


# ## Drop Features that may be meaningless

# In[ ]:


dropped_features=['date_recorded',
                         'funder',
                      'installer',
                      'longitude',
                       'latitude',
                       'wpt_name',
                    'num_private',
                          'basin',
                     'subvillage',
                         'region',
                            'lga',
                           'ward',
                    'recorded_by',
               'scheme_management',
                    'scheme_name',
          'waterpoint_type_group',                  
                'extraction_type',
          'extraction_type_group',
          'extraction_type_class', ]

df_train.drop(columns=dropped_features, inplace=True )
df_test.drop(columns=dropped_features, inplace=True )


# ## One Hot Encode Features with multiple values
# 
# [CE encoder](http://contrib.scikit-learn.org/categorical-encoding/onehot.html)
# 
# SKlearn [Preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
# 

# In[ ]:


cols_enc=['public_meeting','management', 'management_group', 'payment','payment_type',
          'water_quality','quality_group', 'quantity',  'quantity_group'   ,'source', 'source_type','source_class','waterpoint_type','permit'  ]

df_train_enc=pd.get_dummies(df_train, columns=cols_enc, prefix=cols_enc)
df_test_enc=pd.get_dummies(df_test, columns=cols_enc, prefix=cols_enc)


df_test_enc.head(10)


# ### Feature engineering
#    
#    use year_constucted to calculate age of pump

# In[ ]:


# Calculate construction year mean to fill missing data (year=0)
mean_year = df_train_enc[df_train_enc['construction_year']>0]['construction_year'].mean()
df_train_enc.loc[df_train_enc['construction_year']==0, 'construction_year'] = int(mean_year)


# In[ ]:


mean_year = df_test_enc[df_test_enc['construction_year']>0]['construction_year'].mean()
df_test_enc.loc[df_test_enc['construction_year']==0, 'construction_year'] = int(mean_year)


# In[ ]:


df_train_enc['age']=(2018 - df_train_enc.construction_year).astype(float)
df_test_enc['age']=(2018 - df_test_enc.construction_year).astype(float)


# ## Create Model Data

# In[ ]:


X_train = df_train_enc.drop(columns='status_group').copy()
y_train = df_train_enc.status_group.copy()

X_test = df_test_enc.copy()


# In[ ]:


X_train.head()


# In[ ]:


# convert whole dataset to float64

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# ## Attempt 2  :  using SKlearn -  LogisticRegression 
# This gets a [**Kaggle Score**](https://www.kaggle.com/c/ds1-predictive-modeling-challenge/submissions?sortBy=date&group=all&page=1) of : ** .72558**
#  SKLearn [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) Docs
#  
#  [2 WAYS TO IMPLEMENT MULTINOMIAL LOGISTIC REGRESSION IN PYTHON](http://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/)
#  
#  

# In[ ]:


mul_lr = LogisticRegression().fit(X_train, y_train)


# ### Run prediction and create submission csv

# In[ ]:


lr_pred=mul_lr.predict(X_test)

submission_url = '/content/drive/My Drive/Colab Notebooks/submission.csv'
create_submission(lr_pred,X_test,submission_url)


# ## Attempt 3 : Setup a Pipeline and Cross Validate
# This gets a [**Kaggle Score**](https://www.kaggle.com/c/ds1-predictive-modeling-challenge/submissions?sortBy=date&group=all&page=1) of : ** .69424**
# 
# Docs for [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) [scoring parameter](https://scikit-learn.org/stable/modules/model_evaluation.html)
# 
# 

# In[ ]:


pipe = make_pipeline(
    RobustScaler(),  
    SelectKBest(f_classif), 
    LogisticRegression())


# In[ ]:


# select hyper-parameters 
param_grid = {
    'selectkbest__k': [1,2,3], 
    'logisticregression__class_weight': [None, 'balanced'],
    }


# Fit on the train set   3-folds,  scoring=accuracy
gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, 
                  scoring='accuracy', 
                  iid=False,
                  verbose=1)

gs.fit(X_train, y_train)
gs_preds=gs.predict(X_test)


# In[ ]:


submission_url = '/content/drive/My Drive/Colab Notebooks/submission.csv'
create_submission(gs_preds,X_test,submission_url)


# ## Attempt 4 :  Random Forest
# This gets a [**Kaggle Score**](https://www.kaggle.com/c/ds1-predictive-modeling-challenge/submissions?sortBy=date&group=all&page=1) of : ** .79704**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_estimators=200,min_samples_leaf=3 ,n_jobs=-1,max_features=0.25)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
m_preds= m.predict(X_test)


# In[ ]:


submission_url = '/content/drive/My Drive/Colab Notebooks/submission.csv'
create_submission(m_preds,X_test,submission_url)


# ## Attempt 5 :  RobustScaler + Random Forest
# This gets a [**Kaggle Score**](https://www.kaggle.com/c/ds1-predictive-modeling-challenge/submissions?sortBy=date&group=all&page=1) of : ** .?????**

# In[ ]:


transformer = RobustScaler().fit(X)
transformer
transformer.transform(X)


# In[ ]:


m = RandomForestClassifier(n_estimators=200,min_samples_leaf=3 ,n_jobs=-1,max_features=0.25)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
m_preds= m.predict(X_test)


# In[ ]:


submission_url = '/content/drive/My Drive/Colab Notebooks/submission.csv'
create_submission(m_preds,X_test,submission_url)


# ## Attempt : Extreme Gradient Boost
# 
# Practical [XgBoost](https://www.youtube.com/playlist?list=PLZnYQQzkMilqTC12LmnN4WpQexB9raKQG) in Python
# 
# [Notebooks](https://github.com/ParrotPrediction/docker-course-xgboost/tree/master/notebooks) for Practical Xgboost in Python
# 
# 
# 

# In[ ]:





# ## Attempt  : SUPPORT VECTOR MACHINE 
# 
# SVM CLASSIFIER, INTRODUCTION TO [SUPPORT VECTOR MACHINE ALGORITHM](https://dataaspirant.com/2017/01/13/support-vector-machine-algorithm/)
# 

# In[ ]:




