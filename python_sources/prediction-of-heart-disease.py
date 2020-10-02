#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns',None)


# In[ ]:


df = pd.read_csv('../input/heart-disease-uci/heart.csv')


# In[ ]:


df


# we will first check for nan values.

# In[ ]:


df.isnull().sum()


# As you can see there are no nan values present in the dataset.

# We will extract all the features.

# In[ ]:


features = [feature for feature in df.columns if feature!= 'target']


# Dividing it into Discrete and Continous.

# In[ ]:


dis_feature = [ feature for feature in features if len(df[feature].unique()) < 10 ]


# In[ ]:


dis_feature


# We will go for count plot for understanding insight of data.

# In[ ]:


for feature in dis_feature:
    sns.countplot(x=feature,data=df,hue='target')
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.show()


# From above,
# 
#             1 : cp,ca,slope,thal are playing important role with respect to target.
#             2 : fbs has 50-50% probability. so,it is weak to predict the target

# Relationship of every feature with respect to target.

# In[ ]:


for feature in dis_feature:
    df.groupby(feature)['target'].mean().plot()
    plt.xlabel(feature)
    plt.show()


# As, they are not in perfect relationship, we will bring it to monotonic relationship with the help of target guided encoding. 

# In[ ]:


for feature in dis_feature:
    mean = df.groupby(feature)['target'].mean()
    index = mean.sort_values().index
    ordered_labels = { k:i for i,k in enumerate(index,0) }
    df[feature] = df[feature].map(ordered_labels)
    


# In[ ]:


for feature in dis_feature:
    df.groupby(feature)['target'].mean().plot()
    plt.xlabel(feature)
    plt.show()


# As you can see, all the features are now in monotonic relationship. 

# Extracting Continous features.

# In[ ]:


con_feature = [ feature for feature in features if feature not in dis_feature]


# In[ ]:


con_feature


# First and foremost we will check histogram of each feature. 

# In[ ]:


for feature in con_feature:
    df[feature].hist(bins=10)
    plt.xlabel(feature)
    plt.show()


# Above all the features, oldpeak is highly skewed.
# 
# The 1/3 rd of oldpeak are 0(zeros). so,we will use it as it is.

# According to above histograms, there we will be very few outliers.

# Now, we will check for outliers

# In[ ]:


for feature in con_feature:
    sns.boxplot(x=feature,data=df)
    plt.show()


# As predicted there are very few outliers.

# For Feature selection, we are using SelectKbest and chi2

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


selectk = SelectKBest(score_func=chi2,k=9)


# In[ ]:


feature_scores = selectk.fit(df.drop('target',axis=1),df['target'])


# In[ ]:


feature_scores.scores_


# In[ ]:


df_scores = pd.DataFrame(feature_scores.scores_)
df_features = pd.DataFrame(features)


# In[ ]:


features_scores = pd.concat([df_features,df_scores],axis=1)


# In[ ]:


features_scores.columns = ['features','scores']


# In[ ]:


features_scores.sort_values(by='scores',ascending=False,inplace=True)


# In[ ]:


features_scores


# As I already discussed in Feature engneering cp,thal are more correlated with target.

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


# There are many features with correlation more than 0.4.

# Extracting those features which scored more than 18.

# In[ ]:


Best_features = features_scores[features_scores['scores']>18]['features'].values


# In[ ]:


Best_features


# We are using Ensemble technique because it does not over fit. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


model = RandomForestClassifier()


# In[ ]:


cross_val_score(model,df[Best_features],df['target'],cv=10).mean()


# As you can see we have got 82% accuracy.
# 

# We will tune the parameters of Random Forest to improve further accuracy.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


params = {
    'n_estimators' : list(np.arange(10,101,1)),
    'max_depth' :  list(np.arange(3,30,1)),
    'min_samples_leaf' :  list(np.arange(1,10,1)),
    'min_samples_split' :  list(np.arange(1,10,1))
}


# In[ ]:


random_search = RandomizedSearchCV(model,param_distributions=params,n_jobs=-1,n_iter=10,scoring='f1_macro',cv=5,verbose=3)


# In[ ]:


random_search.fit(df[Best_features],df['target'])


# In[ ]:


random_search.best_estimator_


# In[ ]:


model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=22, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=7,
                       min_weight_fraction_leaf=0.0, n_estimators=41,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)


# In[ ]:


cross_val_score(model,df[Best_features],df['target'],cv=10).mean()


# As you can see,Hyper Parameter tuninig improved accuracy to 84%

# #### I hope you learned some new things...

# #### Thank You.
