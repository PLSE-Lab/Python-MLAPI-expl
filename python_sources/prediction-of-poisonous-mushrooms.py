#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('Display.max_columns',None)


# In[ ]:


df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')


# In[ ]:


df


# First we will check is there any nan values present in the Dataset.

# In[ ]:


df.isnull().sum()


# From above, there are no Nan values present in Dataset.

# Now we will analyse each feature

# In[ ]:


for feature in df.columns:
    df[feature].value_counts().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('counts')
    plt.show()


# From above,
# 
#         1 : 'veil-type' has single label.So, we are droping it.
#         2 : 'stalk-root' has '?' label. we will change it to 'o'(other)
#         3 : There are many features having One or Two pre-dominant labels.
#         
#         

# In[ ]:


df.drop('veil-type',axis=1,inplace=True)


# In[ ]:


df['stalk-root'].unique()


# In[ ]:


df['stalk-root'] = np.where(df['stalk-root']=='?','o',df['stalk-root'])


# We are renaming output variable(class) to 'poisonous' for our convenience.

# In[ ]:


df.rename(columns={'class':'poisonous'},inplace=True)


# In[ ]:


df['poisonous'].value_counts()


# Dataset is balanced. Which is positive sign. 

# we are changing (poisonous) 'p' to 1  and (eatable) 'e' to 0.

# In[ ]:


df['poisonous'] = np.where(df['poisonous']=='p',1,0)


# In[ ]:


df['poisonous'].value_counts()


# In[ ]:


features = [ feature for feature in df.columns if feature != 'poisonous' ]


# Analysing every feature with count plot for better understanding of the data. 

# In[ ]:


for feature in features:
    sns.countplot(x=feature,data=df,hue='poisonous')
    plt.show()
    


# From above,
# 
#            1 : In most of the features, the small amount categories holding more information regarding output variable.
#            2 : odor feature is higly correlated with output variable. 

# In[ ]:


for feature in features:
    df.groupby(feature)['poisonous'].mean().plot.bar()
    plt.xlabel(feature)
    plt.show()


# From above,
# 
#        1 : There are one or more categories in each feature having mean = 1 and mean = 0
#            Which helps to easily classify the Mushroooms.
#         
#        2 : Again odor feature holding more information regarding output.

# We are using Target guided encoding for handling categorical features because it gives us monotonic relationship with output variable.
# 
# We can also use mean encoding.
# 
# 

# In[ ]:


df.groupby(['stalk-color-above-ring'])['poisonous'].mean().sort_values()


# We are directly mapping ordinal numbers to the labels accordingly.   

# In[ ]:


for feature in features:
    ordered_labels = df.groupby([feature])['poisonous'].mean().sort_values().index
    ordinal_label = {k:i for i,k in enumerate(ordered_labels,0)}
    df[feature] = df[feature].map(ordinal_label)
    


# In[ ]:


df.groupby(['stalk-color-above-ring'])['poisonous'].mean().sort_values()


# In this way for every feature, we can easily predict whether if value is less than threshold value than it is treated is eatable (0) else poisonous (1)    

# In[ ]:


df


# Monotonic relationship between every feature and Target feature.
# 
# Hence,This is the reason why we used Target guided encoding.

# In[ ]:


for feature in features:
    df.groupby([feature])['poisonous'].mean().plot()
    plt.show()


# for Feature selection we are using SelectKBest, chi2 (chi square). 

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


selectk = SelectKBest(score_func=chi2,k=12)


# In[ ]:


feature_scores = selectk.fit(df.drop('poisonous',axis=1),df['poisonous'])


# In[ ]:


feature_scores.scores_


# We obtained scores related to every feature with respect to output variable.

# In[ ]:


scores_df = pd.DataFrame(feature_scores.scores_)
features_df = pd.DataFrame(features)
feature_scores_df = pd.concat([features_df,scores_df],axis=1) 
feature_scores_df.columns = ['feature','score']


# We are mapping each feature with respect to its score.

# In[ ]:


feature_scores_df.sort_values(by='score',ascending=False,inplace=True)


# In[ ]:


feature_scores_df


# Visualising for better understanding.

# In[ ]:


feature_scores_df['score'].plot(kind='bar')


# We are selecting those features which are scored more than 800 

# In[ ]:


Best_features = feature_scores_df[feature_scores_df['score']>800]


# In[ ]:


Best_features


# In[ ]:


sns.barplot(x='feature',y='score',data=Best_features)


# In[ ]:


Best_features = Best_features['feature'].values


# In[ ]:


Best_features


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# From above, we can notice that odor, gill-color, ring-type, spore-print-color are correlated more than 60% with respect to output variable.

# We are splitting our Dataset into training and testing with the help of train_test_split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df[Best_features],df['poisonous'],test_size = 0.2,random_state=0 )


# In[ ]:


X_train


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score


# #### Good Feature engineering, Feature selection and Balanced Dataset we got Maximum accuracy for all the above models.

# In[ ]:


model = KNeighborsClassifier()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


y_predict = model.predict(X_test)


# In[ ]:


y_pred_proba0 = model.predict_proba(X_train)


# In[ ]:


y_pred_proba1 = model.predict_proba(X_test)


# In[ ]:


roc_auc_score(y_train,y_pred_proba0[:,1])


# In[ ]:


roc_auc_score(y_test,y_pred_proba1[:,1])


# In[ ]:


confusion_matrix(y_test,y_predict)


# In[ ]:


accuracy_score(y_test,y_predict)


# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(model,df[Best_features].sample(frac=1,random_state=0),df['poisonous'].sample(frac=1,random_state=0),cv=5)


# In[ ]:


model = DecisionTreeClassifier()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


y_predict = model.predict(X_test)


# In[ ]:


y_pred_proba0 = model.predict_proba(X_train)


# In[ ]:


y_pred_proba1 = model.predict_proba(X_test)


# In[ ]:


roc_auc_score(y_train,y_pred_proba0[:,1])


# In[ ]:


roc_auc_score(y_test,y_pred_proba1[:,1])


# In[ ]:


confusion_matrix(y_test,y_predict)


# In[ ]:


accuracy_score(y_test,y_predict)


# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(model,df[Best_features].sample(frac=1,random_state=0),df['poisonous'].sample(frac=1,random_state=0),cv=5)


# In[ ]:


model = RandomForestClassifier()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


y_predict = model.predict(X_test)


# In[ ]:


y_pred_proba0 = model.predict_proba(X_train)


# In[ ]:


y_pred_proba1 = model.predict_proba(X_test)


# In[ ]:


roc_auc_score(y_train,y_pred_proba0[:,1])


# In[ ]:


roc_auc_score(y_test,y_pred_proba1[:,1])


# In[ ]:


confusion_matrix(y_test,y_predict)


# In[ ]:


accuracy_score(y_test,y_predict)


# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(model,df[Best_features].sample(frac=1,random_state=0),df['poisonous'].sample(frac=1,random_state=0),cv=5)


# #### I hope you learned some new things.

# #### Thank You...

# In[ ]:




