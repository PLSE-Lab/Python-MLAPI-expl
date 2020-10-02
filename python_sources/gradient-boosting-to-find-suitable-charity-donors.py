#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
print(os.listdir("../input"))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer


# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


data=pd.read_csv('../input/census.csv')


# In[ ]:


data.head(5)


# In[ ]:


y=data['income']


# In[ ]:


features=data.drop(columns=['income']) 


# In[ ]:


income=y.map({'<=50K': 0, '>50K':1})


# In[ ]:


#checking class imbalance 
print(sum(income==0)/len(income))


# In[ ]:


features[['capital-gain','capital-loss']].hist()


# In[ ]:


#applying log transformations to capital-gain and capital-loss
features_log = pd.DataFrame(data = features)
features_log[['capital-gain','capital-loss']] = features[['capital-gain','capital-loss']].apply(lambda x: np.log(x + 1))


# In[ ]:


features_log[['capital-gain','capital-loss']].hist()


# In[ ]:


#normalizing numerical features 
scaler = MinMaxScaler()
numerical_features=['age','education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_log_scaled=pd.DataFrame(data=features_log)
features_log_scaled[numerical_features]=scaler.fit_transform(features_log[numerical_features])
features_log_scaled.head(100)


# In[ ]:


#one hot encoding all the categorical variables
features_final = pd.get_dummies(features_log_scaled)
#printing the number of features after encoding
encoded = list(features_final.columns)
print("{} total features after encoding.".format(len(encoded)))


# In[ ]:


features_final.head(5)


# In[ ]:


X_train, X_test, y_train, y_test= train_test_split(features_final, income, test_size=0.2, random_state=21)

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# I will be using Gradient Boosting classifier for this dataset. 

# In[ ]:


model=GradientBoostingClassifier()
parameters = {'n_estimators':[300,400,500],'min_samples_split':[4],'max_depth':[3]}
scorer = make_scorer(roc_auc_score)
gridcv_object = GridSearchCV(model, parameters, scoring=scorer)
gridcv_fit = gridcv_object.fit(X_train, y_train)
best_model = gridcv_fit.best_estimator_
predictions = (model.fit(X_train, y_train)).predict(X_test)
best_predictions=best_model.predict(X_test)


print("Unoptimized model\n")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("Area under curve on testing data: {:.4f}".format(roc_auc_score(y_test, predictions)))
print("\nOptimized Model\n")
print("Accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Area under curve on the testing data: {:.4f}".format(roc_auc_score(y_test, best_predictions)))


# In[ ]:


probs_train = best_model.predict_proba(X_train)[:, 1]
probs_test = best_model.predict_proba(X_test)[:, 1]
print("score train: {}".format(roc_auc_score(y_train, probs_train)))
print("score test: {}".format(roc_auc_score(y_test, probs_test)))


# In[ ]:


test = pd.read_csv("../input/test_census.csv")
for cat in test:
    first=test[cat][0]
    test[cat].fillna(first, inplace=True)


# In[ ]:


features_log_test = pd.DataFrame(data = test)
features_log_test[['capital-gain','capital-loss']] = features_log_test[['capital-gain','capital-loss']].apply(lambda x: np.log(x + 1))
scaler_test = MinMaxScaler()
features_log_scaled_test=pd.DataFrame(data=features_log_test)
features_log_scaled_test[numerical_features]=scaler.fit_transform(features_log_test[numerical_features])
features_final_test = pd.get_dummies(features_log_scaled_test)


# In[ ]:


features_final_test_1=features_final_test.drop('Unnamed: 0',1)


# In[ ]:


test['id'] = test.iloc[:,0] 
test['income'] = best_model.predict_proba(features_final_test_1)[:, 1]


# In[ ]:


test[['id', 'income']].to_csv("submission.csv", index=False)


# In[ ]:




