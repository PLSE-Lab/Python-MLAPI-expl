#!/usr/bin/env python
# coding: utf-8

# I started by developing ther **baseline model** using the features available in the training dataset- achieved score of around 3.98, implying that the features in training dataset were not very useful. Next, I did some **manual feature engineering by aggregating purchase amount** feature from historical and new merchant transactions getting an accuracy of roughly 3.8. 
# 
# Then I decided to try **featuretools for automated feature engineering and use Permutation Importance to select the relevant features.**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
from matplotlib import pyplot as plt
import gc
print(os.listdir("../input"))
gc.collect()


# In[ ]:


train=pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')


# Now comes the interesting part of running featuretools to get automated feature engineering. This library might take a little while to understand completely but an excellent and elegant solution. While using featuretools for automated feature engineering, below are some of the challenges I encountered:
# 
# 1. Most of the features are categorical in datasets but marked as numerical, so manually I had to change the variable type of many features to Id and Categorical to avoid getting rubbish features.
# 
# 2. The biggest problem by far at least in this competition was the large size of historical transactions file. Everytime, I was running out of RAM and the kernel died. Tackling this, I had to take some redundant steps but yes they worked.
#     * Performed featuretools feature engineering twice on train and test dataset that I could have done in 1 go by combining the two sets.
#     * Performed operations on each dataset and then immediately deleted them to retain precious memory.
#     * Turned off GPU to increase RAM from 14GB to 17GB.

# In[ ]:


import featuretools as ft

es= ft.EntitySet(id= 'train')

variable_types={'feature_1':ft.variable_types.Categorical,'feature_2':ft.variable_types.Categorical, 
                'feature_3':ft.variable_types.Categorical, 'target':ft.variable_types.Id}
es= es.entity_from_dataframe(entity_id='train',dataframe= train, index= 'card_id',variable_types= variable_types)

merchants= pd.read_csv('../input/merchants.csv')
merchants= merchants.drop_duplicates(['merchant_id'])
variable_types={'merchant_group_id':ft.variable_types.Id, 'merchant_category_id':ft.variable_types.Id, 
                'subsector_id':ft.variable_types.Categorical,
               'city_id':ft.variable_types.Id,'state_id':ft.variable_types.Id,'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='merchants', dataframe= merchants, index='merchant_id',variable_types= variable_types)
del merchants
gc.collect()

new_merchant_transactions= pd.read_csv('../input/new_merchant_transactions.csv')
new_merchant_transactions= new_merchant_transactions[(new_merchant_transactions['card_id']).isin(train['card_id'])]
variable_types={'card_id':ft.variable_types.Id, 'state_id':ft.variable_types.Id, 'city_id':ft.variable_types.Id,
               'merchant_category_id':ft.variable_types.Id,'merchant_id':ft.variable_types.Id,'subsector_id':ft.variable_types.Id,
              'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='new_merchant_transactions',dataframe= new_merchant_transactions, make_index= True,
                             index='new_merchants_id',time_index='purchase_date',variable_types= variable_types)
del new_merchant_transactions
gc.collect()

historical_transactions= pd.read_csv('../input/historical_transactions.csv')
historical_transactions= historical_transactions[(historical_transactions['card_id']).isin(train['card_id'])]
variable_types={'card_id':ft.variable_types.Id, 'state_id':ft.variable_types.Id, 'city_id':ft.variable_types.Id,
               'merchant_category_id':ft.variable_types.Id,'merchant_id':ft.variable_types.Id,'subsector_id':ft.variable_types.Id,
              'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='historical_transactions',dataframe= historical_transactions, make_index= True,index='historical_id',
                             time_index='purchase_date',variable_types= variable_types)
del historical_transactions
gc.collect()


# Adding relationships to the entity set is so natural in featuretools just like anybody would be dealing with databases.

# In[ ]:


r_cards_historical= ft.Relationship(es['train']['card_id'],es['historical_transactions']['card_id'])
es= es.add_relationship(r_cards_historical)

r_cards_new_merchants= ft.Relationship(es['train']['card_id'],es['new_merchant_transactions']['card_id'])
es= es.add_relationship(r_cards_new_merchants)

r_merchants_historical= ft.Relationship(es['merchants']['merchant_id'], es['historical_transactions']['merchant_id'])
es= es.add_relationship(r_merchants_historical)

r_merchants_new_merchants= ft.Relationship(es['merchants']['merchant_id'],es['new_merchant_transactions']['merchant_id'])
es= es.add_relationship(r_merchants_new_merchants)


# Comes the final part of feature engineering- DFS or Deep Feature Synthesis where different features get stacked up (that's mean the term deep comes from). I have kept the maximum depth as 1. With more depth, you get transformation of transformations that can be useful at times but are very hard to interpret, so avoiding that.

# In[ ]:


features_train, feature_names_train= ft.dfs(entityset= es, target_entity= 'train', max_depth= 1)

del es
gc.collect()


# In[ ]:


features_train


# Same featuretools engineering for the test set. As I mentioned above, due to memory constraints I had to do this twice by deleting the intermediary files in between.

# In[ ]:


import featuretools as ft

es= ft.EntitySet(id= 'test')

variable_types={'feature_1':ft.variable_types.Categorical,'feature_2':ft.variable_types.Categorical, 
                'feature_3':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='test',dataframe= test, index= 'card_id',variable_types= variable_types)

merchants= pd.read_csv('../input/merchants.csv')
merchants= merchants.drop_duplicates(['merchant_id'])
variable_types={'merchant_group_id':ft.variable_types.Id, 'merchant_category_id':ft.variable_types.Id, 
                'subsector_id':ft.variable_types.Categorical,
               'city_id':ft.variable_types.Id,'state_id':ft.variable_types.Id,'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='merchants', dataframe= merchants, index='merchant_id',variable_types= variable_types)
del merchants
gc.collect()

new_merchant_transactions= pd.read_csv('../input/new_merchant_transactions.csv')
new_merchant_transactions= new_merchant_transactions[(new_merchant_transactions['card_id']).isin(test['card_id'])]
variable_types={'card_id':ft.variable_types.Id, 'state_id':ft.variable_types.Id, 'city_id':ft.variable_types.Id,
               'merchant_category_id':ft.variable_types.Id,'merchant_id':ft.variable_types.Id,'subsector_id':ft.variable_types.Id,
              'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='new_merchant_transactions',dataframe= new_merchant_transactions, make_index= True,
                             index='new_merchants_id',time_index='purchase_date',variable_types= variable_types)
del new_merchant_transactions
gc.collect()

historical_transactions= pd.read_csv('../input/historical_transactions.csv')
historical_transactions= historical_transactions[(historical_transactions['card_id']).isin(test['card_id'])]
variable_types={'card_id':ft.variable_types.Id, 'state_id':ft.variable_types.Id, 'city_id':ft.variable_types.Id,
               'merchant_category_id':ft.variable_types.Id,'merchant_id':ft.variable_types.Id,'subsector_id':ft.variable_types.Id,
              'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='historical_transactions',dataframe= historical_transactions, make_index= True,index='historical_id',
                             time_index='purchase_date',variable_types= variable_types)
del historical_transactions
gc.collect()


# In[ ]:


r_cards_historical= ft.Relationship(es['test']['card_id'],es['historical_transactions']['card_id'])
es= es.add_relationship(r_cards_historical)

r_cards_new_merchants= ft.Relationship(es['test']['card_id'],es['new_merchant_transactions']['card_id'])
es= es.add_relationship(r_cards_new_merchants)

r_merchants_historical= ft.Relationship(es['merchants']['merchant_id'], es['historical_transactions']['merchant_id'])
es= es.add_relationship(r_merchants_historical)

r_merchants_new_merchants= ft.Relationship(es['merchants']['merchant_id'],es['new_merchant_transactions']['merchant_id'])
es= es.add_relationship(r_merchants_new_merchants)


# In[ ]:


features_test, feature_names_test= ft.dfs(entityset= es, target_entity= 'test', max_depth= 1)

del es
gc.collect()


# These are some features that contained string data, so label encoding them below. A better approach could be to label encode all the features before performing, will make that optimization going forward.

# In[ ]:


columns_categorical=['MODE(new_merchant_transactions.authorized_flag)', 'MODE(new_merchant_transactions.category_1)', 
              'MODE(new_merchant_transactions.category_3)', 'MODE(new_merchant_transactions.merchant_id)', 
              'MODE(historical_transactions.authorized_flag)', 'MODE(historical_transactions.category_1)', 
              'MODE(historical_transactions.category_3)', 'MODE(historical_transactions.merchant_id)']


# In[ ]:


Y=features_train['target']

features_train= features_train.drop(columns=['target'])
features_train= features_train.fillna(method= 'bfill')
features_test= features_test.fillna(method= 'bfill')

X= features_train.copy()


# In[ ]:


features_train.to_csv('features_train.csv',index= False)
features_test.to_csv('features_test.csv',index= False)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

temp= features_train.append(features_test)
for i in columns_categorical:
    le= LabelEncoder()
    temp[i]= temp[i].astype('str')
    X[i]= X[i].astype('str')
    features_test[i]= features_test[i].astype('str')
    le.fit(temp[i])
    X[i]= le.transform(X[i])
    features_test[i]= le.transform(features_test[i])


# In[ ]:


from sklearn.model_selection import train_test_split

xtrain,xval,ytrain,yval= train_test_split(X,Y,test_size=0.1)


# Using Light GBM, generally I prefer XGBoost but they provide similar accuracy.

# In[ ]:


from sklearn.metrics import mean_squared_error
import lightgbm as lgb

model= lgb.LGBMRegressor()
model.fit(xtrain,ytrain)

print("RMSE of Validation Data using Light GBM: %.2f" % math.sqrt(mean_squared_error(yval,model.predict(xval))))


# In[ ]:


fig, ax= plt.subplots(figsize=(14,14))
lgb.plot_importance(model, ax= ax)
plt.show()


# To select features, I am using the 3 methods:
# 
# 1. In-built feature importance of Light GBM
# 2. Permutation Importance
# 3. SHAP values
# 
# I prefer to perform feature selection based on Permutation Importance simply because that's the best I understand. Possible, that Feature importances and SHAP values can give better results but I do not understand their maths very well (especially the SHAP values). I will try learning them better.
# 
# Permutation Importance is easy to comprehend and a natural way to remove useless features- if you randomly shuffle a feature and it doesn't reduce your accuracy, then that feature is not a good indicator. If someone had to do all this manually, this is the way to go about it.

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance
 
perm = PermutationImportance(model).fit(xval,yval)
eli5.show_weights(perm, feature_names = xval.columns.tolist())


# In[ ]:


import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(xval)
shap.summary_plot(shap_values, xval)


# This is simply selecting the features based on a threshold from Permutation Importance. One problem that I am facing (not able to resolve completely)- due to the fact that automtaed feature engineering has generated so many junk features that any single feature is having very less impact on overall accuracy. So, even with a low threshold of 0.001, I remove almost 70% of the features generated from featuretools.

# In[ ]:


from sklearn.feature_selection import SelectFromModel

submission= pd.read_csv('../input/sample_submission.csv')
features_test= features_test.fillna(0)
features_test= features_test.reindex(index= submission['card_id'])

sel= SelectFromModel(perm, threshold= 0.002, prefit= True)
X= sel.transform(X)
features_test= sel.transform(features_test)

print("Modified shape:", X.shape)


# In[ ]:


model_1= lgb.LGBMRegressor(learning_rate= 0.1, gamma=1)
model_1.fit(X,Y)


# In[ ]:


ypred= model_1.predict(features_test)

submission['target']=ypred
submission.to_csv('submission.csv', index= False)


# To increase accuracy further, I see two options (please let me know if you have other ideas as well):
# 
# 1. Perform manual feature engineering based on domain knowledge. One, I need to thoroughly understand how cards business work but then lot of features are anonymized and it is difficult to comprehend them.
# 
# 2. Hyper tune the paramters in Light GBM/ XGBoost model.
