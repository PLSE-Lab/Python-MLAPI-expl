#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


costa = pd.read_csv('../input/train.csv')

* Read the "train" file after loading the required libraries
# In[ ]:


costa.head()


# In[ ]:


costa['v2a1'].isnull().sum()


# * Checking one column for missing values. Now we need to check the extent of the missing values
# - 1st cut, looks like we can drop off the columns which have many missing values
# - Check for outliers
# - check for feature importance using selecKbest & Random forest
# - Feature engg./Feature selection is going to be very important as we need to signfificantly reduce the feature space

# In[ ]:


col_list = []
for feature in costa.columns: # Loop through all columns in the dataframe
    if costa[feature].isnull().sum() > 0: # Only apply for columns with categorical strings
        col_list.append(feature)


# In[ ]:


col_list


# In[ ]:


X = costa.drop(col_list, axis =1)


# * First cut analysis, we've dropped columns which have significantly high Null values

# # Visually going through EACH of the variables mentioned

# * We can see in the below box plot how the 'escolari' variable varies for different target variables
# and we see it seems to vary with each category. So can be considered in the 1st cut analysis

# In[ ]:


sns.boxplot(x="Target", y="escolari",data=costa, palette="coolwarm")


# * In the below groupby table, we can see that 'hogar_mayor' has different distribution with the target variable
# and can be considered as 1st cut variable
# 
# * Also - Select variables on common-sense basis for identifying poverty levels by household
# - Aggregate at household level (Since many variables are at a household level) and make prediction for each head of household - variables to aggregate can be number of individuals/by age in the household, total education etc

# In[ ]:


X.groupby('hogar_mayor')['Target'].value_counts()


# In[ ]:


sns.countplot(x='cielorazo',data=costa, hue ='Target')


# # Another plot (Countplot) to help us visualize the variable's possible link/impact on the end "Target"

# # 1st cut variables chosen, to check output
# - Using boxplot to see variation of feature with target variable
# - countplot
# - groupby & value counts for distribution across segments
# 
# Need to be aggregated at household level against the head of the household
# - rooms - >5rooms seem to fit target 4 - Already aggregated output
# - r4h3 - Total males in the household - Already aggregated output
# - r4m3 - Total females in the household - Already aggregated output
# - r4t3 - Total persons in the household - Already aggregated output
# - escolari - years of schooling (total for a household, or just use the squared metric later) = check for total household
#     - have 2 variables for a household
#         - total yrs of schooling for everyone in a household
#         - yrs of schooling of head of household
# - paredblolad - if predominant material on the outside wall is block or brick - ALREADY aggregated
# - cielorazo =1 if the house has ceiling - ALREADY aggregated
# - epared1, =1 if walls are bad - ALREADY aggregated
# - epared2, =1 if walls are regular - ALREADY aggregated
# - epared3, =1 if walls are good - ALREADY aggregated
# - etecho1, =1 if roof are bad - ALREADY aggregated
# - etecho2, =1 if roof are regular - ALREADY aggregated
# - etecho3, =1 if roof are good - ALREADY aggregated
# - eviv1, =1 if floor are bad - ALREADY aggregated
# - eviv2, =1 if floor are regular - ALREADY aggregated
# - eviv3, =1 if floor are good - ALREADY aggregated
# - hogar_adul - no. of adults in a household - ALREADY aggregated
# - hogar_mayor - # of individuals 65+ in the household - ALREADY aggregated
#     - SQBescolari - NEED NOT be considered as above we've considered escolari already

# # Now that we have out 1st cut variables 
# - Make a NEW dataframe with the above variables
#     - addl columns for schooling yrs to be created (In fact this also can be postponed and we work on just the schooling year of the head of the houshold)
#     - Select only rows for the head of household =1 (parentesco1, =1 if household head)

# In[ ]:


columns_list = ['rooms','r4h3', 'r4m3', 'r4t3', 'escolari','paredblolad', 'cielorazo', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2' , 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'hogar_adul', 'hogar_mayor', 'Target']


# In[ ]:


new = X[X['parentesco1']==1]


# In[ ]:


final = new[columns_list]


# In[ ]:


features = final[[i for i in list(final.columns) if i != 'Target']]


# In[ ]:


features.info()


# # Lets run our 1st basic algo and see !

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,final['Target'],
                                                    test_size=0.30)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,rfc_pred))


# In[ ]:


print(classification_report(y_test,rfc_pred))


# # Check model performance on the TEST dataset

# In[ ]:


costa_test = pd.read_csv('../input/test.csv')


# In[ ]:


col1_list = []
for feature in costa_test.columns: # Loop through all columns in the dataframe
    if costa_test[feature].isnull().sum() > 0: # Only apply for columns with categorical strings
        col1_list.append(feature)


# In[ ]:


col1_list


# In[ ]:


X1 = costa_test.drop(col1_list, axis =1)


# In[ ]:


newtest = X1[X1['parentesco1']==1]


# In[ ]:


newtest.info()


# In[ ]:


columns_list1 = ['Id', 'rooms','r4h3', 'r4m3', 'r4t3', 'escolari','paredblolad', 'cielorazo', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2' , 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'hogar_adul', 'hogar_mayor', 'idhogar']


# In[ ]:


final1 = newtest[columns_list1]


# In[ ]:


final2 = final1.drop('Id', axis =1)


# In[ ]:


final3 = final2.drop('idhogar', axis=1)


# In[ ]:


final1.reset_index(inplace = True)


# In[ ]:


final1.head()


# In[ ]:


rfc_pred1 = rfc.predict(final3)


# In[ ]:


len(rfc_pred1)


# In[ ]:


my_submission = pd.DataFrame({'Target': rfc_pred1})


# In[ ]:


final_submit = final1.join(my_submission)


# In[ ]:


final_submit.head()


# In[ ]:


my_submit1 = final_submit[['Target', 'idhogar']]


# In[ ]:


submission_base = X1[['Id', 'idhogar']].copy()


# In[ ]:


sample_submission = submission_base.merge(my_submit1, 
                                       on = 'idhogar',
                                       how = 'left').drop(columns = ['idhogar'])


# In[ ]:


median = sample_submission['Target'].median()


# In[ ]:


sample_submission['Target'].fillna(median, inplace= True)
sample_submission['Target'] = sample_submission['Target'].astype(int)


# In[ ]:


sample_submission.info()


# In[ ]:


#sample_submission.to_csv('sample_submission.csv', index = False)


# # We got a poor result of 0.318 on RF with the current set of variables.
# Lets try out another result with XGBoost algorithm

# In[ ]:


from xgboost.sklearn import XGBClassifier 


# In[ ]:


xclas = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=1500,
       n_jobs=1, nthread=None, objective='multi:softprob', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)


# In[ ]:


xclas.fit(X_train, y_train)  


# In[ ]:


xgb_pred1 = xclas.predict(final3)


# In[ ]:


len(xgb_pred1)


# In[ ]:


my_submission_xgb = pd.DataFrame({'Target': xgb_pred1})


# In[ ]:


final_submit_xgb = final1.join(my_submission_xgb)


# In[ ]:


my_submit_xgb = final_submit_xgb[['Target', 'idhogar']]


# In[ ]:


submission_base_xgb = X1[['Id', 'idhogar']].copy()


# In[ ]:


submission_xgb = submission_base_xgb.merge(my_submit_xgb, 
                                       on = 'idhogar',
                                       how = 'left').drop(columns = ['idhogar'])


# In[ ]:


median = submission_xgb['Target'].median()


# In[ ]:


submission_xgb['Target'].fillna(median, inplace= True)
submission_xgb['Target'] = submission_xgb['Target'].astype(int)


# In[ ]:


submission_xgb.to_csv('submission_xgb.csv', index = False)


# # Great - so we improved from .328 to .338 just by using a different algorithm. Now lets play around with the hyper-parameters in XGB

# # Making n_trees = 1000, has further improved the result to .357!

# # Lets keep tuning

# In[ ]:




