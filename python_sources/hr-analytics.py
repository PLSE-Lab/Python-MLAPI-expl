#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
data = pd.read_csv("../input/wns-inno/train_LZdllcl.csv")


# In[ ]:


data.head()


# In[ ]:


#import pixiedust


# In[ ]:


#display(data)


# In[ ]:


data = data.drop(['region','employee_id'],axis =1)


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.isna().sum()


# In[ ]:


data["education"].fillna( method ='ffill', inplace = True)
data = data.fillna(data.mean())
data.isna().sum()


# In[ ]:


#data = data.dropna()


# In[ ]:


data.shape


# In[ ]:


data.dtypes


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[ ]:


# Categorical boolean mask
categorical_feature_mask = data.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = data.columns[categorical_feature_mask].tolist()
categorical_cols


# In[ ]:


#['department', 'region', 'education', 'gender', 'recruitment_channel']
print(data['recruitment_channel'].nunique())
print(data['department'].nunique())
print(data['education'].nunique())
print(data['gender'].nunique())


# In[ ]:


print(data['department'].unique())
print(data['recruitment_channel'].unique())
print(data['education'].unique())
print(data['gender'].unique())


# In[ ]:


# instantiate labelencoder object
le = LabelEncoder()
# apply le on column gender
data['gender'] = le.fit_transform(data['gender'])
data.head(2)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
# instantiate OneHotEncoder
features = ['department', 'education', 'recruitment_channel']
ohe = OneHotEncoder(categorical_features = features, sparse=False ) 
# categorical_features = boolean mask for categorical columns
# sparse = False output an array not sparse matrix


# In[ ]:


# apply OneHotEncoder on categorical feature columns
#X_ohe = ohe.fit_transform(data) # It returns an numpy array
ohe = OneHotEncoder(sparse=False)
X_ohe = ohe.fit_transform(data[['department', 'education', 'recruitment_channel']])


# In[ ]:


X_ohe.shape


# In[ ]:


type(X_ohe)


# In[ ]:


X_ohe


# In[ ]:


df= pd.get_dummies(data['department'], prefix=['department'],drop_first=True)
df1 =  pd.get_dummies(data['education'], prefix=['education'],drop_first=True)
df2 =  pd.get_dummies(data['recruitment_channel'], prefix=['RC'],drop_first=True)
data = pd.concat([data, df, df1,df2],axis=1)
data = data.drop(['department', 'education', 'recruitment_channel'],axis=1)


# In[ ]:


data.head(2)


# In[ ]:


data.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = data.drop(['is_promoted'],axis=1)
Y = data['is_promoted']


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state = 42, test_size = 0.2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X_train,Y_train)


# In[ ]:


predict1 = logit.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, roc_auc_score, roc_curve
print(classification_report(Y_test,predict1))


# In[ ]:


fpr,tpr,threshold = roc_curve(Y_test,logit.predict_proba(X_test)[:,1])


# In[ ]:


logit_roc_auc_1 = roc_auc_score(Y_test,logit.predict(X_test))
logit_roc_auc_1


# In[ ]:


from sklearn.metrics import f1_score
# f1 score
score = f1_score(predict1, Y_test)
score


# In[ ]:


import matplotlib.pyplot as plt 
plt.plot(fpr,tpr,label = 'Logistic Regression (Sensitivity = %0.3f)'%logit_roc_auc_1)
plt.plot([0,1],[0,1],'r--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = "lower right")


# We can see that recall for class 1 is very bad. Thus our model is not good.

# In[ ]:


pd.value_counts(data['is_promoted'])


# There is class imbalance in our data. We will use SMOTE or ADASYN for overcoming class imbalance problem.

# In[ ]:


from imblearn.over_sampling import SMOTE
from collections import Counter
# applying SMOTE to our data and checking the class counts
X_resampled, y_resampled = SMOTE().fit_resample(X, Y)
print(sorted(Counter(y_resampled).items()))


# In[ ]:


X1_train,X1_test,Y1_train,Y1_test = train_test_split(X_resampled,y_resampled,test_size = 0.2, random_state =2 )


# In[ ]:


logit.fit(X1_train, Y1_train)


# In[ ]:


pred1 = logit.predict(X1_test)


# In[ ]:


print(classification_report(Y1_test, pred1))


# In[ ]:


from sklearn.metrics import f1_score
# f1 score
score = f1_score(pred1, Y1_test)
score


# In[ ]:


from imblearn.over_sampling import ADASYN
from collections import Counter
# applying SMOTE to our data and checking the class counts
X_resampled1, y_resampled1 = ADASYN().fit_resample(X, Y)
print(sorted(Counter(y_resampled1).items()))


# In[ ]:


X2_train,X2_test,Y2_train,Y2_test = train_test_split(X_resampled1,y_resampled1,test_size = 0.2, random_state =2 )


# In[ ]:


logit.fit(X2_train, Y2_train)


# In[ ]:


pred2 = logit.predict(X2_test)


# In[ ]:


print(classification_report(Y2_test, pred2))


# In[ ]:


from sklearn.metrics import f1_score
# f1 score
score = f1_score(pred2, Y2_test)
score


# In[ ]:


test = pd.read_csv("../input/wns-inno/test_2umaH9m.csv")


# In[ ]:


test.isna().sum()


# In[ ]:


test1 = pd.read_csv("../input/wns-inno/test_2umaH9m.csv")


# In[ ]:


test.head()


# In[ ]:


test.keys()


# In[ ]:


test = test.drop(['region','employee_id'],axis =1)
test["education"].fillna( method ='ffill', inplace = True)
test = test.fillna(test.mean())
test.isna().sum()


# In[ ]:


#['department', 'region', 'education', 'gender', 'recruitment_channel']
print(test['recruitment_channel'].nunique())
print(test['department'].nunique())
print(test['education'].nunique())
print(test['gender'].nunique())


# In[ ]:


# instantiate labelencoder object
le = LabelEncoder()
# apply le on column gender
test['gender'] = le.fit_transform(test['gender'])
test.head(2)


# In[ ]:


df4= pd.get_dummies(test['department'], prefix=['department'],drop_first=True)
df5 =  pd.get_dummies(test['education'], prefix=['education'],drop_first=True)
df6 =  pd.get_dummies(test['recruitment_channel'], prefix=['RC'],drop_first=True)


# In[ ]:


test = pd.concat([test, df4, df5,df6],axis=1)
test = test.drop(['department','education', 'recruitment_channel'],axis=1)


# In[ ]:


test.shape


# In[ ]:


test_pred = logit.predict(test)
len(test_pred)


# In[ ]:


import numpy as np
employee_id=np.array(test1['employee_id'])
len(employee_id)


# In[ ]:


submission = pd.DataFrame({'employee_id': employee_id, 'is_promoted': list(test_pred)}, columns=['employee_id', 'is_promoted'])


# In[ ]:


submission.head()


# In[ ]:


submission.shape


# In[ ]:


# Install `XlsxWriter` 
#!pip install XlsxWriter


# In[ ]:


# Specify a writer
#writer = pd.ExcelWriter('submission.xlsx', engine='xlsxwriter')

# Write your DataFrame to a file     
#submission.to_excel(writer, 'Sheet1')

# Save the result 
#writer.save()


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random_forest1 = RandomForestClassifier( max_depth=15)
random_forest1.fit(X2_train, Y2_train)


# In[ ]:


pred_forest = random_forest1.predict(X2_test)
#X2_train,X2_test,Y2_train,Y2_test


# In[ ]:


print(classification_report(Y2_test, pred_forest))


# In[ ]:


score = f1_score(pred_forest, Y2_test)
score


# In[ ]:


test_pred_forest = random_forest1.predict(test)
len(test_pred_forest)


# In[ ]:


submission = pd.DataFrame({'employee_id': employee_id, 'is_promoted': list(test_pred_forest)}, columns=['employee_id', 'is_promoted'])


# In[ ]:


submission.head()


# In[ ]:


# Specify a writer
#writer = pd.ExcelWriter('submission.xlsx', engine='xlsxwriter')

# Write your DataFrame to a file     
#submission.to_excel(writer, 'Sheet1')

# Save the result 
#writer.save()


# In[ ]:




