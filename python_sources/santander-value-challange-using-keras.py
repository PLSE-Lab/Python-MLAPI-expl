#!/usr/bin/env python
# coding: utf-8

# # Santander Value Prediction Challenge

# In this competition, Santander Group is asking Kagglers to help them identify the value of transactions for each potential customer. This is a first step that Santander needs to nail in order to personalize their services at scale.

# ## 1. Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


# ## 2.Import dataset

# In[ ]:


train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')

print("Training Set:")
n_train_data=len(train_df)
n_train_features=train_df.shape[1]

print("Number of Records: {}".format(n_train_data))
print("Number of Features:{}".format(n_train_features))

print ("\nTesting set:")
n_test_data  = len(test_df)
n_test_features = test_df.shape[1]
print ("Number of Records: {}".format(n_test_data))
print ("Number of Features: {}".format(n_test_features))


# As we see in the data set, the training set have 4993 features (columns) but only have 445559 records (rows). Test set has 49342 records which is much more than the training set.

# ## 3.Understand the data

# In[ ]:


train_df.head(10)


# Here I list the top 10 rows of traning set, noticed that there are many zero values in the data and the columns' name are anonymized.

# In[ ]:


dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# Most columns in the data set are integer and float number but there is only one column which has object data. This column is the ID column which we is not useful for us.

# ### 3.1 Check null values

# In[ ]:


print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))
print("\nTotal Test Features with NaN Values = " + str(test_df.columns[test_df.isnull().sum() != 0].size))


# There is no NaN values in both training and testing data.

# ### 3.2 Check columns with constant data
# 

# In[ ]:


unique_df = train_df.nunique().reset_index()  ## check number of distinct observations in each column
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1] ## if the number of distinct observation in each column is 1 then this column has constant value


# In[ ]:


constant_df.shape


# There are 256 columns have constant value so we can get ride of these columns

# In[ ]:


train_df.drop(constant_df.col_name.tolist(),axis=1,inplace=True) ## Drop 256 columns with constant values


# In[ ]:


train_df.shape


# ### 3.3 Check and Remove Duplicate Columns

# In[ ]:


train_df=train_df.T.drop_duplicates().T


# In[ ]:


train_df.shape


# There are 6 duplicate columns and remove 5 of them

# ### 3.4 Split data into training and target data

# In[ ]:


X_train=train_df.drop(['ID','target'],axis=1)
y_train=np.log1p(train_df['target'].values.astype(int))

X_test = test_df.drop(constant_df.col_name.tolist() + ["ID"], axis=1)


# In[ ]:


X_test.shape


# ## 4.Feature Selection
# 

# In this section I will use three methods to determine importance of features.

# In[ ]:


feat_labels=list(X_train)


# Create feature lables from training set

# ### 4.1 Feature importance using Gradient Boosting Regressor

# In[ ]:


clf_gb = GradientBoostingRegressor(random_state = 42)
clf_gb.fit(X_train, y_train)
print(clf_gb)


# In[ ]:


feat_importances = pd.Series(clf_gb.feature_importances_, index=feat_labels)
feat_importances = feat_importances.nlargest(25)
plt.figure(figsize=(16,15))
feat_importances.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()


# Here I plot the top 25 features gradient boosting regressor. Below list the name of top 10 features and there importance.

# In[ ]:


print(pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(10))


# In[ ]:


pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(200).sum()


# >**The top 200 features will cover more than 90% of the total importance.**

# ### 4.2 Feature importance using Random Froest Regressor

# In[ ]:


clf_rf = RandomForestRegressor(random_state = 42)
clf_rf.fit(X_train, y_train)
print(clf_rf)


# In[ ]:


feat_importances_rf = pd.Series(clf_rf.feature_importances_, index=feat_labels)
feat_importances_rf = feat_importances_rf.nlargest(25)
plt.figure(figsize=(16,15))
feat_importances_rf.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:


print(pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(10))


# In[ ]:


pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(200).sum()


# >**Top 200 features only cover about 60% of total importance using random forest.**

# In[ ]:


s1 = pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(10).index
s2 = pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(10).index


# In[ ]:


common_features = pd.Series(list(set(s1).intersection(set(s2)))).values

print(common_features)


# Compare two methods, for top 10 features there are 6 features in common (list above).

# In[ ]:


pd.Series(clf_gb.feature_importances_, index=X_train.columns)[common_features].sum()


# In[ ]:


pd.Series(clf_rf.feature_importances_, index=X_train.columns)[common_features].sum()


# ### 4.3 PCA transformation

# In[ ]:


from sklearn.decomposition import PCA
model_PCA=PCA(n_components=3)
model_PCA.fit(X_train)


# In[ ]:


transformed=model_PCA.transform(X_train)


# In[ ]:


print(transformed)


# ## 5.Modeling

# First I will build a NN according to 6 main features mentioned in the previous section.

# In[ ]:


common_features=np.append(common_features,'target')


# In[ ]:


common_features


# In[ ]:


train_df1=train_df[common_features]


# In[ ]:


X_train1=train_df1.drop(['target'],axis=1)
y_train1=np.log1p(train_df1['target'].values.astype(int))


# In[ ]:


y_train1


# In[ ]:


from sklearn.model_selection import train_test_split
X_train_1, X_test_1,y_train_1,y_test_1=train_test_split(X_train1,y_train1,test_size=0.4,random_state=0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_1=sc.fit_transform(X_train_1)
X_test_1=sc.transform(X_test_1)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


# In[ ]:


def baseline_model():
    model=Sequential()
    model.add(Dense(units=6,kernel_initializer='normal',activation='relu',input_dim=6))
    model.add(Dense(units=3,kernel_initializer='normal',activation='relu'))
    model.add(Dense(units=1,kernel_initializer='normal'))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model
    
    
   # model.fit(X_train_1,y_train_1,batch_size=10,epochs=100)


# In[ ]:


seed=7
np.random.seed(seed)
estimator=KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=5,verbose=1)


# In[ ]:


kfold=KFold(n_splits=3,random_state=seed)
results=cross_val_score(estimator,X_train1,y_train1,cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# In[ ]:


np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(pipeline, X_train1,y_train1, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# In[ ]:


model=Sequential()
model.add(Dense(units=6,kernel_initializer='normal',activation='relu',input_dim=6))
model.add(Dense(units=3,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=1,kernel_initializer='normal'))
model.compile(optimizer='adam',loss='mean_squared_error')
prediction=model.predict(X_train1)


# In[ ]:


prediction


# In[ ]:


y_train1


# In[ ]:


plt.plot(prediction)
plt.plot(y_train1)
plt.show()


# Compare between prediction (blue) and actual data (orange)

# In[ ]:


# result = pd.DataFrame({'ID':ID,'target':submissions})
sub = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


sub.shape


# In[ ]:


test_df.shape


# In[ ]:




