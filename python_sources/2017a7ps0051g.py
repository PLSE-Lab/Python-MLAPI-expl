#!/usr/bin/env python
# coding: utf-8

# # Evaluative Lab 1

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Exploration

# In[ ]:


df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.describe(include='object')


# ## Checking Null Values

# In[ ]:


df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)
df_dtype_nunique.columns = ["dtype","unique"]
df_dtype_nunique


# In[ ]:


missing_count = df.isnull().sum()
missing_count[missing_count>0]


# In[ ]:


df.fillna(value=df.mean(),inplace=True)


# In[ ]:


df.isnull().any().any()


# ### Log transform features

# In[ ]:


#df['feature8'] = df['feature8'] + 1  ## shifting feature8's min from 0 to 1


# In[ ]:


# for i in range(1,12):
#        df['feature'+str(i)] = np.log(df['feature'+str(i)])


# ## Data Visualization

# In[ ]:


df.corr()


# In[ ]:


corr_mat=df.corr(method='pearson')
plt.figure(figsize=(15,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')


# In[ ]:


sns.regplot(x=df['feature6'],y=df['rating'],data=df)


# In[ ]:


sns.boxplot(x=df['type'], y=df['rating'], data = df)


# In[ ]:


sns.barplot(x=df['rating'].value_counts().index, y=df['rating'].value_counts())


# ### One hot encoding of 'type' feature

# In[ ]:


df = pd.get_dummies(data=df,columns=['type'])


# ### Splitting the dataset before upsampling

# In[ ]:


df_train = df.sample(frac=0.95,random_state=100) ## 1 for final submission
df_test = df.drop(df_train.index)


# ## Data Upsampling

# In[ ]:


from sklearn.utils import resample


# In[ ]:


df_train['rating'].value_counts()


# In[ ]:


df_test['rating'].value_counts()


# In[ ]:


df_majority = df_train[df_train['rating'].isin([2,3])]
df_minority4 = df_train[df_train['rating']==4]
df_minority15 = df_train[df_train['rating'].isin([1,5])]
df_minority06 = df_train[df_train['rating'].isin([0,6])]


# In[ ]:


df_minority_upsampled4 = resample(df_minority4, replace=True, n_samples=1000, random_state=1) 


# In[ ]:


df_minority_upsampled15 = resample(df_minority15, replace = True, n_samples=1200, random_state=1)


# In[ ]:


df_minority_upsampled06 = resample(df_minority06, replace = True, n_samples=500, random_state=1)


# In[ ]:


df_upsampled = pd.concat([df_majority,df_minority_upsampled06,df_minority_upsampled15,df_minority_upsampled4])


# In[ ]:


df_upsampled['rating'].value_counts()


# In[ ]:


sns.barplot(x=df_upsampled['rating'].value_counts().index, y=df_upsampled['rating'].value_counts())


# ### Turn upsampling on/off

# In[ ]:


#df_train = df_upsampled #Comment to turn off


# ## Feature selection and split

# In[ ]:


#X = data.drop(['id', 'feature1', 'feature2', 'feature4','feature8', 'feature9', 'type', 'feature10', 'feature11', 'rating'],axis=1)

X_train = df_train.drop(['id','rating'],axis=1)
y_train = df_train['rating']
X_val = df_test.drop(['id','rating'],axis=1)
y_val = df_test['rating']


# ## Feature scaling

# In[ ]:


from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# In[ ]:


X_val = scaler.transform(X_val)


# ## Visualize variance in data

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# ## Import Models

# In[ ]:


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error


# ### Final Model used - ExtraTreesRegressor with 500 estimators and maxdepth 50

# In[ ]:


# reg1 = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1]).fit(X_train,y_train)
# reg2 = LinearRegression().fit(X_train,y_train)
# reg3 = Ridge().fit(X_train,y_train)
# reg4 = Lasso().fit(X_train,y_train)
# reg5 = ElasticNet().fit(X_train,y_train)
# reg6 = BayesianRidge().fit(X_train,y_train)
# reg7 = KNeighborsRegressor().fit(X_train,y_train)
# reg8 = DecisionTreeRegressor().fit(X_train,y_train)
# reg9 = GradientBoostingClassifier().fit(X_train,y_train)
# reg10 = GradientBoostingRegressor().fit(X_train,y_train)
# reg11 = AdaBoostRegressor().fit(X_train,y_train)
# reg12 = LogisticRegression().fit(X_train,y_train)
# reg13 = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
# reg14 = RandomForestRegressor(n_estimators=500).fit(X_train,y_train)
# reg15 = MLPClassifier().fit(X_train,y_train)
# reg16 = MLPRegressor().fit(X_train,y_train)
reg17 = ExtraTreesRegressor(n_estimators=500, max_depth=50).fit(X_train,y_train)
# reg18 = ExtraTreesClassifier(n_estimators=900,min_samples_leaf=1,max_depth=None).fit(X_train,y_train)
# reg19 = GaussianNB().fit(X_train,y_train)
# reg20 = GaussianProcessRegressor.fit(X_train,y_train)


# ### Train RMSEs

# In[ ]:


y_pred1 = reg17.predict(X_train)
#y_pred2 = reg18.predict(X_train)
#y_pred3 = reg19.predict(X_train)

rmse1 = np.sqrt(mean_squared_error(y_pred1,y_train))
#rmse2 = np.sqrt(mean_squared_error(y_pred2,y_train))
#rmse3 = np.sqrt(mean_squared_error(y_pred3,y_train))


# In[ ]:


print(rmse1,sep="\n") #train rmse


# ### Validation RMSEs

# In[ ]:


y_pred1 = reg17.predict(X_val)
#y_pred2 = reg2.predict(X_val)
#y_pred3 = reg3.predict(X_val)

rmse1 = np.sqrt(mean_squared_error(np.round(y_pred1),y_val))
#rmse2 = np.sqrt(mean_squared_error(np.round(y_pred2),y_val))
#rmse3 = np.sqrt(mean_squared_error(np.round(y_pred3),y_val))


# In[ ]:


print(rmse1,sep="\n") #val rmse


# ## HyperParameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV
 
param_grid = {
    'min_samples_split':[4,2,3],
    'min_samples_leaf': [1,2,3],
    'max_depth': [50,100,None],
    'n_estimators': [500]
}

rf = ExtraTreesRegressor()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs=-1, verbose=2)


# In[ ]:


grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_


# ## Ensemble with Voting Regressor

# In[ ]:


reg1 = ExtraTreesRegressor(min_samples_leaf=1,min_samples_split=2,n_estimators=1000).fit(X_train,y_train)
reg2 = ExtraTreesRegressor(n_estimators=350,max_depth=100).fit(X_train,y_train)
reg3 = ExtraTreesRegressor(n_estimators=515,max_depth=51).fit(X_train,y_train)


# In[ ]:


from sklearn.ensemble import VotingRegressor


# In[ ]:


regGod = VotingRegressor(estimators=[('et1', reg1) , ('et2', reg2), ('et3', reg3)])


# In[ ]:


regGod.fit(X_train,y_train)


# In[ ]:


y_god = regGod.predict(X_val)
rmse_god = (np.sqrt(mean_squared_error(np.round(y_god),y_val)))
print(rmse_god) # Val rmse


# ## Analysing Predictions

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(accuracy_score(y_val,np.round(y_god)))
print(confusion_matrix(y_val,np.round(y_god)))
print(classification_report(y_val,np.round(y_god))) 


# ### Customized neural network training

# In[ ]:


'''
Didn't work well

reg20 = MLPClassifier(warm_start=True, learning_rate_init=0.00001, hidden_layer_sizes=[50, 50])
import pickle
rmses = []
for i in range(2500):
    reg20.fit(X_train, y_train)
    y_pred = reg20.predict(X_val)
    rmse20 = np.sqrt(mean_squared_error(np.round(y_pred),y_val))
    print(i, rmse20)
    if i>0 and rmse20<np.min(rmses):
        pickle.dump(reg20, open("best_mlp_model.p", "wb"))
        print("Saving best model in epoch", i, ". RMSE =", rmse20)
    rmses.append(rmse20)
plt.plot(np.arange(0, 2500), rmses)
'''


# ## Visualizing validation predictions

# In[ ]:


df1 = pd.DataFrame({'Actual': y_val, 'Predicted': np.round(y_pred1)})
df1head = df1.head(20)


# In[ ]:


df1head.plot(kind='bar',figsize=(10,8))
plt.show()


# ## Test set predictions

# In[ ]:


df_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')


# In[ ]:


df_test.describe()


# In[ ]:


missing_count = df_test.isnull().sum()
missing_count[missing_count>0]


# In[ ]:


df_test.fillna(value=df_test.mean(),inplace=True)


# ### Log transform 

# In[ ]:


#df_test['feature8'] = df_test['feature8'] + 1


# In[ ]:


# for i in range(1,12):
#        df_test['feature'+str(i)] = np.log(df_test['feature'+str(i)])


# ### One hot encoding

# In[ ]:


df_test = pd.get_dummies(data=df_test,columns=['type'])


# In[ ]:


X_test = df_test.drop(['id'],axis=1)


# In[ ]:


X_test_scaled = scaler.transform(X_test)


# In[ ]:


X_test_scaled[0] ##Checking same number of columns


# In[ ]:


X_train[0]


# ### Predicting

# In[ ]:


y_test = reg3.predict(X_test_scaled)


# In[ ]:


y_test


# In[ ]:


y_test = np.round(y_test)


# In[ ]:


y_test


# In[ ]:


np.unique(y_test)


# In[ ]:


submission = pd.concat([df_test['id'],pd.Series(y_test)],axis=1)
submission.columns = ['id','rating']
submission.head()


# In[ ]:


submission['rating'] = submission['rating'].astype(int)


# ## Exporting predictions

# In[ ]:


submission.to_csv('/kaggle/input/eval-lab-1-f464-v2/submission.csv',index=False)


# In[ ]:


submission['rating'].value_counts()/len(submission)


# In[ ]:


df['rating'].value_counts()/len(df)


# ## Saving best model

# In[ ]:


import pickle
pickle.dump(reg1, open("/kaggle/input/eval-lab-1-f464-v2/best_model.p", "wb")) #reg5 = ExtraTreesRegressor(n_estimators=500,max_depth=50).fit(X_train,y_train)

