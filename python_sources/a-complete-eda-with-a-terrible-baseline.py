#!/usr/bin/env python
# coding: utf-8

# Documenting some of my thoughts while doing Exploratory Data Analysis for this competition. Hopefully it will inspire some ideas.
# 
# 
# ----------

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# **How many variables and what are they?**

# In[ ]:


train.shape


# In[ ]:


test.shape


# Interesting. Same number of observations for train and test data. Any missing values?

# In[ ]:


train.isnull().sum().value_counts()


# In[ ]:


import missingno as mn
mn.matrix(train)


# Looks like there are no missing values unless they are coded differently. Nice clean data. Curious what are they?

# In[ ]:


train.dtypes.value_counts()


# One float type variable. 8 Object - Categorical variable. 369 variables of integer type - guess there are quite a number of indicator variables inside. Let's check out.

# In[ ]:


summary = train.describe().transpose()
summary[(summary['min']==0)&(summary['max']==1)].head()


# In[ ]:


summary[(summary['min']==0)&(summary['max']==1)].shape


# So likely there are 356/369 indicator variables as they are integer type with only 0 and 1. Wondering what about the rest 369 - 356 = 13 variables. If we exclude ID, it will be 12.

# In[ ]:


summary[summary['std']==0]


# So the other 12 variables are all just zeros. This is very suprising as wondering why it is included in the dataset at all. Curious to see if they have similar distribution in the test dataset.

# In[ ]:


summary_test = test.describe().transpose()
summary_test[summary_test['std']==0]


# In[ ]:


summary.loc[['X257','X258','X295','X296','X369']]


# Note that test has a different set of variables with zero standard deviation but all of them have extremely low mean values showing that there are very few positive observations. Such differences between train set and test set suggests that we need to be very careful with those variables. On another note, this shows that the it is impossible to create a perfect similar training set and test set. We may be able to use the differences to our advantage in creative ways or at least should be cautious about it.

# Now let's look closely at each of the variable types. Start with float as there is only one of such type and it is the outcome variable.

# In[ ]:


plt.hist(train['y'],bins=50)


# Interesting bipolar? distribution with a bit long tail and outliers. May consider transformations. Let's look at categorical variables.

# **Categorical Variables**

# In[ ]:


cat = train.select_dtypes(include=['object'])
cat.head()


# In[ ]:


fig,ax = plt.subplots(8,2,figsize=(30,20))
fig.tight_layout()
for i in range(8):
    sns.countplot(cat.iloc[:,i],ax=ax[i,0])
    sns.boxplot(x=cat.iloc[:,i],y=train['y'],ax=ax[i,1])


# Some variables have a lot of categories with few observations. 
# Some quick observations:
# X1 - 'aa' has very few values but very high y value. If it is outlier,likely to impact the predictions. 
# X4 - majority lies in 'd',a,b,c almost have zero counts.
# X8 - Relatively even distribution of both counts of each category and associated y values compared with other categorial variables.
# Consider dimention reduction to reduce number of groups - either based on simple rules such as group those with few observations together or use other tricks such as those in neural networks. Another possibility as discussed earlier is to try to compare with test data distribution to identify categories of interest for small observations.
# Next let's move on to those indicator variables

# **Indicator Variables**

# In[ ]:


ind = train.select_dtypes(include=['int64']).drop('ID',axis=1)
ind.head()


# Let's first see if some variables are correlated.

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(ind.corr())


# It seems some a bit messy. Let's see if PCA could help to reduce the dimentions a bit.

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA().fit(ind)
plt.plot(np.cumsum(pca.explained_variance_ratio_)[0:50])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# It seems that 20-50 components could explain around 80% to 90% of the variance, which is quite a bit reduction, 10%-20% of the original 368 variables. Therefore maybe worth trying.

# For curiosity, let's see if any of the variables form clusters.

# In[ ]:


pca = PCA(n_components=30)
pca.fit(ind)
x_pca = pca.transform(ind)
x_pca.shape


# In[ ]:


import time
from sklearn.manifold import TSNE
n_sne = len(x_pca)
time_start = time.time()
tsne = TSNE(n_components=2,verbose=1,perplexity=20,n_iter=300)
tsne_results = tsne.fit_transform(x_pca)


# In[ ]:


plt.scatter(tsne_results[:,0],tsne_results[:,1],c=train['y'],edgecolor='None',cmap=plt.cm.get_cmap('rainbow',30))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


# Nothing particular interesting other than there is a cluster with y less than 100. Maybe further tuning the variable could shed more insights. T-SNE components could be used as variables as well for initial variable selection

# Next, let's see the distribution of each of the 'indicator variables' and how they are correlated with y.

# In[ ]:


col = np.array(ind.columns[ind.std()==0])
ind2 = ind.drop(col,axis=1)
def myfunction(x):
    return np.corrcoef(x,train['y'])[0,1]
y_corr = ind2.apply(myfunction,axis=0)
x_mean = ind2.apply(np.mean,axis=0)
n = np.array(y_corr.index)


# In[ ]:


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
plt.scatter(x_mean,y_corr)
for i, txt in enumerate(n):
    ax.annotate(txt, (x_mean[i],y_corr[i]),fontsize=10)
plt.xlabel('Mean')
plt.ylabel('Correlation with Y')


# Intuitively, an indicator variable is potentially very helpful in predicting if it has certain variance and is highly associated with the outcome variable. Say if we narrow the mean to between 0.2 and 0.8 (we don't want too many 0s or 1s) and correlation to be above 0.2 or below -0.2 (significant positive or negative correlation), we find the following variables to be very interesting: X314,X261,X118,X275,X51,X311,X178,X250,X313,X127. We could potentially create interaction terms between them.

# **Feature importance - What do models say?**

# Other insights will probably come from modeling - eg Top important variables from Xgboost and Linear Models. Let's find out!

# First, let me clean up the house a bit - simple one hot encoding for categorical variables

# In[ ]:


cat = train.select_dtypes(['object'])
cat_one_hot = pd.get_dummies(cat)


# In[ ]:


cat.shape


# In[ ]:


cat_one_hot.shape


# In[ ]:


cat_one_hot.head()


# In[ ]:


cat_one_hot.mean().head()


# We created 195 indicator variables out of 8 categorical variables. Needless to say, this is a lot and will likely cause problems as we only have 4209 observations for training! As discussed previously, some of them have very few observations and could potentially be dropped/re-grouped. I will come back to this later. Right now let me see what the original variables without any feature engineering could give me.

# In[ ]:


train2 = pd.get_dummies(train).drop('ID',axis=1)
test2 = pd.get_dummies(test).drop('ID',axis=1)


# In[ ]:


train2.shape


# In[ ]:


test2.shape


# Interestingly, train set and test set and of different shapes. Let's find out why

# In[ ]:


train2.columns.values[~train2.columns.isin(test2.columns.values)]


# In[ ]:


test2.columns.values[~test2.columns.isin(train2.columns.values)]


# So it appears that those categories with very few observations causes the problem. A simple but maybe naive way is to delete those variables from both. We may look again later in future to see if there is anything we could exploit here.

# In[ ]:


train2 = train2[train2.columns.values[train2.columns.isin(test2.columns.values)]]
test2 = test2[test2.columns.values[test2.columns.isin(train2.columns.values)]]


# In[ ]:


y = train['y']
y.shape


# So now we have 553 indicator variables and 1 outcome variable of float type. Let's get ready for modeling. With some many columns and comparably few observations, algorithms with ability to select variables automatically should be considered first. Let's try two popular ones: Tree model - Xgboost;Linear model - Lasso/ElasticNet regression; 

# **Xgboost - " When in doubt, use Xgboost. "**

# First, let me set up a 10-fold cross-validation for Xgboost - I use random split assuming that the IDs does not contain any time information.

# In[ ]:


import xgboost as xgb
import sklearn
from sklearn import cross_validation
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV


# In[ ]:


### Very expensive grid search to find good parameters. Commented out for computational purpose.
#param_test1 = {
# 'max_depth':list(range(3,6,1)),
# 'n_estimators':list(range(500,3000,500))
# 'learning_rate':list(range(0.005,0.1,0.01))
#}
#gsearch1 = GridSearchCV(
#    estimator = xgb.XGBRegressor(learning_rate =0.005, n_estimators=1000, max_depth=5,min_child_weight=1,subsample=0.95,
#                                 objective= 'reg:linear',seed=27), 
#    param_grid = param_test1, scoring='r2',cv=10)
#gsearch1.fit(train2,y)
#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# CV result for the best combination is ard 0.58 with std 0.1 showing that sample matters a bit here. In below, we manually divide data into 90% training and 10% testing, for that particular split, r square is only ard 0.525.

# In[ ]:


X = train2
Y = y
seed = 1
test_size = 0.10 
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,Y,test_size=test_size,random_state=seed) 
y_mean = np.mean(y_train)


# In[ ]:


model = xgb.XGBRegressor(max_depth=4,learning_rate=0.005,n_estimators=2000,subsample=0.95,
                        objective='reg:linear')
model.fit(X_train,y_train)
print(model)
y_pred = model.predict(X_test)
r2 = r2_score(y_test,y_pred)
print(r2)


# Now let's take a look at the important variables.

# In[ ]:


feat_names = X_train.columns.values
feat_imp = model.feature_importances_
imp_map = pd.Series(feat_imp,index=feat_names)


# In[ ]:


imp_map.sort_values(ascending=False,inplace=True)
top30 = pd.Series(imp_map[0:30]).reset_index()
top30.columns = ['Variable','Importance']


# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(top30['Importance'],top30['Variable'])
plt.xlabel('Importance')


# Further investigation into each of those variables are preferred for potential feature engineering. Now let's move on to linear models for a while.

# **Linear Models**

# Let's see how linear models compare with tree-based models on the original features. In particular, linear models with automatic model selection - Ridge or Lasso. For simplicity, we explore elasti net which uses a combination of L1 and L2 regularizations.

# In[ ]:


from sklearn.linear_model import ElasticNet


# In[ ]:


#param_test2 = {
# 'l1_ratio':list(np.arange(0,1,0.3))
#}
#gsearch2 = GridSearchCV(
#    estimator = ElasticNet(l1_ratio=0,fit_intercept=True,max_iter=5000), 
#    param_grid = param_test2, scoring='r2',cv=10)
#gsearch2.fit(train2,y)
#gsearch2.grid_scores_, gsearch2.best_params_


# Cross validation shows that lasso is the best option giving a r2 ard 0.4 - much worse than xgboost suggesting that there are interactions between variables. Let's see what are the variables with largest (absolute value of) positive and negative coefficients!

# In[ ]:


model2 = ElasticNet(l1_ratio=0,max_iter=5000)
model2.fit(X_train,y_train)
print(model2)
y_pred = model2.predict(X_test)
r2 = r2_score(y_test,y_pred)
print(r2)


# In[ ]:


coef = model2.coef_
variable = train2.columns.values
var_coef = pd.DataFrame(pd.Series(coef,index=variable)).reset_index()
var_coef.columns = ['Variable','Coefficient']
var_coef['Abs_Coef'] = np.abs(var_coef['Coefficient'])
var_coef.sort_values('Abs_Coef',ascending=False,inplace=True)
var_coef['Sign'] = 'Neg'
var_coef['Sign'][var_coef['Coefficient']>0] = 'Pos'


# In[ ]:


lasso_top30 = var_coef[0:30]


# In[ ]:


plt.figure(figsize=(8,8))
sns.pointplot(lasso_top30['Abs_Coef'],lasso_top30['Variable'],hue=lasso_top30['Sign'],join=False)
plt.xlabel('Coefficient')


# Note that the top30 from xgboost and top30 from lasso are quite different. Let's take a detailed look.

# In[ ]:


top30['Variable'][top30['Variable'].isin(lasso_top30['Variable'])]


# Only 5 of the top variables overlap!

# **Concluding Thoughts**

# Personally I feel there are below take aways from this exploratory data analysis: 
# 
#  1. There are quite big number of indicator variables present in the dataset with some of them with very little variance. Given the size of the data, dimention reduction and variable selection is going to be key.
#  2. Baseline is not terribly bad - my simple Xgboost is ard 0.54 when submitting. The rank is terrible but expected given how little - almost 0 feature engineering I have done for the purpose of EDA. In reality, not sure how much this matters to the host - a gain from 0.54 to 0.57+? 
#  3. Seems this is going to be another feature engineering and model ensemble competition - hopefully this EDA will give you some ideas! This is especially true when we don't know what the variables mean so we sort of have to rely on EDA and modeling experiments to find out. 

# 
