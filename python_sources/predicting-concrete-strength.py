#!/usr/bin/env python
# coding: utf-8

# # To Predict the Strength of Concrete
# 
# 
# 
# ## **Context:**
# Concrete is the most important material in civil engineering. The concrete compressive strength is a highly nonlinear function of age and ingredients. These ingredients include cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggregate.
# 
# ## **Data Description:**
# The actual concrete compressive strength (MPa) for a given mixture under a
# specific age (days) was determined from laboratory. Data is in raw form (not scaled).The data has 8 quantitative input variables, and 1 quantitative output variable, and 1030 instances (observations).
# 
# ## **Objective:**
# Modeling of strength of high performance concrete using Machine Learning

# ### Lets import all the required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import iqr 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading the Dataset
data = pd.read_csv('/kaggle/input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv')
# lets take a look on dataset
data.head()


# In[ ]:


#Renaming feature names
col_map = {'Cement (component 1)(kg in a m^3 mixture)': 'cement',
 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'slag',
 'Fly Ash (component 3)(kg in a m^3 mixture)': 'ash',
 'Water  (component 4)(kg in a m^3 mixture)': 'water',
 'Superplasticizer (component 5)(kg in a m^3 mixture)': 'superplastic',
 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'coarseagg',
 'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'fineagg',
 'Age (day)': 'age',
 'Concrete compressive strength(MPa, megapascals) ': 'strength'}


# In[ ]:


data.rename(columns=col_map,inplace=True)
data.head()


# ### Lets Explore the Dataset

# In[ ]:


# Checking for null values in dataset
data.info()


# ## Exploratory Data Analysis

# In[ ]:


# Five Point Statistics Summary
#data.describe().T
summary = data.describe().T
summary['Diff'] = summary['mean'] - summary['50%']
summary


# ### Inference
# 1. Slag, Ash & Age has right skewed distribution as Mean > Median.
# 2. Even the Cement Variable is slightly right skewed.
# 
# ### Now lets plot and check the distribution

# In[ ]:


sns.pairplot(data)


# ### Inference
# 1. As inferred, slag, ash & age are right skewed and variable cement is also slight right skewed.
# 2. Most of the other features has multimodal distribution.
# 
# ## Univariate Analysis

# In[ ]:


## Plotting BoxPlot to perform univariate analysis
data.plot(kind='box',figsize=(15,8))


# 1. Feature "fineagg" & "age" has extreme outliers.
# 2. Feature "Slag", "Water" and "superplastic" also has outlier but are not extreme so we do not need to treat them.

# ### Outlier Treatment
# 1. Find out the samples which are beyond the (IQR*1.5 + Q3) values.
# 2. Replace all such values with the Median.
# 
# Note: Median is robust to the effect of Outlier.

# In[ ]:


# Function to find the Upper Cut-off & Median Value for the Given Variable
def outlier_cap(df):
    IQR = iqr(df)
    Q3 = np.percentile(df,75)
    ucap = IQR*1.5 + Q3
    median = df[df<ucap].median()
    return ucap,median


# In[ ]:


# Treating fineagg outlier
f,((ax_box,ax_box_post),(ax_hist,ax_hist_post)) = plt.subplots(2,2,gridspec_kw={'height_ratios':(0.15,0.85)},figsize=(10,7))
sns.boxplot(data['fineagg'],ax=ax_box).set_title("fineagg_Pre")
sns.distplot(data['fineagg'],ax=ax_hist)
ucap_fineagg,median_fineagg = outlier_cap(data['fineagg'])
data.loc[data['fineagg']>ucap_fineagg,'fineagg'] = median_fineagg
sns.boxplot(data['fineagg'],ax=ax_box_post).set_title("fineagg_Post")
sns.distplot(data['fineagg'],ax=ax_hist_post)


# In[ ]:


# Treating age outlier
f,((ax_box,ax_box_post),(ax_hist,ax_hist_post)) = plt.subplots(2,2,gridspec_kw={'height_ratios':(0.15,0.85)},figsize=(10,7))
sns.boxplot(data['age'],ax=ax_box).set_title("age_Pre")
sns.distplot(data['age'],ax=ax_hist)
ucap_age,median_age = outlier_cap(data['age'])
data.loc[data['age']>ucap_age,'age'] = median_age
sns.boxplot(data['age'],ax=ax_box_post).set_title("age_Post")
sns.distplot(data['age'],ax=ax_hist_post)


# ## Bivariate Analysis

# In[ ]:


#%matplotlib notebook
plt.figure(figsize=(9,5))
corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
#ax = sns.heatmap(corr,annot=True,linewidth=0.5)
with sns.axes_style("white"):
    ax = sns.heatmap(corr,annot=True,linewidth=2,
                mask = mask,cmap="magma")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# - Cement, Superplastic & age is highly correlated with dependent variable.
# - water is highly correlated with superplastic.
# - As we know the perfect ratio of water & cement gives concrete the desired strength.
# - Lets try to take a ratio of cement & water to make new feature.

# In[ ]:


# Feature Engineering
data['water_cement'] = data['cement']/data['water']
data.corr()


# - From above chart, new created feature water_cement is showing a good correlation with dependent variable.
# - Also as superplastic is highly correlated with another independent variable i.e water.
# - lets drop water & cement from the dataset

# In[ ]:


#Splitting the dataset in X & y
X = data.drop(['strength','cement','water'],axis=1)
y = data['strength']


# In[ ]:


# Importing preprocessing & sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Splitting the dataset into training & testing dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=400)

#Scaling the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Lasso Regression

# In[ ]:


# Importing linear , Tree & Ensemble model libraries
from sklearn.linear_model import Lasso,Ridge
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


#Lets create a Lasso object & try to fit our model
reg_lasso = Lasso(alpha=0.1,max_iter=10e5)
reg_lasso.fit(X_train_scaled,y_train)
#Checking the train & test score
print("Train Score {} & Test Score {}".format(reg_lasso.score(X_train_scaled,y_train),reg_lasso.score(X_test_scaled,y_test)))


# In[ ]:


# Lets check the Feature Coefficent of Lasso Model
values = np.argsort(reg_lasso.coef_)
val = sorted(reg_lasso.coef_)
columns = X.columns.tolist()
col = [columns[i] for i in values]
plt.bar(col,val)
plt.title("Lasso Co-efficient Plot")
plt.xticks(rotation=45)


# # Ridge Regression

# In[ ]:


#Lets create a Ridge object & try to fit our model
reg_ridge = Ridge(alpha=0.01,max_iter=10e5)
reg_ridge.fit(X_train_scaled,y_train)
print("Train Score {} & Test Score {}".format(reg_ridge.score(X_train_scaled,y_train),reg_ridge.score(X_test_scaled,y_test)))


# In[ ]:


# Lets check the Feature Coefficent of Ridge Model
values = np.argsort(reg_ridge.coef_)
val = sorted(reg_ridge.coef_)
columns = X.columns.tolist()
col = [columns[i] for i in values]
plt.title("Ridge Co-efficient Plot")
plt.bar(col,val)
plt.xticks(rotation=45)


# >Bar Chart Shows that "courseagg" has co-efficient close to zero.
# >New Engineering Feature has the highest coe-efficient values.

# # DecisionTree Model

# In[ ]:


#Modelling with DecisionTreeRegressor
reg_dt = DecisionTreeRegressor()
reg_dt.fit(X_train,y_train)
print("Train Score {:.2f} & Test Score {:.2f}".format(reg_dt.score(X_train,y_train),reg_dt.score(X_test,y_test)))


# - As usual Decision Tree seems to have overfit.
# - Let try to tune to hyperparameter.

# In[ ]:


#Tuning Hyperparameter max_depth & min_sam_split of DecisionTreeRegressor
max_d = list(range(1,10))
min_sam_split = list(range(10,50,15))
from sklearn.model_selection import GridSearchCV
gridcv = GridSearchCV(reg_dt,param_grid={'max_depth':max_d,'min_samples_split':min_sam_split},n_jobs=-1)
gridcv.fit(X_train,y_train)


# In[ ]:


print("Parameters :",gridcv.best_params_)
print("Train Score {:.2f} & Test Score {:.2f}".format(gridcv.score(X_train,y_train),gridcv.score(X_test,y_test)))


# In[ ]:


# Lets find out the feature importance based on DecisionTree Model
importances = reg_dt.feature_importances_
col = X.columns.tolist()
indices = importances.argsort()[::-1]
names = [col[i] for i in indices]

# Plotting Feature importance Chart
plt.title("Decision Tree: Feature Importance")
plt.bar(range(X.shape[1]),importances[indices])
plt.xticks(range(X.shape[1]),names,rotation=45);


# > water_cement & age followed by slag seems to be the most importance feature.
# 
# # Lets use Ensemble Technique
# 
# ### RandomForest Model
# 

# In[ ]:


#Modelling with RandomForestRegressor
reg_rfe = RandomForestRegressor()
reg_rfe.fit(X_train,y_train)
print("Train Score {:.2f} & Test Score {:.2f}".format(reg_rfe.score(X_train,y_train),reg_rfe.score(X_test,y_test)))


# In[ ]:


# Tuning Hyperparameter of DecisionTreeRegressor
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Maximum number of levels in tree
max_depth = list(range(10,110,10))
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
from sklearn.model_selection import GridSearchCV
gridcv_rf = GridSearchCV(reg_rfe,param_grid={'n_estimators':n_estimators,'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf},n_jobs=-1)
gridcv_rf.fit(X_train,y_train)


# In[ ]:


print("Best Parameters:",gridcv_rf.best_params_)
print("Train Score {:.2f} & Test Score {:.2f}".format(gridcv_rf.score(X_train,y_train),gridcv_rf.score(X_test,y_test)))


# > With Hyperparameter Tuning using Gridsearchcv, we have been able to improve the accuracy of Model from 90% to 92%.

# ### We have used following model and the results are :
# 1. Lasso : Train Score 0.70 & Test Score 0.70
# 2. Ridge : Train Score 0.70 & Test Score 0.70
# 3. Decision Tree : Train Score 0.93 & Test Score 0.83
# 4. RandomForest : Train Score 0.98 & Test Score 0.92
# 
# Now Lets Try to model it using XGBoost Regressor

# # XGBoost

# In[ ]:


import xgboost
reg_xgb = xgboost.XGBRegressor()
# Fitting model 
reg_xgb.fit(X_train,y_train)
print("Train Score {:.2f} & Test Score {:.2f}".format(reg_xgb.score(X_train,y_train),reg_xgb.score(X_test,y_test)))


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

gridcv_xgb = GridSearchCV(reg_xgb,param_grid={'n_estimators':list(range(100,1000,100)),'max_depth':list(range(1,10,1)),
                                     'learning_rate':[0.0001,0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.25,0.3]})
gridcv_xgb.fit(X_train,y_train);


# In[ ]:


print("Best Parameter:",gridcv_xgb.best_params_)
print("Train Score {:.2f} & Test Score {:.2f}".format(gridcv_xgb.score(X_train,y_train),gridcv_xgb.score(X_test,y_test)))


# # <u>Summary :</u>
# 
# ### Linear Model
# - Accuracy on Test Data Set with Lasso Regression : 70%
# - Accuracy on Test Data Set with Ridge Regression : 70%
# 
# ### Tree Model
# - Accuracy on Test Data Set with DecisionTree Regression : 85%
# 
# ### Ensemble Model
# - Accuracy on Test Data Set with RandomForest Regression : 92%
# 
# ### XGBoost Model
# - Accuracy on Test Data Set with XGBoost Regression : 94%
# 
# > So far <b>XGBoostRegression</b> proved to be the best performing model with 94% accuracy.
# 

# In[ ]:




