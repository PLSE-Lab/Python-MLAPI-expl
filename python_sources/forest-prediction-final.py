#!/usr/bin/env python
# coding: utf-8

# * In this competition you are asked to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data).
# * The data is in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.

# # Loading Packages

# In[ ]:


get_ipython().system('pip install seaborn')


# In[ ]:


# for data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualization
from matplotlib import pyplot as plt
# to include graphs inline within the frontends next to code
import seaborn as sns

# preprocessing functions and evaluation models
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier

# machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier


# # Loading Data

# In[ ]:


submission_sample = pd.read_csv('../input/forest-cover-type-prediction/sampleSubmission.csv')
train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')


# # Cleaning and Understanding Data

# In[ ]:


print("Number of rows and columns in the train dataset are:", train.shape)


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.dtypes


# In[ ]:


print(list(enumerate(train.columns)))


# In[ ]:


train.iloc[:,1:10].describe()


# In[ ]:


train.nunique()


# In[ ]:


train.isna().sum()


# Checking Outliers

# In[ ]:


def outlier_function(df, col_name):
    ''' this function detects first and third quartile and interquartile range for a given column of a dataframe
    then calculates upper and lower limits to determine outliers conservatively
    returns the number of lower and uper limit and number of outliers respectively
    '''
    first_quartile = np.percentile(np.array(df[col_name].tolist()), 25)
    third_quartile = np.percentile(np.array(df[col_name].tolist()), 75)
    IQR = third_quartile - first_quartile
                      
    upper_limit = third_quartile+(3*IQR)
    lower_limit = first_quartile-(3*IQR)
    outlier_count = 0
                      
    for value in df[col_name].tolist():
        if (value < lower_limit) | (value > upper_limit):
            outlier_count +=1
    return lower_limit, upper_limit, outlier_count


# In[ ]:


# loop through all columns to see if there are any outliers
for i in train.columns:
    if outlier_function(train, i)[2] > 0:
        print("There are {} outliers in {}".format(outlier_function(train, i)[2], i))


# We will take closer look to below 4 columns:
# * 
# * Horizontal_Distance_To_Hydrology
# * Vertical_Distance_To_Hydrology
# * Horizontal_Distance_To_Roadways
# * Horizontal_Distance_To_Fire_Points
# We are not going to consider other columns for potential outlier elimination because their data range is already fixed between 0 and 255 (e.g. Hillsahde columns) or they seem like one-hot-encoded columns (e.g. Soil type and Wilderness areas).
# 
# Recall the data ranges of those 4 columns:
# 
# * Horizontal_Distance_To_Hydrology: 0, 1343
# * Vertical_Distance_To_Hydrology: -146, 554
# * Horizontal_Distance_To_Roadways: 0, 6890
# * Horizaontal_Distance_To_Firepoints: 0, 6993
# Horizaontal_Distance_To_Firepoints having the highest number of outliers and widest data range.
# 
# 
# 
# 

# Observations from cleaning and understanding data:
# 
# * Train dataset has 15120 rows and 56 columns.
# * Each column has numeric (integer/float) datatype.
# * There are no NA in the dataset.Thus dataset is properly formatted.
# * 4 columns had outliers.
# * Cover_Type is our label/target column.

# In[ ]:


trees = train[(train['Horizontal_Distance_To_Fire_Points'] > outlier_function(train, 'Horizontal_Distance_To_Fire_Points')[0]) &
              (train['Horizontal_Distance_To_Fire_Points'] < outlier_function(train, 'Horizontal_Distance_To_Fire_Points')[1])]
trees.shape


# # Exploratory Data Analysis

# Checking If Wilderness Area and Soil Type are binary or not?

# In[ ]:


size=10
Uni = []
for i in range(size+1,len(train.columns)-1):
    Uni.append(pd.unique(train[train.columns[i]].values))


# In[ ]:


Uni


# Yes, Wilderness Areas and Soil Types have binary values.

# Check if Tree belong to multiple soil types and wilderness areas ?

# In[ ]:


train.iloc[:,11:15].sum(axis=1).sum()


# In[ ]:


train.iloc[:,15:55].sum(axis=1).sum()


# Wilderness_Area and Soil_Type1-40 having only binary values and only one soil_type out of 40 soil types or wilderness_area out of 4 types being equal to 1, shows that they are one-hot-encoded columns.
# 
# One important thing about trees, they can only belong to one soil type or one wilderness area.
# 
# 

# Distribution of Trees

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(train['Cover_Type'])
plt.xlabel("Type of Cpver", fontsize=12)
plt.ylabel("Rows Count", fontsize=12)
plt.show()


# Distribution of trees shows perfect uniform distribution.
# 
# Here are the 7 types of the trees, numbered from 1 to 7 in the Cover_Type column:
# 
# 1) Spruce/Fir
# 
# 2) Lodgepole Pine
# 
# 3) Ponderosa Pine
# 
# 4) Cottonwood/Willow
# 
# 5) Aspen
# 
# 6) Douglas-fir
# 
# 7) Krummholz

# Distribution and relationship of continuous variables (Elevation, Aspect, Slope, Distance and Hillsahde columns)

# In[ ]:


fig, axes = plt.subplots(nrows = 2,ncols = 5,figsize = (25,15))
g= sns.FacetGrid(train, hue='Cover_Type',height=5)
(g.map(sns.distplot,train.columns[1],ax=axes[0][0]))
(g.map(sns.distplot, train.columns[2],ax=axes[0][1]))
(g.map(sns.distplot, train.columns[3],ax=axes[0][2]))
(g.map(sns.distplot, train.columns[4],ax=axes[0][3]))
(g.map(sns.distplot, train.columns[5],ax=axes[0][4]))
(g.map(sns.distplot, train.columns[6],ax=axes[1][0]))
(g.map(sns.distplot, train.columns[7],ax=axes[1][1]))
(g.map(sns.distplot, train.columns[8],ax=axes[1][2]))
(g.map(sns.distplot, train.columns[9],ax=axes[1][3]))
(g.map(sns.distplot, train.columns[10],ax=axes[1][4]))
plt.close(2)
plt.legend()


# Looking at distribution plots above, it is observed that distribution of numerical variables is not normal. Some of them are right skewed and some of them are left skewed. Distribution of Aspects seems like binary.
# 
# In conclusion, we have to scale our numerical variables before they go in model.

# In[ ]:


size=10
fig, axes = plt.subplots(nrows = 2,ncols = 5,figsize = (25,10))
for i in range(0,size):
    row = i // 5
    col = i % 5
    ax_curr = axes[row, col]
    sns.boxplot(x="Cover_Type", y=train.columns[i], data=train,ax=ax_curr);


# Observations:
# 
# * Forest "Cover_Type" 1 and 7 have higher "Elevation" than others while 4 has lowest among the all.
# * All Forest "Cover_Type" are spreded out in "Aspect".
# * Forest Cover Type 1, 2 and 7 have higher Horizontal_Distance_To_Roadways.
# * There is not much variation in distribution of Forest Cover Type for each of the other features.
# * Other features have almost same level of distribution of Forest Cover Types.

# In[ ]:


sns.pairplot(train, hue='Cover_Type', vars=train.columns[1:11])


# In[ ]:


# Bivariate EDA
pd.crosstab(train.Soil_Type31, train.Cover_Type)


# In[ ]:


#Convert dummy features back to categorical
x = train.iloc[:,15:55]
y = train.iloc[:,11:15]
y = pd.DataFrame(y)
x = pd.DataFrame(x)
s2 = pd.Series(x.columns[np.where(x!=0)[1]])
s3 = pd.Series(y.columns[np.where(y!=0)[1]])
train['soil_type'] = s2
train['Wilderness_Area'] = s3
train.head()


# In[ ]:


# Create a new dataset exluding dummies variable for Mutivariate EDA
df_viz = train.iloc[:, 0:15]
df_viz = df_viz.drop(['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 
                      'Wilderness_Area4'], axis = 1)
df_viz.head()


# In[ ]:


plt.figure(figsize=(15,10))
pd.crosstab(train.Wilderness_Area, train.Cover_Type).plot.bar(figsize=(10,10),stacked = True)


# Observations:
# 
# * Wildnerss Area 2 is very rare compared to other types and is observed on,y in Cover Type 1 and 7 mainly.
# * All Forest Cover Type 4 have Wildnerss Area 4 only
# * Wildneress Area 4 is mainly obeserved in Cover Type 3, 4 and 6.
# * Wildnerss Area 3 is observed in all Cover Types except cover type 4.

# In[ ]:


#plt.figure(figsize=(30,30))
pd.crosstab(train.soil_type, train.Cover_Type).plot.bar(figsize=(20,10),stacked = True)


# Observations:
# 
# * Soil Type 18,19,21,25,26,27,28,34,36,37,8 and 9 are vary rarely oberved
# * Soil 10 is observed maximum number of times (>2200 occurences)
# * Soil types in Cover Type 3 anad cover type 6 are similar. Look at the Brown and Green bar from the graph
# * Soil types in Cover Type 1 and 2 are similar. Look at orange and blue bars

# In[ ]:


corr = df_viz.corr()

# plot the heatmap
plt.figure(figsize=(14,12))
colormap = plt.cm.RdBu
sns.heatmap(corr,linewidths=0.1, 
            square=False, cmap=colormap, linecolor='white', annot=True)
plt.title('Pearson Correlation of Numeric Features', size=14)


# There are some correlation pairs with strong negative and positive correlations. Please look at the graph, the darker the color, more the correlation

# # Feature Engineering

# In[ ]:


train=trees


# In[ ]:


def add_feature(data):   
    data['Ele_minus_VDtHyd'] = data.Elevation-data.Vertical_Distance_To_Hydrology
    data['Ele_plus_VDtHyd'] = data.Elevation+data.Vertical_Distance_To_Hydrology
    data['Distanse_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology']**2+data['Vertical_Distance_To_Hydrology']**2)**0.5
    data['Hydro_plus_Fire'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Fire_Points']
    data['Hydro_minus_Fire'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Fire_Points']
    data['Hydro_plus_Road'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Roadways']
    data['Hydro_minus_Road'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Roadways']
    data['Fire_plus_Road'] = data['Horizontal_Distance_To_Fire_Points']+data['Horizontal_Distance_To_Roadways']
    data['Fire_minus_Road'] = data['Horizontal_Distance_To_Fire_Points']-data['Horizontal_Distance_To_Roadways']
    return data


# In[ ]:


train = add_feature(train)
test = add_feature(test)


# In[ ]:


#X_train = train.drop(['Id','Cover_Type','soil_type','Wilderness_Area'], axis = 1)
X_train = train.drop(['Id','Cover_Type'], axis = 1)
y_train = train.Cover_Type
X_test = test.drop(['Id'], axis = 1)


# # Logistics regression****

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nlr_pipe = Pipeline(\n    steps = [\n        ('scaler', MinMaxScaler()),\n        ('classifier', LogisticRegression(solver='lbfgs', n_jobs=-1))\n    ]\n)\n\nlr_param_grid = {\n    'classifier__C': [1, 10, 100,1000],\n}\n\n\nnp.random.seed(1)\ngrid_search = GridSearchCV(lr_pipe, lr_param_grid, cv=5, refit='True')\ngrid_search.fit(X_train, y_train)\n\nprint(grid_search.best_score_)\nprint(grid_search.best_params_)")


# # Random Forest****

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nrf_pipe = Pipeline(\n    steps = [\n        ('classifier', RandomForestClassifier(n_estimators=500))\n    ]\n)\n\nparam_grid = {\n       'classifier__min_samples_leaf': [1,4,7],\n    'classifier__max_depth': [34,38,32],\n}\n\nnp.random.seed(1)\nrf_grid_search = GridSearchCV(rf_pipe, param_grid, cv=5, refit='True', n_jobs=-1)\nrf_grid_search.fit(X_train, y_train)\n\nprint(rf_grid_search.best_score_)\nprint(rf_grid_search.best_params_)")


# In[ ]:


rf_model = rf_grid_search.best_estimator_

cv_score = cross_val_score(rf_model, X_train, y_train, cv = 5)
print(cv_score)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))


# In[ ]:


rf = rf_grid_search.best_estimator_.steps[0][1]


# In[ ]:


feat_imp = rf.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature':X_train.columns,
    'feat_imp':feat_imp
})

feat_imp_df.sort_values(by='feat_imp', ascending=False).head(10)


# In[ ]:


sorted_feat_imp_df = feat_imp_df.sort_values(by='feat_imp', ascending=True)
plt.figure(figsize=[6,6])
plt.barh(sorted_feat_imp_df.feature[-20:], sorted_feat_imp_df.feat_imp[-20:])
plt.show()


# # ****Gradient Boosting

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nxgd_pipe = Pipeline(\n    steps = [\n        ('classifier', XGBClassifier(n_estimators=50, subsample=0.5))\n    ]\n)\n\nparam_grid = {\n    'classifier__learning_rate' : [0.45],\n    'classifier__min_samples_split' : [8, 16, 32],\n    'classifier__min_samples_leaf' : [2],\n    'classifier__max_depth': [15]\n    \n}\n\nnp.random.seed(1)\nxgd_grid_search = GridSearchCV(xgd_pipe, param_grid, cv=5,\n                              refit='True', verbose = 10, n_jobs=-1)\nxgd_grid_search.fit(X_train, y_train)\n\nprint(xgd_grid_search.best_score_)\nprint(xgd_grid_search.best_params_)")


# In[ ]:


xgd_model = xgd_grid_search.best_estimator_

cv_score = cross_val_score(xgd_model, X_train, y_train, cv = 5)
print(cv_score)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))


# # ****ExtraTree Random Forest

# In[ ]:


xrf_pipe = Pipeline(
    steps = [
        ('classifier', ExtraTreesClassifier(n_estimators=500,random_state=0, criterion = 'entropy'))
    ]
)


xrf_param_grid = {
    'classifier__min_samples_leaf': [1,4,7],
    'classifier__max_depth': [34,38,32],
}

np.random.seed(1)
xrf_grid_search = GridSearchCV(xrf_pipe, xrf_param_grid, cv=5, refit='True', n_jobs=-1)
xrf_grid_search.fit(X_train, y_train)

print(xrf_grid_search.best_score_)
print(xrf_grid_search.best_params_)


# In[ ]:


xrf_model = xrf_grid_search.best_estimator_

cv_score = cross_val_score(xrf_model, X_train, y_train, cv = 5)
print(cv_score)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))


# In[ ]:


final_model = xrf_grid_search.best_estimator_.steps[0][1]
final_model.fit(X_train, y_train)







# In[ ]:


y_pred = final_model.predict(X_test)


# In[ ]:


print(len(test.Id))


# In[ ]:


print(len(y_pred))


# In[ ]:


from collections import Counter
Counter(y_pred)


# In[ ]:


submission_sample.head()


# In[ ]:


submission = pd.DataFrame({'Id': test.Id,
                           'Cover_Type': y_pred})
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

