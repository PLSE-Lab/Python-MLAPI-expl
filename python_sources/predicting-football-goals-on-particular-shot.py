#!/usr/bin/env python
# coding: utf-8

# ## Loading the required libraries

# In[ ]:


import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score




print(os.listdir("../input"))


# Load the dataset

# In[ ]:


df =pd.read_csv("../input/data.csv", index_col=0)


# # Data Preperation
# 
# In this section analyse the dataset and their features.
# 
# Analyse categorical features and numerical features 
# 
# check for columns that they are containing how many null values
# 
# then fill the null values by the most occuring value of that column
# 

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df['index_col'] = df.index


# In[ ]:


#Total columns
df.columns


# In[ ]:


df = df[['index_col','is_goal','shot_id_number','match_event_id', 'location_x', 'location_y', 'remaining_min',
       'power_of_shot', 'knockout_match', 'game_season', 'remaining_sec',
       'distance_of_shot', 'area_of_shot', 'shot_basics',
       'range_of_shot', 'team_name', 'date_of_game', 'home/away', 'lat/lng', 'type_of_shot', 'type_of_combined_shot',
       'match_id', 'team_id', 'remaining_min.1', 'power_of_shot.1',
       'knockout_match.1', 'remaining_sec.1', 'distance_of_shot.1']]
df.head()


# In[ ]:


print(df.info())


# In[ ]:


print('Categorical columns are \n')
cat_columns = [x for x in df.dtypes.index if df.dtypes[x]=='object']
num_columns = [x for x in df.dtypes.index if df.dtypes[x]!='object']
print(cat_columns)
print('\n')
print('Non categorical columns are \n')
print(num_columns)


# In[ ]:


#df2=df.copy()


# In[ ]:


df.apply(lambda x: sum(x.isnull()))


# In[ ]:


df = df.fillna(df.mode().iloc[0])


# In[ ]:


df.apply(lambda x: sum(x.isnull()))


# In[ ]:


df.shape


# In[ ]:


corr=df.corr()["is_goal"]
corr[np.argsort(corr, axis=0)[::-1]]


# # EDA

# Check for the correlations

# In[ ]:


#VISUALIZATION
#plotting correlations
num_feat=df.columns[df.dtypes!=object]
num_feat=num_feat[1:-1] 
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(df[col].values, df.is_goal.values)[0,1])

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(16,10))
rects = ax.barh(ind, np.array(values), color='red')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t y");


# Correlation matrix

# In[ ]:


corrMatrix=df[num_feat].corr()
sns.set(font_scale=1.10)
plt.figure(figsize=(19, 19))
sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');


# #### Let's see how the numeric data is distributed.
# 

# In[ ]:



df.hist(bins=10, figsize=(20,15), color='#E14906')
plt.show()


# #### Checking how the mean of features remaining_sec,distance_of_shot,power_of_shot are related with the target column IS_GOAL

# In[ ]:


ax = df.groupby('is_goal').remaining_sec.mean().plot(kind='bar')
ax.set_xlabel("is_goal(outcome)")
ax.set_ylabel("mean remaining_sec")


# In[ ]:


ax = df.groupby('is_goal').distance_of_shot.mean().plot(kind='bar')
ax.set_xlabel("is_goal(outcome)")
ax.set_ylabel("mean distance_of_shot")


# In[ ]:


ax = df.groupby('is_goal').power_of_shot.mean().plot(kind='bar')
ax.set_xlabel("is_goal(outcome)")
ax.set_ylabel("mean power_of_shot")


# #### Counts of different categorical features and their comparision 
# 

# In[ ]:


import seaborn as sns
sns.set(style="white")
fig, ax = plt.subplots(figsize=(12,8))
sns.countplot(x="type_of_combined_shot", data=df, palette="Set2")
ax.set_title("Different type_of_combined_shot", fontsize=20)
ax.set_xlabel("type_of_combined_shot")
plt.show()


# In[ ]:


import seaborn as sns
sns.set(style="white")
fig, ax = plt.subplots(figsize=(12,8))
sns.countplot(x="shot_basics", data=df, palette="Set2")
ax.set_title("Different shot_basics", fontsize=20)
ax.set_xlabel("shot_basics")
plt.show()


# In[ ]:


import seaborn as sns
sns.set(style="white")
fig, ax = plt.subplots(figsize=(19,14))
sns.countplot(x="area_of_shot", data=df, palette="Set2")
ax.set_title("Different area_of_shot", fontsize=20)
ax.set_xlabel("area_of_shot")
plt.show()


# ##### Plotting relation between the type_of_combined_shot and  power_of_shot with respect to target

# In[ ]:


sns.factorplot('type_of_combined_shot','power_of_shot',hue='is_goal',data=df )


# ##### Plotting relation between the type_of_combined_shot and  remaining_min with respect to target

# In[ ]:


sns.factorplot('type_of_combined_shot','remaining_min',hue='is_goal',data=df )


# ### Relation between the calegorical data and the mean of the target 

# In[ ]:


df.groupby('game_season').is_goal.mean().plot(kind='bar')


# In[ ]:


df.groupby('area_of_shot').is_goal.mean().plot(kind='bar')


# In[ ]:


df.groupby('shot_basics').is_goal.mean().plot(kind='bar')


# In[ ]:


df.groupby('range_of_shot').is_goal.mean().plot(kind='bar')


# In[ ]:


df.groupby('area_of_shot').is_goal.mean().plot(kind='bar')


# In[ ]:


df.groupby('type_of_combined_shot').is_goal.mean().plot(kind='bar')


# # Feature Engineering

# ## Check for the categorical data so that we can populate our dataset and we can do encoding to the dataset

# In[ ]:


#Filter categorical variables
categorical_columns = [x for x in df.dtypes.index if df.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['date_of_game','team_name','lat/lng']]
#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (df[col].value_counts())


# #### Populating the dataset

# In[ ]:


df=pd.get_dummies(df, columns=["type_of_combined_shot"])
df=pd.get_dummies(df, columns=["home/away"])
df=pd.get_dummies(df, columns=["shot_basics"])
df=pd.get_dummies(df, columns=["area_of_shot"])
df=pd.get_dummies(df, columns=["game_season"])
df=pd.get_dummies(df, columns=["team_name"])
df=pd.get_dummies(df, columns=["type_of_shot"])
df=pd.get_dummies(df, columns=["range_of_shot"])
df=pd.get_dummies(df, columns=["lat/lng"])


# In[ ]:


get_ipython().run_cell_magic('time', '', "df['year'] = pd.DatetimeIndex(df['date_of_game']).year\ndf['month'] = pd.DatetimeIndex(df['date_of_game']).month\ndf['day'] = pd.DatetimeIndex(df['date_of_game']).day\ndf.head()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df.drop(['date_of_game', 'match_event_id','match_id','team_id','shot_id_number'], axis=1, inplace=True)")


# In[ ]:


df.describe()


# #### Standerizing some column whose values are large

# In[ ]:


df["location_x"] = df["location_x"] / df["location_x"].max()
df["location_y"] = df["location_y"] / df["location_y"].max()
df["remaining_sec"] = df["remaining_sec"] / df["remaining_sec"].max()
df["distance_of_shot"] = df["distance_of_shot"] / df["distance_of_shot"].max()
df["remaining_min.1"] = df["remaining_min.1"] / df["remaining_min.1"].max()
df["knockout_match.1"] = df["knockout_match.1"] / df["knockout_match.1"].max()
df["remaining_sec.1"] = df["remaining_sec.1"] / df["remaining_sec.1"].max()
df["distance_of_shot.1"] = df["distance_of_shot.1"] / df["distance_of_shot.1"].max()


# In[ ]:


df.describe()


# In[ ]:


train_df, test_df = train_test_split(df ,test_size=0.19, random_state=2)


# In[ ]:


test_df=test_df.drop('is_goal',axis=1)


# In[ ]:


print (train_df.shape)
print (test_df.shape)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'missing_data(train_df)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'missing_data(test_df)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df.describe()')


# In[ ]:


get_ipython().run_line_magic('time', '')
test_df.describe()


# In[ ]:


sns.countplot(train_df['is_goal'], palette='Set3')


# In[ ]:


# Density plots of features
# Let's show now the density plot of variables in train dataset.

# We represent with different colors the distribution for values with target value 0 and 1.

def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


# In[ ]:


# Apply when dataset contain only numerical data

t0 = train_df.loc[train_df['is_goal'] == 0]
t1 = train_df.loc[train_df['is_goal'] == 1]
features = train_df.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)


# In[ ]:


# Distribution of mean and std
# Let's check the distribution of the mean values per row in the train and test set.

plt.figure(figsize=(16,6))
features = train_df.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train_df[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[ ]:


# Distribution of skew and kurtosis
# Let's see now what is the distribution of skew values per rows and columns.

# Let's see first the distribution of skewness calculated per rows in train and test sets.

plt.figure(figsize=(16,6))
plt.title("Distribution of skew per row in the train and test set")
sns.distplot(train_df[features].skew(axis=1),color="red", kde=True,bins=120, label='train')
sns.distplot(test_df[features].skew(axis=1),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[ ]:


print('Train and test columns: {} {}'.format(len(train_df.columns), len(test_df.columns)))


# # Model Building

# In[ ]:


# Model
# From the train columns list, we drop the ID and target to form the features list.

features = [c for c in train_df.columns if c not in ['index_col', 'is_goal']]
target = train_df['is_goal']


# In[ ]:


# We define the hyperparameters for the model.

param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}


# In[ ]:


#run the model.

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


# In[ ]:


cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:150].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,28))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
plt.savefig('FI.png')


# In[ ]:


# sub_df = pd.DataFrame({"shot_id_number":test_df["index_col"].values})
# sub_df["is_goal"] = predictions
# sub_df.to_csv("submissionsub4.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


Y = df['is_goal']
X = df.drop('is_goal', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,Y ,test_size=0.19, random_state=6)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers
from sklearn.metrics import accuracy_score


# In[ ]:


print('Building Neural Network model...')
adam = optimizers.adam(lr = 0.005, decay = 0.00001)

model = Sequential()
model.add(Dense(48, input_dim=X_train.shape[1],
                kernel_initializer='normal',
                #kernel_regularizer=regularizers.l2(0.02),
                activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(48,
                #kernel_regularizer=regularizers.l2(0.02),
                activation="tanh"))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam')

history = model.fit(X_train, y_train, validation_split=0.19, epochs=16, batch_size=24)


# In[ ]:


# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:


#Predict on test set
predictions_NN_prob = model.predict(X_test)


# In[ ]:


predictions_NN_prob


# In[ ]:


sub_df = pd.DataFrame({"shot_id_number":test_df["index_col"].values})
sub_df["is_goal"] = predictions_NN_prob


# In[ ]:


sub_df = sub_df.iloc[0:5000]


# In[ ]:


sub_df.to_csv("submissionfinal.csv", index=False)


# In[ ]:


# # Logistic Regression
# log_reg = LogisticRegression()
# log_scores = cross_val_score(log_reg, X_train, y_train, cv=3)
# log_reg_mean = log_scores.mean()
# print(log_reg_mean)


# In[ ]:


# %%time
# log_reg.fit(X_train, y_train)
# #Prediction using test data
# label_pred_lreg = log_reg.predict(X_test)
# #classification accuracy
# from sklearn import metrics
# print(metrics.accuracy_score(y_test, label_pred_lreg))
# # evaluate predictions
# accuracy_r = metrics.accuracy_score(y_test, label_pred_lreg)
# print("Accuracy: %.2f%%" % (accuracy_r * 100.0))


# In[ ]:


# # Gradient Boosting Classifier
# grad_clf = GradientBoostingClassifier()
# grad_scores = cross_val_score(grad_clf, X_train, y_train, cv=3)
# grad_mean = grad_scores.mean()
# print(grad_mean)


# In[ ]:


# %%time
# grad_clf.fit(X_train, y_train)
# #Prediction using test data
# label_pred_clf = grad_clf.predict(X_test)
# #classification accuracy
# from sklearn import metrics
# print(metrics.accuracy_score(y_test, label_pred_clf))
# # evaluate predictions
# accuracy_g = metrics.accuracy_score(y_test, label_pred_clf)
# print("Accuracy: %.2f%%" % (accuracy_g * 100.0))


# In[ ]:


# sub_df = pd.DataFrame({"shot_id_number":test_df["index_col"].values})
# sub_df["is_goal"] = predictions
# sub_df.to_csv("submissionsub3.csv", index=False)

