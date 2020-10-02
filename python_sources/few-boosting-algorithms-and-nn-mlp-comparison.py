#!/usr/bin/env python
# coding: utf-8

# ![](https://i.imgur.com/9BUtrm4.png)

# In[ ]:


#import libraries
import pandas as pd
import numpy as np
import lightgbm
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
import os
print(os.listdir("../input"))


# In[ ]:


#import dataset
df = pd.read_csv('../input/bank-customer-churn-modeling/Churn_Modelling.csv')
print(df.shape)
df.head()


# In[ ]:


#Check balance between output classes
y = df["Exited"].value_counts()
sns.barplot(y.index, y.values)


# In[ ]:


#Check balance between states
geo = df.Geography.value_counts()
sns.barplot(geo.index, geo.values)


# In[ ]:


#Check balance between gender
gen = df.Gender.value_counts()
sns.barplot(gen.index, gen.values)


# In[ ]:


#count the exit ratio in countries
countries = ['France', 'Germany', 'Spain']

value_indexes = []

for value in countries:
  df_state = df[df['Geography']==value]
  value_index = (df_state.Exited != 0).values.sum()/len(df_state)
  value_indexes.append(value_index)

dictionary = dict(zip(countries, value_indexes))

df['Geo_ratio'] = df['Geography'].map(dictionary)
df.head()


# In[ ]:


#count the exit ratio for gender index
gender_list = ['Female', 'Male']

value_indexes = []

for value in gender_list:
  df_gender = df[df['Gender']==value]
  value_index = (df_gender.Exited != 0).values.sum()/len(df_gender)
  value_indexes.append(value_index)

dictionary_gen = dict(zip(gender_list, value_indexes))

df['Gen_ratio'] = df['Gender'].map(dictionary_gen)
df.head()


# In[ ]:


#check number of uniques
df.nunique()


# In[ ]:


#drop any nans if exist
df = df.dropna()
print(df.shape)


# In[ ]:


#drop useless data
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace = True)
df.head()


# In[ ]:


#list of categorical columns
FILTER_COLS = ['Geography', 'Gender']


# In[ ]:


#one hot encoding for categorical data by using pandas get_dummies method
df_dummies = pd.get_dummies(df, columns=FILTER_COLS)
df_dummies.head()
df_dummies.shape


# In[ ]:


#Normalize continuous data by using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_dummies[['Tenure', 'Balance', 'EstimatedSalary', 'Age', 'CreditScore', 'NumOfProducts']] = scaler.fit_transform(df_dummies[['Tenure', 'Balance', 'EstimatedSalary', 'Age', 'CreditScore', 'NumOfProducts']])
df_dummies.head()


# In[ ]:


#Link the value of the churn cases to the country of occurrence
df_dummies['Geography_France'] = df_dummies.Geography_France * df_dummies.Geo_ratio
df_dummies['Geography_Germany'] = df_dummies.Geography_Germany * df_dummies.Geo_ratio
df_dummies['Geography_Spain'] = df_dummies.Geography_Spain * df_dummies.Geo_ratio
df_dummies.drop(columns="Geo_ratio", inplace = True)

#Link the value of the churn cases to the gender of customer
df_dummies['Gender_Female'] = df_dummies.Gender_Female * df_dummies.Gen_ratio
df_dummies['Gender_Male'] = df_dummies.Gender_Male * df_dummies.Gen_ratio
df_dummies.drop(columns="Gen_ratio", inplace = True)
df_dummies.head()


# In[ ]:


#get corelation bar plot
plt.figure(figsize=(15,8))
df_dummies.corr()['Exited'].sort_values(ascending = False).plot(kind='bar')


# In[ ]:


#get corleation heatmap
f = plt.figure(figsize=(13, 10))
plt.matshow(df_dummies.corr(), fignum=f.number)
plt.xticks(range(df_dummies.shape[1]), df_dummies.columns, fontsize=10, rotation=90)
plt.yticks(range(df_dummies.shape[1]), df_dummies.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)


# In[ ]:


#divide dataset on X and y
X = df_dummies.drop(columns='Exited')
y= df.Exited.astype(int)
print(X.shape, y.shape)


# In[ ]:


#prepare 3 types of datasets
from sklearn.model_selection import train_test_split
x, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)

print(x.shape, y.shape)
print(x_test.shape, y_test.shape)
print(x_valid.shape, y_valid.shape)


# In[ ]:


#train LightGBM model
train_data = lightgbm.Dataset(x_train, label=y_train)
test_data = lightgbm.Dataset(x_test, label=y_test)
max_depth = 5
num_leaves = 2**max_depth

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': max_depth,
    'is_unbalance': 'true', #due to unbalanced dataset this option should be set as true
    'boosting': 'gbdt',
    'num_leaves': num_leaves,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0,
    'gpu_use_dp' : True
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=500,
                       early_stopping_rounds=25)


# In[ ]:


#make a LightGBM predict for validation set
y_pred = model.predict(x_valid)


# In[ ]:


#train XGB model, due to unbalanced dataset scale_pos_weight parameter should be equal class_1/class_0
from xgboost import XGBClassifier
model_xgb = XGBClassifier(learning_rate = 0.05, max_depth=max_depth, scale_pos_weight=0.25)
model_xgb.fit(x_train,y_train)


# In[ ]:


#make a XGB predict for validation set
y_xgb_pred = model_xgb.predict_proba(x_valid)
y_xgb_pred = y_xgb_pred[:,1:]


# In[ ]:


#train GBC classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate = 0.05, max_depth=max_depth, max_leaf_nodes = num_leaves)
gbc.fit(x_train, y_train)


# In[ ]:


#make a GBC predict for validation set
y_gbc_pred = gbc.predict_proba(x_valid)
y_gbc_pred = y_gbc_pred[:,1:]


# In[ ]:


get_ipython().system('pip install catboost')
#check documentation for any info: https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
#due to unbalanced dataset class_weights should be set as the list with a y ratio values of class_0 and class_1 
from catboost import CatBoostClassifier
catB = CatBoostClassifier(iterations=500,
                          learning_rate=0.05,
                          depth=max_depth,
                          class_weights = [0.8, 0.2])

catB.fit(x_train,
          y_train,
          verbose=False)


# In[ ]:


#make a catBoost predict for validation set
y_cat_pred = catB.predict_proba(x_valid)
y_cat_pred = y_cat_pred[:,1:]


# In[ ]:


#Binary Classification with Keras MLP Classifier
#Initializing model method
 def create_m(input_shape = x_train.shape[1]):
   model = Sequential()
   model.add(Dense(64, input_dim=input_shape, activation='relu'))
   model.add(Dropout(0.3))
   model.add(Dense(32, activation='relu'))
   model.add(Dropout(0.3))
   model.add(Dense(64, activation='relu'))
   model.add(Dropout(0.3))
   model.add(Dense(16, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))
   # Compile model
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   print(model.summary())
   return model


# In[ ]:


#fit model and predict the the valid set

from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping
#due to unbalanced dataset class_weights is counted by a class_weight method from sklearn library also early stopping method is on to prevent overfitting

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

print(class_weights)

es = EarlyStopping(monitor='accuracy', mode='max', min_delta=0.01, verbose=1, patience=15)

NN_model = create_m()

NN_model.fit(x_train, y_train, epochs=50, batch_size=128, class_weight=class_weights, callbacks=[es])

y_NN_pred = NN_model.predict(x_valid)


# In[ ]:


#print Auc Score for models
print('LGBM Roc Auc Score is equal: ', round(roc_auc_score(y_valid, y_pred),5))
print('XGB Roc Auc Score is equal: ', round(roc_auc_score(y_valid, y_xgb_pred),5))
print('GBC Roc Auc Score is equal: ', round(roc_auc_score(y_valid, y_gbc_pred),5))
print('CatBoost Roc Auc Score is equal: ', round(roc_auc_score(y_valid, y_cat_pred),5))
print('NN MLP Roc Auc Score is equal: ', round(roc_auc_score(y_valid, y_NN_pred),5))

#count values for plot
ns_probs = [0 for _ in range(len(y_valid))]
ns_fpr, ns_tpr, _ = roc_curve(y_valid, ns_probs)
fpr, tpr, thresholds = roc_curve(y_valid, y_pred)
xfpr, xtpr, xthresholds = roc_curve(y_valid, y_xgb_pred)
gfpr, gtpr, gthresholds = roc_curve(y_valid, y_gbc_pred)
cfpr, ctpr, cthresholds = roc_curve(y_valid, y_cat_pred)
nnfpr, nntpr, nntresholds = roc_curve(y_valid, y_NN_pred)

#plot the roc curve for the model
plt.figure(figsize=(10,5))
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, linestyle='--', label='LightGBM')
plt.plot(xfpr, xtpr, linestyle='--', label='XGBoost')
plt.plot(gfpr, gtpr, linestyle='--', label='GradientBoosting')
plt.plot(cfpr, ctpr, linestyle='--', label='CatBoost')
plt.plot(nnfpr, nntpr, linestyle='--', label='NeuralMLP')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

#set the size of the plot

#show the legend
plt.legend()
#show the plot
plt.show()

