#!/usr/bin/env python
# coding: utf-8

# # Necessary Imports (Must Run)

# In[ ]:


import numpy as np
import pandas as pd
import sklearn 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler


# # Loading Data and Data Manipulation (Must Run)

# Basic pandas dataset of train and test set respectively.

# In[ ]:


train_data_pd = pd.read_csv("train_set.csv", header=0)
test_data_pd = pd.read_csv("test_set.csv", header=0)


# As numpy arrays

# In[ ]:


train_data_np = train_data_pd.to_numpy()
test_data_np = test_data_pd.to_numpy()


# Numpy arrays in x and y form

# In[ ]:


train_data_x = train_data_np[:,2:]
train_data_y = train_data_np[:,1]
test_data_x = test_data_np[:,1:]


# ## Additional Functions (Must Run)

# The function n_hottest_hot(X,y,n,sel_k_best = None) does the following:
# *   Finds all categorical columns in **X**(name ending in cat)
# *   Converts them into one-hot encoded version and removes them from dataset **X**
# *   Takes the **n** columns from the one-hot encoded with the highest correlation to the target (**y**), and appends them to the dataset **X**
#   *   if sel_k_best is provided (list of columns), these columns are chosen instead of running correlation (so that test set will select the same as train set)
# *   Returns the resulting dataset and selected one-hot columns.
# 
# This function does not reduce the dataset, but converts categorical data into a more useable format, while discarding categories that do not have much impact on the target. **The function operates on pandas dataframes.**
# 
# 
# 

# In[ ]:


#returns the data where the categorical features are removed and the n best hot_encoded_cat_features are appended
def n_hottest_hot(X, y ,n, sel_k_best = None):
    cat_indices = [(X.columns[i].endswith("cat") or X.columns[i].endswith("cat")) for i in np.arange(X.shape[1])]
    cat_data = X.loc[:,cat_indices]
    #take only the one with less than 10 categories - otherwise it takes to much space...
    onehot = sklearn.preprocessing.OneHotEncoder(sparse = False)
    onehot.fit(cat_data)
    hot_cats = onehot.transform(cat_data)
    if sel_k_best is not None:
        k_best = hot_cats[:,sel_k_best.get_support()]
    else:
        k_Best = SelectKBest()
        sel_k_best = SelectKBest(k=n).fit(hot_cats , y)
        k_best = hot_cats[:,sel_k_best.get_support()]
    k_best = pd.DataFrame(k_best)
    not_cat_indices = [ not(X.columns[i].endswith("cat") or X.columns[i].endswith("cat")) for i in np.arange(X.shape[1])]
    result = X.loc[:,not_cat_indices]
    result = pd.concat([result, k_best], axis = 1)
    return result, sel_k_best


# n_best returns a selector array which selects the best **n** columns from a given dataset **data_x** in correlation to the target **data_y**. This function reduces the dataset through feature selection and **operates on numpy arrays.**

# In[ ]:


def n_best(data_x, data_y, n):
  select_k_best = SelectKBest(f_classif, k=n)
  select_k_best.fit_transform(data_x, data_y)
  return select_k_best.get_support()


# Another alternative is the function n_best_onehot. This function performs feature selection first, and then one-hot encodes any selected categorical features after. This process produces different results, but in theory they are worse, because feature selection is done before one-hot encoding, meaning categories are falsely assumed to be closer or further from each other based on their numerical standing before selection.

# In[ ]:


#Returns n_best_result-features with the categorical features one-hot-encoded
#data_x is a numpy array of features from data_pandas_x (i.e. train_data_x for train_data_pd)
#data_pandas_x is the original pandas dataset (i.e. train_data_pd)
#n_best_result is the result of calling the n_best function on the numpy array and corresponding y
#offset is the first column of the pandas dataset where features start (2 for training, 1 for testing)
def n_best_onehot(data_x, data_pandas_x, n_best_result, offset=2):
  cols = [i for i,v in enumerate(n_best_result) if v == True]

  manuals = []
  removefromcols = []

  for i in cols:
    if(data_pandas_x.columns[i+offset].endswith("cat")):
      manuals.append(pd.get_dummies(data_pandas_x[data_pandas_x.columns[i+offset]]))
      removefromcols.append(i)

  for i in removefromcols:
    cols.remove(i)

  data_x_new = data_x[:,cols]

  for m in manuals:
    m_np = m.to_numpy()
    data_x_new = np.append(data_x_new, m_np, axis=1)

  return data_x_new


# # Data Visualization (Optional)

# First we want to have an overview of all columns of the data.

# In[ ]:


with pd.option_context('display.max_columns', 60):
    #print(df.describe(include='all'))
    print(train_data_pd.describe(include='all'))


# Next we want to generate some violin plots to get a feeling of which categories might be of high value... We start with the categorical features:

# In[ ]:


def violin(X,y):       
  data = pd.concat([y,X],axis=1)
  data = pd.melt(data,id_vars="target",
                      var_name="features",
                      value_name='value')
  plt.figure(figsize=(20,10))
  sns.violinplot(x="features", y="value", hue="target", data=data,split=True, inner="quart")
  plt.xticks(rotation=90)


# In[ ]:


cat_indices = [(train_data_pd.columns[i].endswith("cat") or train_data_pd.columns[i].endswith("cat")) for i in np.arange(train_data_pd.shape[1])]
cat_data = train_data_pd.loc[:,cat_indices]

violin(cat_data.loc[:,(cat_data.std() < 1)] ,train_data_pd["target"] )
violin(cat_data.loc[:,(cat_data.std() >= 1)] ,train_data_pd["target"] )


# We can clearly see that ps_car_11_cat has over 100 categories - thats a lot and probably would take too much space to onehot-encode this one...

# Then we check out the binary data:

# In[ ]:


bin_indices = [(train_data_pd.columns[i].endswith("bin") or train_data_pd.columns[i].endswith("bin")) for i in np.arange(train_data_pd.shape[1])]
bin_data = train_data_pd.loc[:,bin_indices]

violin(bin_data, train_data_pd["target"])


# And finally the rest of the data

# In[ ]:


rest_indices = [not(bin_indices[i] or cat_indices[i]) for i in np.arange(train_data_pd.shape[1])]
rest_data = train_data_pd.loc[:,rest_indices]
rest_data = rest_data.drop(["id", "target"], axis = 1)

violin( rest_data.loc[:,(rest_data.std() > 1)] ,train_data_pd["target"] )
violin( rest_data.loc[:,(rest_data.std() <= 1)] ,train_data_pd["target"] )


# Now lets have a look at the correlation and whether we can drop some features due to high correlation

# In[ ]:


correlation = train_data_pd.corr()

plt.figure(dpi=1200)
plt.figure(figsize=(40,20))
sns.heatmap(correlation, annot=True)
#plt.savefig("correlation.svg")
plt.show()


# Highest correlation is 0.89 where we could still lose some information when deleting one of the two features so we keep all.

# # Best Implementation: Neural Network
# The neural network implementation managed to score a Macro-F1 score of **0.52942** on the test set, which was our best submission. The neural network is not optimized. It is likely possible to score over 0.53 with a better implementation.
# 
# **Note that the Neural Network may produce better or worse results than recorded, since the process is not really deterministic and EarlyStopping might stop the learning process at bad epochs.**

# The final submitted implementation first one-hot encodes the categorical data and takes the 50 most impactful categories into the dataset. This data is then scaled using a standard scaler. From the resulting 93 columns total, the 25 best correlating features with the training set are selected. A neural network with one dense hidden layer of 10 neurons is then trained using a batch size of 256 and class weight 1:14.25, for 500 epochs or until loss stops decreasing for 20 epochs in a row. This neural network is then used to classify test data.

# In this first step, we separate X and Y from the pandas dataframe, and replace the categorical columns with the 50 one-hot encoded columns of highest correlation in the training data. We store the selected one-hot columns in selected_cols.

# In[ ]:


#get x and y of train set in pandas form
pandas_x = train_data_pd.drop(["target", "id"], axis=1)
pandas_y = train_data_pd["target"]

#get dataset with one-hot-encoding
onehot_pandas_x, selected_cols = n_hottest_hot(pandas_x, pandas_y, 50)
print(onehot_pandas_x.describe)

#convert to numpy array
onehot_np_x = onehot_pandas_x.to_numpy()

print(onehot_np_x.shape)


# In a second step, we scale the training data using the StandardScaler, then select the 25 features with overall highest correlation out of a mix of basic features and one-hot encoded ones.

# In[ ]:


#standardize data
scaler = StandardScaler().fit(onehot_np_x)
onehot_scaled = scaler.transform(onehot_np_x)
best_feats = n_best(onehot_scaled, train_data_y, 25)
onehot_scaled = onehot_scaled[:,best_feats]


# The EarlyStopping class from Keras will monitor the loss for us. It gets called as a callback after each iteration, and will stop the learning early if loss has not decreased for a set number of episodes (here: 20)

# In[ ]:


# Create instance of EarlyStopping to stop when loss is no longer decreased for 5 epochs
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)


# Our Neural Network has 1 hidden layer with 10 neurons, using relu as an activation function. The final activation function for classification is sigmoid, to act as a "chance" of belonging to either class.

# In[ ]:


model = Sequential()
model.add(Dense(10, input_dim=25, activation='relu'))
model.add(Dense(1, input_dim=10, activation='sigmoid'))
#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], weighted_metrics=["accuracy"])


# We fit the model on the prepared training data

# In[ ]:


#fit model on large epoch sizes to quickly converge (500 epochs or until no more loss reduction for 20 epochs)
model.fit(onehot_scaled, train_data_y, epochs=500, batch_size=256, class_weight={0:1,1:14.25}, callbacks=[es])


# To predict the test set, we select the same columns and scaling as for the test set, and feed it to the neural net. We make sure to use selected_cols to select the same one-hot columns, and the 25 best_feats we calculated earlier.

# In[ ]:


# predict test set
pd_test_x = test_data_pd.drop(["id"], axis=1)
#get dataset with one-hot-encoding
onehot_test, selected_cols = n_hottest_hot(pd_test_x, pandas_y, 50, sel_k_best=selected_cols)

#transform as with the train set
onehot_test = scaler.transform(onehot_test)
onehot_test = onehot_test[:,best_feats]

#predict, round the prediction to 0 or 1
preds = model.predict(onehot_test)
preds = np.rint(preds)
preds = preds.astype(int)

# how many are 0 and 1 class
amount_0 = sum(preds == 0) / len(preds)
amount_1 = sum(preds == 1) / len(preds)

print("Ratio of 0 classified data:", amount_0)
print("Ratio of 1 classified data:", amount_1)


# Output Data to File

# In[ ]:


data_out = pd.DataFrame(test_data_pd['id'].copy())
data_out.insert(1, "target", preds, True)
data_out.to_csv('submission.csv',index=False)


# # Other Implementation: Logistic Regression
# Logistic Regression on a somewhat optimized dataset managed to achieve a Macro-F1 Score of **0.52771**.
# 
# 
# *   The 14 best features are extracted (This value was determined by testing different feature amounts and comparing macro score in a test subset of the training data)
# *   The columns are selected from the x data, and categorial columns are encoded by a one-hot encoding
# *   A standard scaler is used to standardize features
# *   The model is trained with following non-default values, which were determined by a grid search over possible hyperparameters:
#     * C = 1.0
#     * class_weight:{0:1, 1:14}
#     * solver: 'liblinear'
#     * penalty: 'l1'
#     * max_iter: 10000
# *   The same scaling and feature selection is applied to the test set
# *   The test set is predicted
# *   The test set is output to a file.
# 
# 

# In[ ]:


#get standard scaler
scaler = StandardScaler()

#get n best columns (here, 14) from training set
n_best_res = n_best(train_data_x, train_data_y, 14)

#fit the training set to the one hot encoding of these columns
train_x = n_best_onehot(train_data_x, train_data_pd, n_best_res, 2)

#fit scaler and scale
scaler.fit(train_x, train_data_y)
train_x = scaler.transform(train_x)

#train model
clf = LogisticRegression(C=1.0, class_weight={0:1, 1:14}, dual=False,
                      fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                      max_iter=10000, multi_class='auto', n_jobs=None,
                      penalty='l1', random_state=None, solver='liblinear',
                      tol=0.0001, verbose=0, warm_start=False)
clf.fit(train_x, train_data_y)

#take the same columns and encoding for the test set and scale the same way
x_test = n_best_onehot(test_data_x, test_data_pd, n_best_res, 1)
x_test = scaler.transform(x_test)

# predict test set
preds = clf.predict(x_test)
preds = np.rint(preds)
preds = preds.astype(int)

# how many are 0 and 1 class
amount_0 = sum(preds == 0) / len(preds)
amount_1 = sum(preds == 1) / len(preds)

print("Ratio of 0 classified data:", amount_0)
print("Ratio of 1 classified data:", amount_1)

#create file
data_out = pd.DataFrame(test_data_pd['id'].copy())
data_out.insert(1, "target", preds, True)
data_out.to_csv('submission.csv',index=False)

