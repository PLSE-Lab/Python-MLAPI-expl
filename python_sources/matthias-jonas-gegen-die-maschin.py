#!/usr/bin/env python
# coding: utf-8

# # Machine Learning 2020 Course Projects
# 
# ## Project Schedule
# 
# In this project, you will solve a real-life problem with a dataset. The project will be separated into two phases:
# 
# 27th May - 10th June: We will give you a training set with target values and a testing set without target. You predict the target of the testing set by trying different machine learning models and submit your best result to us and we will evaluate your results first time at the end of phase 1.
# 
# 9th June - 24th June: Students stand high in the leader board will briefly explain  their submission in a proseminar. We will also release some general advice to improve the result. You try to improve your prediction and submit final results in the end. We will again ask random group to present and show their implementation.
# The project shall be finished by a team of two people. Please find your teammate and REGISTER via [here](https://docs.google.com/forms/d/e/1FAIpQLSf4uAQwBkTbN12E0akQdxfXLgUQLObAVDRjqJHcNAUFwvRTsg/alreadyresponded).
# 
# The submission and evaluation is processed by [Kaggle](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71).  In order to submit, you need to create an account, please use your team name in the `team tag` on the [kaggle page](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71). Two people can submit as a team in Kaggle.
# 
# You can submit and test your result on the test set 2 times a day, you will be able to upload your predicted value in a CSV file and your result will be shown on a leaderboard. We collect data for grading at 22:00 on the **last day of each phase**. Please secure your best results before this time.
# 
# 

# ## Project Description
# 
# Car insurance companies are always trying to come up with a fair insurance plan for customers. They would like to offer a lower price to the careful and safe driver while the careless drivers who file claims in the past will pay more. In addition, more safe drivers mean that the company will spend less in operation. However, for new customers, it is difficult for the company to know who the safe driver is. As a result, if a company offers a low price, it bears a high risk of cost. If not, the company loses competitiveness and encourage new customers to choose its competitors.
# 
# 
# Your task is to create a machine learning model to mitigate this problem by identifying the safe drivers in new customers based on their profiles. The company then offers them a low price to boost safe customer acquirement and reduce risks of costs. We provide you with a dataset (train_set.csv) regarding the profile (columns starting with ps_*) of customers. You will be asked to predict whether a customer will file a claim (`target`) in the next year with the test_set.csv 
# 
# ~~You can find the dataset in the `project` folders in the jupyter hub.~~ We also upload dataset to Kaggle and will test your result and offer you a leaderboard in Kaggle. Please find them under the Data tag on the following page:
# https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71

# ## Phase 1: 26th May - 9th June
# 
# ### Data Description
# 
# In order to take a look at the data, you can use the `describe()` method. As you can see in the result, each row has a unique `id`. `Target` $\in \{0, 1\}$ is whether a user will file a claim in his insurance period. The rest of the 57 columns are features regarding customers' profiles. You might also notice that some of the features have minimum values of `-1`. This indicates that the actual value is missing or inaccessible.
# 

# In[ ]:


#Quick load dataset and check
import pandas as pd


# If you are not on google colab or you dont want to link your drive, skip this part.

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# Change TRAIN_SET_PATH and TEST_SET_PATH if your datasets are located different.

# In[ ]:


TRAIN_SET_PATH = "/content/drive/My Drive/colab/train_set.csv"
TEST_SET_PATH = "/content/drive/My Drive/colab/test_set.csv"
data_train = pd.read_csv(TRAIN_SET_PATH)
data_test = pd.read_csv(TEST_SET_PATH)


# ## Our Code
# 

# ### Feature Selection

# In[ ]:


n = 20

#Helper Functions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer


def getTrainingValues(properties=[], impute=True):
  df = pd.read_csv(TRAIN_SET_PATH)
  
  if(impute):
    simple_imp = SimpleImputer(missing_values=-1, strategy='mean')
    new_df = pd.DataFrame(simple_imp.fit_transform(df))
    new_df.columns = df.columns
    new_df.index = df.index
    df = new_df

  if len(properties) == 0:
    # separate target & id from values 
    properties = list(df.columns.values)
    properties.remove('target')
    properties.remove('id')

  X = df[properties]
  y = df['target'].astype(int)
  return X, y

def getTestValues(properties=[]):
  df = pd.read_csv(TEST_SET_PATH)
  orig = df

  # do we need to impute here??
  simple_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
  new_df = pd.DataFrame(simple_imp.fit_transform(df))
  new_df.columns = df.columns
  new_df.index = df.index
  df = new_df

  if len(properties) == 0:
    properties = list(df.columns.values)

  X = df[properties]
  return orig, X

def getNBestFeatures(n):
  X, y = getTrainingValues(impute = True)
  bestfeatures = SelectKBest(score_func=f_classif, k=n)
  fit = bestfeatures.fit(X,y)
  dfscores = pd.DataFrame(fit.scores_)
  dfcolumns = pd.DataFrame(X.columns)
  featureScores = pd.concat([dfcolumns,dfscores],axis=1)
  featureScores.columns = ['Specs','Score']
  # print(featureScores.nlargest(n,'Score'))
  return featureScores.nlargest(n,'Score')['Specs']

def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

def calc_class_weights(target):
  neg, pos = np.bincount(target)
  total = neg + pos
  print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
  weight_for_0 = (1 / neg)*(total)/2.0 
  weight_for_1 = (1 / pos)*(total)/2.0
  class_weight = {0: weight_for_0, 1: weight_for_1}
  print('Weight for class 0: {:.2f}'.format(weight_for_0))
  print('Weight for class 1: {:.2f}'.format(weight_for_1))
  return class_weight

getNBestFeatures(n)


# ### Model
# 
# 34        ps_car_13
# 33        ps_car_12
# 16    ps_ind_17_bin
# 27    ps_car_07_cat
# 19        ps_reg_02
# 5     ps_ind_06_bin
# 6     ps_ind_07_bin
# 24    ps_car_04_cat
# 23    ps_car_03_cat
# 20        ps_reg_03
# 22    ps_car_02_cat
# 4     ps_ind_05_cat
# 15    ps_ind_16_bin
# 36        ps_car_15
# 18        ps_reg_01
# 14        ps_ind_15
# 25    ps_car_05_cat
# 28    ps_car_08_cat
# 0         ps_ind_01
# 21    ps_car_01_cat

# In[ ]:


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import imblearn
from imblearn.over_sampling import SMOTE

X, y = getTrainingValues(getNBestFeatures(n), impute=False)

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(n,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#variate between epochs and batch size
model.fit(X_train, y_train, epochs=10, batch_size=64)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


# In[ ]:


from sklearn import metrics

def get_score(y_test,y_pred):
  score = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', zero_division='warn')
  print("binary f1 score is: ",score)
  score = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', zero_division='warn')
  print("weighted f1 score is: ",score)
  score = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro', zero_division='warn')
  print("macro f1 score is: ",score)
  score = metrics.accuracy_score(y_test,y_pred)
  print("total acc is: ",score)
  return score


# In[ ]:


# test our results on ones and zeros 

# treshold for the probability to predict 0/1
TRESH = 0.5

y_pred = model.predict(X_test)

y_pred[y_pred < TRESH]  = 0
y_pred[y_pred >= TRESH] = 1
get_score(y_test, y_pred)


X_pos, y_pos = extrac_one_label(X_test, y_test, 1)
X_neg, y_neg = extrac_one_label(X_test, y_test, 0)

y_negpred = model.predict(X_neg)
y_negpred[y_negpred < TRESH]  = 0
y_negpred[y_negpred >= TRESH] = 1
print("Accuracy of predicting 0:", sum(y_negpred==0)/len(y_negpred))
print("sum 0:", sum(y_negpred==0))


y_pospred = model.predict(X_pos)
y_pospred[y_pospred < TRESH]  = 0
y_pospred[y_pospred >= TRESH] = 1
print("Accuracy of predicting 1:", sum(y_pospred==1)/len(y_pospred))
print("sum 1:", sum(y_pospred==1))


# ### Submission
# 
# Please only submit the csv files with predicted outcome with its id and target [here](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71). Your column should only contain `0` and `1`.

# In[ ]:


data_real, data_test = getTestValues(getNBestFeatures(n))
y_target = model.predict(data_test)

print((y_target))


y_target[y_target < TRESH]  = 0
y_target[y_target >= TRESH] = 1


print(sum(y_target))

data_out = pd.DataFrame(data_real['id'].copy())
data_out.insert(1, "target", y_target.astype(int), True)
data_out.to_csv('submission.csv',index=False)
data_out

