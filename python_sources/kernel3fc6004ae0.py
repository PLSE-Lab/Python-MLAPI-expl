#!/usr/bin/env python
# coding: utf-8

# ### Tags
# - Kaggle Problem [https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star]
# - All numerical Features.
# - Binary Classification.
# - Data Size around 771 KB.
# - Data Downloaded : in Personal Projects , ML , DataSet
# - No seperate data set for training and testing
# - Compare Random Forest, SVC and TF DNN Classifier.
# - Effect of normalized data on different models
# 

# In[2]:


##Data Understanding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from IPython.display import Image

##Data Normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

## Modeling
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf

## Model Evaludation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


# In[3]:


print(tf.__version__)


# In[4]:



data = pd.read_csv("../input/pulsar_stars.csv")


# #### Data MetaData

# - Number of rows around 17k and number of features is 8
# - All the Fields are numrical
# - No Null Values

# In[5]:


print(data.shape)


# In[6]:


data.head()


# In[7]:


def panda_null_col_evaluation(data):
    return data.columns[data.isna().any()].tolist()
print(panda_null_col_evaluation(data))


# #### Data Understanding

# #### Importtant things related to data from metadata.
# - Ratio of target is not evenly distributed.
# 

# In[8]:


data.corr()


# ##### Columns which have high impact on target 
# - Excess kurtosis of the integrated profile
# - Skewness of the integrated profile
# - Mean of the integrated profile
# - Standard deviation of the DM-SNR curve	

# In[9]:


data.describe()


# In[10]:


# distribution of taeget class
print('Target Class Count', data.target_class.value_counts())

ration = ((int)(data.target_class.value_counts()[0])) / data.shape[0]
print('Negative Class Ration ', ration)
print('Positive Class Ration ', ((int)(data.target_class.value_counts()[1])) / data.shape[0])


# In[11]:


plt.hist(data.target_class, bins=2, rwidth=0.8)


# In[13]:


data.loc[:, data.columns !=  'target_class'].hist(figsize= [20,20], layout=[4,2])


# In[14]:


sns.pairplot(data=data,
             palette="husl",
             hue="target_class",
             vars=[" Mean of the integrated profile",
                   " Excess kurtosis of the integrated profile",
                   " Skewness of the integrated profile",
                   " Mean of the DM-SNR curve",
                   " Excess kurtosis of the DM-SNR curve",
                   " Skewness of the DM-SNR curve"])
plt.suptitle("PairPlot of Data Without Std. Dev. Fields",fontsize=20)

plt.tight_layout()
plt.show() 


# In[15]:


plt.figure(figsize=(18,20))

plt.subplot(4,2,1)
sns.violinplot(data=data,y=" Mean of the integrated profile",x="target_class")

plt.subplot(4,2,2)
sns.violinplot(data=data,y=" Mean of the DM-SNR curve",x="target_class")

plt.subplot(4,2,3)
sns.violinplot(data=data,y=" Standard deviation of the integrated profile",x="target_class")

plt.subplot(4,2,4)
sns.violinplot(data=data,y=" Standard deviation of the DM-SNR curve",x="target_class")

plt.subplot(4,2,5)
sns.violinplot(data=data,y=" Excess kurtosis of the integrated profile",x="target_class")

plt.subplot(4,2,6)
sns.violinplot(data=data,y=" Skewness of the integrated profile",x="target_class")

plt.subplot(4,2,7)
sns.violinplot(data=data,y=" Excess kurtosis of the DM-SNR curve",x="target_class")

plt.subplot(4,2,8)
sns.violinplot(data=data,y=" Skewness of the DM-SNR curve",x="target_class")


plt.suptitle("ViolinPlot",fontsize=30)

plt.show()


# ### Data Seperation
# - Since we dont have training and validation set we will create the same from dataset
# - To distribute the target class correctly , we will first devide the data by target class and then take 20% from each subset.
# 

# In[16]:


data_negative = data.loc[data['target_class'] == 0]
data_positive = data.loc[data['target_class'] == 1]
print('Data Negative shape ', data_negative.shape)
print('Data Negative shape ', data_positive.shape)


# In[17]:


##Shuffle Data Before Splitting
def shuffle(data_frame):
     return data_frame.reindex(np.random.permutation(data_frame.index))

data_negative = shuffle(data_negative)
data_positive = shuffle(data_positive)


# In[18]:


## Split Data
def split_training_and_test(data_frame, training_percentage):
    training_number = data_frame.shape[0] * training_percentage / 100
    test_number = data_frame.shape[0] - training_number
    return data_frame.head(int(training_number)), data_frame.tail(int(test_number))


data_negative_train, data_negative_val = split_training_and_test(data_negative, 80)
data_positive_train, data_positive_val = split_training_and_test(data_positive, 80)


# In[19]:


print('Data Positive train', data_positive_train.shape)
print('Data Positive val', data_positive_val.shape)

print('Data Negative train', data_negative_train.shape)
print('Data Negative val', data_negative_val.shape)


# In[20]:


data_train = shuffle(pd.concat([data_positive_train, data_negative_train]))
data_val = shuffle(pd.concat([data_positive_val, data_negative_val]))

print('Training Set Shape', data_train.shape)
print('Validation Set Shape', data_val.shape)


# In[21]:


def seperate_feature_and_target(data, feature_name):
    return data_train.loc[:, data.columns !=  feature_name], data_train[feature_name] 


# In[22]:


X_train, y_train = seperate_feature_and_target(data_train, 'target_class')
X_val, y_val = seperate_feature_and_target(data_val, 'target_class')


# In[23]:


min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()


# In[24]:



X_train_min_max = min_max_scaler.fit_transform(X_train)
X_val_min_max = min_max_scaler.transform(X_val)

X_train_std = standard_scaler.fit_transform(X_train)
X_val_std = standard_scaler.transform(X_val)


# #### Model Evaluation 
# 
# We will try following models for binary classification 
# 
# - Random Forest With and Without Normalization Normalization
# - Support Vector Machine With and Without Normalization
# - TensorFlow DNN Classifier with and without normalization. [Also compare different optimizers with normalized data]

# ##### Random Forest Model

# In[25]:


def evaluate(model, params, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(model, params, n_jobs=-1, cv=3)
    grid_search.fit(X_train, y_train)
    print('Best Params', grid_search.best_params_)
    print("Best Score", grid_search.best_score_)
    prediction = grid_search.best_estimator_.predict(X_test)
    print('Accuracy Score', accuracy_score(y_test, prediction))
    print('Classification  Report', classification_report(y_test, prediction))    


# In[26]:


get_ipython().run_cell_magic('time', '', "parameters = {'max_depth':[11,12,13,None], 'min_samples_split':[20,22,25,30], 'criterion':['gini'],\n             'n_estimators':[8,9,10,11]}\nmodel = RandomForestClassifier(random_state=42)\nevaluate(model, parameters, X_train, y_train, X_val, y_val)")


# In[27]:


get_ipython().run_cell_magic('time', '', "parameters = {'max_depth':[11,12,13,None], 'min_samples_split':[20,22,25,30], 'criterion':['gini'],\n             'n_estimators':[8,9,10,11]}\nmodel = RandomForestClassifier(random_state=42)\nevaluate(model, parameters, X_train_min_max, y_train, X_val_min_max, y_val)")


# In[28]:


get_ipython().run_cell_magic('time', '', "parameters = {'max_depth':[11,12,13,None], 'min_samples_split':[20,22,25,30], 'criterion':['gini'],\n             'n_estimators':[8,9,10,11]}\nmodel = RandomForestClassifier(random_state=42)\nevaluate(model, parameters, X_train_std, y_train, X_val_std, y_val)")


# #### Support Vector Machine

# In[29]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "kernel":[ \'linear\',\'rbf\']\n    }\nmodel = SVC(random_state=42)\nevaluate(model, parameters, X_train, y_train, X_val, y_val)')


# In[30]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "kernel":[ \'linear\',\'rbf\']\n    }\nmodel = SVC(random_state=42)\nevaluate(model, parameters, X_train_min_max, y_train, X_val_min_max, y_val)')


# In[31]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "kernel":[ \'linear\',\'rbf\'],\n    "degree":[2]\n    }\nmodel = SVC(random_state=42)\nevaluate(model, parameters, X_train_std, y_train, X_val_std, y_val)')


# #### Tensor Flow DNN Classifer Without Normalization

# In[32]:


hidden_units = [3]
batch_size = 10
steps = 2000
model_dir = "/Users/amitjain/personalProjects/Machine-Learning/Complete_Guide_Self/Simple Classification/Complete Practise Of Classification/tf_models_pred"


# In[33]:


t0 = time.time()
run = "/without_normalized_1"
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = hidden_units, n_classes=2, feature_columns= feature_cols,
                                        model_dir=model_dir + run)
dnn_clf.fit(X_train, y_train, batch_size = batch_size, steps = steps)
t1 = time.time()
print("Training Completed in ", t1 - t0)


# In[34]:


y_pred = dnn_clf.predict(X_val)
predictions = list(y_pred)


# In[35]:


print('Without normalization : time taken ', t1-t0)
print('Accuracy Score', accuracy_score(y_val, predictions))
print('Classification  Report', classification_report(y_val, predictions))  


# #### Tensor Flow DNN Classifer With Normalization

# In[36]:


hidden_units = [3]
batch_size = 10
steps = 2000
model_dir = "/Users/amitjain/personalProjects/Machine-Learning/Complete_Guide_Self/Simple Classification/Complete Practise Of Classification/tf_models_pred"


# In[37]:


t0 = time.time()
run = "/with_normalized"
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = hidden_units, n_classes=2, feature_columns= feature_cols,
                                        model_dir=model_dir + run)
dnn_clf.fit(X_train_std, y_train, batch_size = batch_size, steps = steps)
t1 = time.time()
print("Training Completed in ", t1 - t0)


# In[38]:


y_pred = dnn_clf.predict(X_val_std)
predictions = list(y_pred)


# In[39]:


print('Without normalization : time taken ', t1-t0)
print('Accuracy Score', accuracy_score(y_val, predictions))
print('Classification  Report', classification_report(y_val, predictions))  


# In[40]:


hidden_units = [3]
batch_size = 120
steps = 4000
model_dir = "/Users/amitjain/personalProjects/Machine-Learning/Complete_Guide_Self/Simple Classification/Complete Practise Of Classification/tf_models_pred"
optimizer = tf.train.AdamOptimizer( learning_rate=0.0009,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08)


# In[41]:


t0 = time.time()
run = "/final"
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = hidden_units, n_classes=2, feature_columns= feature_cols,
                                        model_dir=model_dir + run,
                                        optimizer=optimizer)
dnn_clf.fit(X_train_std, y_train, batch_size = batch_size, steps = steps)
t1 = time.time()
print("Training Completed in ", t1 - t0)


# In[42]:


y_pred = dnn_clf.predict(X_val_std)
predictions = list(y_pred)


# In[43]:


print('Without normalization : time taken ', t1-t0)
print('Accuracy Score', accuracy_score(y_val, predictions))
print('Classification  Report', classification_report(y_val, predictions))  


# ### Things To be Learn From This NoteBook

# - Random Forest in total is very fast , completed in 5s. Normalization didn't had much impact in the performance , reduced the time 5-10%
# - SVC in generall have taken more time compared to Random Forest Tree , performance is little less than RFT. Best kernel came to be 'linear'. Poly kernel increase the execution time way too much, should check when will poly kernels should e used. For SVM execution time was very effective after normalization.
# - Tensorflow [DNNClassifier] , initial model normalization help with seed and evaluation both. 
# - In our case Adam Optimizer was better tha Gradient for Various comination of hyper parameter.
# - In this particular case , simple Model [single layer ] was more than enough. batch size had good effect on intial training curve.
