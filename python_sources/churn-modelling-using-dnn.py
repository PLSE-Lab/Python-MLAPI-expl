#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
# calculate accuracy measures and confusion matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, auc
warnings.filterwarnings('ignore')


# In[ ]:


import tensorflow as tf
print(tf.__version__)


# ### EDA & Data Preprocessing

# #### 1. Read the dataset

# In[ ]:



#importing the dataset, set RowNumber as index
#load the csv file and make the data frame
df = pd.read_csv('/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv',index_col='RowNumber')


# In[ ]:


df.shape # 10,000 rows, 13 columns


# In[ ]:


df.head(2) #Exited is target column


# In[ ]:


#Check datatypes
df.info()


# Surname, Gender and Gepgraphy are Object type

# In[ ]:


#Check for missing values
df.isna().sum()


#  There is no missing values

# In[ ]:


#look at distribution of exited and non-exited customers


# In[ ]:


sns.countplot(x="Exited", data=df)


# Data set has has only around 2000 exited customers and about 8000 Customers are still with Bank- it has bias towards existing customers.

# In[ ]:


sns.countplot(x="Gender", data=df)


# Bank has bout 4500 female customers and 5500 male customers

# In[ ]:


sns.countplot(x="Geography", data=df)


# Most of the Customers are from France,  Customers from spain and Genrmany are about half in numbers of France

# In[ ]:


sns.countplot(x="Exited", hue="Gender", data=df)


# Above plot says that female customers have higher propensity to exit the Bank

# In[ ]:


sns.countplot(x="Exited", hue="Geography", data=df)


# Customers from Germany have highest propensity to to exit the Bank

# In[ ]:


#Lets Check Distribution of exited/non-exited Customers as per the age


# In[ ]:


sns.distplot(df['Age'][df['Exited']==0],color='blue',label='non-exited')
sns.distplot(df['Age'][df['Exited']==1],color='red',label='exited')
plt.show()


# Age distribution of customers who exited bank is normally distributed while those who stays with bank is right skewed
# indicating that most of the existing customers of bank are lower than 50 years of age. This may also indicate that
# old age customers have exited the bank.

# #### 2. Drop the columns which are unique for all users like IDs & 3. Distinguish the feature and target set

# In[ ]:


df.columns


# In[ ]:


# Convert data into feature and Target set. Also CustomerId and Surname will not contribute to model building
#hence we wil drop these 2 colmns as well
X=df.drop(labels=['CustomerId','Surname','Exited'], axis=1) # Feature Set
y=df['Exited'] # Target set


# In[ ]:


X.info()


# In[ ]:


# Geography and gender are object type, we will convert this into one hot encoding


# In[ ]:


X= pd.get_dummies(X)


# In[ ]:


X.info()


# Object columns- Geography and Genders have been converted to one hot encoded columns

# In[ ]:


#Lets Check first few rows of feature set


# In[ ]:


X.head()


# #### 4. Divide the data set into training and test sets

# In[ ]:


from sklearn.model_selection import train_test_split
#test train split
test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[ ]:


#hCheck Shape of test/trainset
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# #### 5. Normalize the train and test data 
# 
# a)Normalise Following features using standard scaler: CreditScore,Age, tenure,Balance,NumOfProducts,EstimatedSalary as these  have running/continuous values
# 
# b)We will not normalise following features as they have discrete values either 0 or 1: HasCrCard,IsActiveMember,Geography_France,Geography_Germany,Geography_Spain,Gender_Female,Gender_Male
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[ ]:


X_train[['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']].head(2)


# In[ ]:


X_train.head(2)


# In[ ]:


scaler.fit(X_train[['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']])


# In[ ]:


X_train_scaled=scaler.transform(X_train[['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']])


# In[ ]:


# Transform test set on the same fit as train set
X_test_scaled=scaler.transform(X_test[['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']])


# ##### Following step puts back scaled data into the dataframe for the columns which  have been scaled while keeping other data intact

# In[ ]:


# Put back scaled data into the dataframe for the columns which  have been scaled while keeping other data intact
X_train[['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']]=X_train_scaled


# In[ ]:


X_train.head(2)


# In[ ]:


X_test[['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']]=X_test_scaled


# In[ ]:


X_test.head(2)


# ##### Convert Data into Numpy arrays

# In[ ]:


# Convert Data into Numpy arrays
X_train_array=np.array(X_train)
X_test_array=np.array(X_test)
y_train_array=np.array(y_train)
y_test_array=np.array(y_test)


# In[ ]:



X_train_array.shape,X_test_array.shape,y_train_array.shape,y_test_array.shape#check shapes of array


# ### MODEL BUILDING

# #### 6. Initialize & build the model (Basic Model with 2 hidden layers)

# In[ ]:


# Initialize Sequential model
model = tf.keras.models.Sequential()


# Add Input layer to the model
model.add(tf.keras.Input(shape=(13,))) # 13 Features

# Batch Normalization Layer
model.add(tf.keras.layers.BatchNormalization())

# Hidden layers
model.add(tf.keras.layers.Dense(13, activation='relu', name='Layer_1'))
model.add(tf.keras.layers.Dense(10, activation='relu', name='Layer_2'))

#Output layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='Output'))


# ##### Compile Model

# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ##### Summarise Model

# In[ ]:


model.summary()


# #### Fit Model & Prediction

# In[ ]:


model.fit(X_train_array, y_train_array, validation_data=(X_test_array, y_test_array), epochs=150,
          batch_size = 32)


# #### 5.Predict the results using 0.5 as a threshold

# In[ ]:


model.predict(X_test_array)[:5] # Observe first 5 probabilities


# In[ ]:


th=0.5 # Threshold
y_test_preds = np.where(model.predict(X_test_array) > th, 1, 0)


# In[ ]:


y_test_preds[:5] # Observe First 5 predictions


# #### Confusion Matrix

# In[ ]:


# Confusion matrix with optimal Threshold on test set
metrics.confusion_matrix(y_test, y_test_preds)


# In[ ]:


print('Test Metrics at 0.5 Threshold with basic DNN model\n')
Test_Metrics_Basic_DNN=pd.DataFrame(data=[accuracy_score(y_test, y_test_preds), 
                   recall_score(y_test, y_test_preds), 
                   precision_score(y_test, y_test_preds),
                   f1_score(y_test, y_test_preds)], columns=['Basic DNN'],
             index=["accuracy", "recall", "precision", "f1_score"])
print(Test_Metrics_Basic_DNN)


# ### MODEL TUNING

# #### A) With 3 Dense Layer

# In[ ]:


# Initialize Sequential model
model = tf.keras.models.Sequential()


# Add Input layer to the model
model.add(tf.keras.Input(shape=(13,))) # 13 Features

# Batch Normalization Layer
model.add(tf.keras.layers.BatchNormalization())

# Hidden layers
model.add(tf.keras.layers.Dense(13, activation='relu', name='Layer_1'))
model.add(tf.keras.layers.Dense(13, activation='relu', name='Layer_2'))
model.add(tf.keras.layers.Dense(10, activation='relu', name='Layer_3'))
#Output layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='Output'))


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X_train_array, y_train_array, validation_data=(X_test_array, y_test_array), epochs=150,
          batch_size = 32)


# In[ ]:


th=0.5 # Threshold
y_test_preds = np.where(model.predict(X_test_array) > th, 1, 0)


# In[ ]:


print('Test Metrics at 0.5 Threshold with 3 Hidden layer DNN model\n')
Test_Metrics_3_HiddenLayer_DNN=pd.DataFrame(data=[accuracy_score(y_test, y_test_preds), 
                   recall_score(y_test_array, y_test_preds), 
                   precision_score(y_test_array, y_test_preds),
                   f1_score(y_test_array, y_test_preds)], columns=['3 Hidden Layer DNN'],
             index=["accuracy", "recall", "precision", "f1_score"])
print(Test_Metrics_3_HiddenLayer_DNN)


# In[ ]:


# Confusion matrix with optimal Threshold on test set
metrics.confusion_matrix(y_test_array, y_test_preds)


# ##### Not much improvement in accuracy, precision has improved, recall has gone down,overall accuracy is almost same

# #### B) With Batch normalisation after each hidden layer

# In[ ]:


# Initialize Sequential model
model = tf.keras.models.Sequential()


# Add Input layer to the model
model.add(tf.keras.Input(shape=(13,))) # 13 Features

# Batch Normalization Layer
#model.add(tf.keras.layers.BatchNormalization())

# Hidden layers
model.add(tf.keras.layers.Dense(13, activation='relu', name='Layer_1'))
# Batch Normalization Layer
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(13, activation='relu', name='Layer_2'))
# Batch Normalization Layer
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(10, activation='relu', name='Layer_3'))
# Batch Normalization Layer
model.add(tf.keras.layers.BatchNormalization())
#Output layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='Output'))


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train_array, y_train_array, validation_data=(X_test_array, y_test_array), epochs=150,
          batch_size = 32)


# In[ ]:


th=0.5 # Threshold
y_test_preds = np.where(model.predict(X_test_array) > th, 1, 0)


# In[ ]:


print('Test Metrics at 0.5 Threshold with  Batch Norm after each hidden layer DNN model\n')
Test_Metrics_BatchNorm=pd.DataFrame(data=[accuracy_score(y_test, y_test_preds), 
                   recall_score(y_test_array, y_test_preds), 
                   precision_score(y_test_array, y_test_preds),
                   f1_score(y_test_array, y_test_preds)], columns=['BatchNorm Hidden layers'],
             index=["accuracy", "recall", "precision", "f1_score"])
print(Test_Metrics_BatchNorm)


# In[ ]:


# Confusion matrix with optimal Threshold on test set
metrics.confusion_matrix(y_test_array, y_test_preds)


# #### Overall Accuracy, recall and F1 score has improved after adding Batch Normalisation after each hidden layer

# #### C) Using Weight and Bias initializer

# In[ ]:


from keras import initializers


# In[ ]:


# Initialize Sequential model
model = tf.keras.models.Sequential()


# Add Input layer to the model
model.add(tf.keras.Input(shape=(13,))) # 13 Features


# Hidden layers
model.add(tf.keras.layers.Dense(13, kernel_initializer='he_normal', bias_initializer='Ones',activation='relu', name='Layer_1'))
# Batch Normalization Layer
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(13, kernel_initializer='he_normal',bias_initializer='Ones',activation='relu', name='Layer_2'))
# Batch Normalization Layer
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(10,kernel_initializer='he_normal',bias_initializer='Ones', activation='relu', name='Layer_3'))
# Batch Normalization Layer
model.add(tf.keras.layers.BatchNormalization())
#Output layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='Output'))


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X_train_array, y_train_array, validation_data=(X_test_array, y_test_array), epochs=50,
          batch_size = 32)


# In[ ]:


th=0.5 # Threshold
y_test_preds = np.where(model.predict(X_test_array) > th, 1, 0)


# In[ ]:


print('Test Metrics at 0.5 Threshold withv Weight and Bias initialization &  Batch Norm after each hidden layer DNN model\n')
Test_Metrics_Weight_Init=pd.DataFrame(data=[accuracy_score(y_test, y_test_preds), 
                   recall_score(y_test_array, y_test_preds), 
                   precision_score(y_test_array, y_test_preds),
                   f1_score(y_test_array, y_test_preds)], columns=['Weight Initialize'],
             index=["accuracy", "recall", "precision", "f1_score"])
print(Test_Metrics_Weight_Init)


# In[ ]:


# Confusion matrix with optimal Threshold on test set
metrics.confusion_matrix(y_test_array, y_test_preds)


# #### Accuracy has dropped

# #### D) Apply Dropout

# In[ ]:


# Initialize Sequential model
model = tf.keras.models.Sequential()


# Add Input layer to the model
model.add(tf.keras.Input(shape=(13,))) # 13 Features


# Hidden layers
model.add(tf.keras.layers.Dense(13, activation='relu', name='Layer_1'))
model.add(tf.keras.layers.Dense(13, activation='relu', name='Layer_2'))

# Dropout layer
model.add(tf.keras.layers.Dropout(0.5))

# Hidden layers
model.add(tf.keras.layers.Dense(10, activation='relu', name='Layer_3'))


# Dropout layer
model.add(tf.keras.layers.Dropout(0.3))

#Output layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='Output'))


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train_array, y_train_array, validation_data=(X_test_array, y_test_array), epochs=100,
          batch_size = 32, verbose=0)


# In[ ]:


th=0.5 # Threshold
y_test_preds = np.where(model.predict(X_test_array) > th, 1, 0)


# In[ ]:


print('Test Metrics at 0.5 Threshold Dropout DNN model\n')
Test_Metrics_DropOut=pd.DataFrame(data=[accuracy_score(y_test, y_test_preds), 
                   recall_score(y_test_array, y_test_preds), 
                   precision_score(y_test_array, y_test_preds),
                   f1_score(y_test_array, y_test_preds)], columns=['DropOut'],
             index=["accuracy", "recall", "precision", "f1_score"])
print(Test_Metrics_DropOut)


# In[ ]:


# Confusion matrix with optimal Threshold on test set
metrics.confusion_matrix(y_test_array, y_test_preds)


# #### Still Accuracy has not improved 

# ### MODEL COMPARISON

# In[ ]:


Model_Comparison_df=Test_Metrics_Basic_DNN
Model_Comparison_df['3 Hidden Layer DNN']=Test_Metrics_3_HiddenLayer_DNN['3 Hidden Layer DNN']
Model_Comparison_df['BatchNorm Hidden layers']=Test_Metrics_BatchNorm['BatchNorm Hidden layers']
Model_Comparison_df['Weight Initialize']=Test_Metrics_Weight_Init['Weight Initialize']
Model_Comparison_df['DropOut']=Test_Metrics_DropOut['DropOut']
#Model_Comparison_df['Naive Bayes']=Naive_Bayes_metrics['Naive Bayes']
Model_Comparison_df


# ##### Among the models tried above Model with bath normalization after each hidden layer gives best Accurracy and F1 score
