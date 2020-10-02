#!/usr/bin/env python
# coding: utf-8

# # Autoencoder Architecture
# 
# ## Introduction
# Autoencoder is a kind of Deep Learning architectures. Autoencoder architecture encompasses two sub-systems as encoder and decoder. Both these sub-systems are made up of independent Neural Network with a defined set of layers and activation functions. The fundamental characteristic feature of Autoencoder architecture is extracting the latent(hidden) data points from the given dataset. 
# 
# This notebook is created out of inspiration from the post on [Geekforgeek](https://www.geeksforgeeks.org/ml-classifying-data-using-an-auto-encoder/)

# ## Dataset-Credit card transactions
# 
# The dataset we're going to use in this kernel is `creditcard.csv` which basically a credit card transactions in the past. Using an encoder-decorder system we will find the hidden data points and apply a linear classifier to detect the Fraud(1) or Genuine/not-fraud (0) credit card transactions. 
# 
# ## Import dependent libraries

# In[ ]:


import pandas as pd
import numpy as np
import os

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# SKLearn related libraries
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Keras NN related libraries
from keras import layers
from keras.layers import Input, Dense
from keras.models import Model, Sequential 
from keras import regularizers


# ## Loading Dataset

# In[ ]:


data_path = '/kaggle/input/creditcardfraud/creditcard.csv'

# print(os.path.exists(data_path))

# Load the data
card_df = pd.read_csv(data_path, header=0)


# In[ ]:


card_df.info()
print("===="*30)
card_df.head()


# ## Exploratory Data Analysis
# 
# 1. Exploring on statistics information about the data

# In[ ]:


card_df.describe().T


# In[ ]:


# Unique class labels
print(f"Unique classes in the dataset are : {np.unique(card_df['Class'])}" )


# In[ ]:


card_df.groupby('Class')['Class'].count().plot.bar(logy=True)


# ## Trasnformation
# 
# Data transformation is one of the steps in data processing. We need to transform certain attributes value so that it makes sense in the further analysis. 

# In[ ]:


# Change the time attribute in day
card_df['Time'] = card_df['Time'].apply(lambda t: (t/3600) % 24 )


# In[ ]:


# Sampling of data
normal_trans = card_df[card_df['Class'] == 0].sample(4000)
fraud_trans = card_df[card_df['Class'] == 1]


# In[ ]:


reduced_set = normal_trans.append(fraud_trans).reset_index(drop=True)


# In[ ]:


print(f"Cleansed dataset shape : {reduced_set.shape}")


# ## Split the Dataset

# In[ ]:


# Splitting the dataset into X and y features
y = reduced_set['Class']
X = reduced_set.drop('Class', axis=1)


# In[ ]:


print(f"Shape of Features : {X.shape} and Target: {y.shape}")


# ## Visualize the data with t-SNE
# 
# TNSE(t-distributed Stochastic Neighbor Embedding) is one of the dimensionality reduction method other than PCA and SVD. This will supress some noise and speed up the computation of pairwise distance between samples. 

# In[ ]:


def dimensionality_plot(X, y):
    sns.set(style='whitegrid', palette='muted')
    # Initializing TSNE object with 2 principal components
    tsne = TSNE(n_components=2, random_state = 42)
    
    # Fitting the data
    X_trans = tsne.fit_transform(X)
    
    plt.figure(figsize=(12,8))
    
    plt.scatter(X_trans[np.where(y == 0), 0], X_trans[np.where(y==0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Normal')
    plt.scatter(X_trans[np.where(y == 1), 0], X_trans[np.where(y==1), 1], marker='o', color='k', linewidth='1', alpha=0.8, label='Fraud')
    
    plt.legend(loc = 'best')
    
    plt.show()


# In[ ]:


# Invoking the method dimensionality_plot
dimensionality_plot(X, y)


# ## Normalize and Scale the features

# In[ ]:


scaler = RobustScaler().fit_transform(X)

# Scaled data
X_scaled_normal = scaler[y == 0]
X_scaled_fraud = scaler[y == 1]


# ## Building Autoencoder Model

# In[ ]:


print(f"Shape of the input data : {X.shape[1]}")


# In[ ]:


# Input layer with a shape of features/columns of the dataset
input_layer = Input(shape = (X.shape[1], ))

# Construct encoder network
encoded = Dense(100, activation= 'tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(50, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(25, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(12, activation = 'tanh', activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(6, activation='relu')(encoded)

# Decoder network
decoded = Dense(12, activation='tanh')(encoded)
decoded = Dense(25, activation='tanh')(decoded)
decoded = Dense(50, activation='tanh')(decoded)
decoded = Dense(100, activation='tanh')(decoded)

output_layer = Dense(X.shape[1], activation='relu')(decoded)

# Building a model
auto_encoder = Model(input_layer, output_layer)


# In[ ]:


# Compile the auto encoder model
auto_encoder.compile(optimizer='adadelta', loss='mse')

# Training the auto encoder model
auto_encoder.fit(X_scaled_normal, X_scaled_normal, batch_size=32, epochs=20, shuffle=True, validation_split=0.20)


# ## Using Autoencode to encode data

# In[ ]:


latent_model = Sequential()
latent_model.add(auto_encoder.layers[0])
latent_model.add(auto_encoder.layers[1])
latent_model.add(auto_encoder.layers[2])
latent_model.add(auto_encoder.layers[3])
latent_model.add(auto_encoder.layers[4])


# In[ ]:


normal_tran_points = latent_model.predict(X_scaled_normal)
fraud_tran_points = latent_model.predict(X_scaled_fraud)
# Making as a one collection
encoded_X = np.append(normal_tran_points, fraud_tran_points, axis=0)
y_normal = np.zeros(normal_tran_points.shape[0])
y_fraud = np.ones(fraud_tran_points.shape[0])
encoded_y = np.append(y_normal, y_fraud, axis=0)


# In[ ]:


# Calling TSNE plot function
dimensionality_plot(encoded_X, encoded_y)


# We can observe that the encoded fraud data points have been moved towards one cluster, whereas there are only few fraud transaction datapoints are there among the normal transaction data points. 
# 
# ## Split into Train and Test

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_enc_train, X_enc_test, y_enc_train, y_enc_test = train_test_split(encoded_X, encoded_y, test_size=0.3)


# In[ ]:


print(f"Encoded train data X: {X_enc_train.shape}, Y: {y_enc_train.shape}, X_test :{X_enc_test.shape}, Y_test: {y_enc_test.shape}")
print(f"Actual train & test data X: {X_train.shape}, Y: {X_train.shape}, X_test :{X_test.shape}, Y_test: {y_test.shape}")


# ## Non-linear Classifier

# In[ ]:


# Instance of SVM
svc_clf = SVC()

svc_clf.fit(X_train, y_train)

svc_predictions = svc_clf.predict(X_test)


# In[ ]:


print("Classification report \n {0}".format(classification_report(y_test, svc_predictions)))


# In[ ]:


print("Accuracy of SVC \n {:.2f}".format(accuracy_score(y_test, svc_predictions)))


# ## Linear Classifier
# 
# Now let's apply linear classifier to classify the data and observe the result. We will use **Logistic Regression** to build the model.

# In[ ]:


lr_clf = LogisticRegression()

lr_clf.fit(X_enc_train, y_enc_train)

# Predict the Test data
predictions = lr_clf.predict(X_enc_test)


# In[ ]:


print("Classification report \n {0}".format(classification_report(y_enc_test, predictions)))


# In[ ]:


print("Accuracy score is : {:.2f}".format(accuracy_score(y_enc_test, predictions)))


# ## Conclusion
# 
# In this analysis, we have found that Support Vector Machine classifier is able to classify the data upto **93%** without encoding and decoding. However, the effect of autoencoder comes when the data gets transformed from non-linear to linearly separable then linear classifier like **Logistic Regression** could perform in a better way.
# 
# The accuracy score of Logistic Regression can go upto **97%**, this is something not happens too often in logistic algorithm. 

# In[ ]:




