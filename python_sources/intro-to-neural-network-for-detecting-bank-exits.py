#!/usr/bin/env python
# coding: utf-8

# # Given Dataset consisting of Bank Customer Information, we need to create a model that will predict if a customer will leave the bank or not.

# ### Import the necessary Libraries

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense


# In[ ]:



### Removes warnings that occassionally show up
import warnings
warnings.filterwarnings('ignore')


# ### Reading the dataset

# In[ ]:


data_bank=pd.read_csv('/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv')
data_bank.head()


# ### EXPLORATORY DATA ANALYSIS

# In[ ]:


data_bank.shape


# In[ ]:


data_bank.info()


# In[ ]:


data_bank.isna().sum()


# The above information indicates that there are no missing values

# In[ ]:


# 5 point summary
data_bank.describe().T


# In[ ]:


# CORRELATION - HEAT MAP
colormap = plt.cm.plasma
plt.figure(figsize=(17,10))
plt.title('Correlation of Cutomer Exiting Bank', y=1.05, size=15)
sns.heatmap(data_bank.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, 
            linecolor='white', annot=True)


# Now, Checking number of people exited from the bank via Geography. i.e. from mentioned Countries

# In[ ]:


sns.countplot(x="Geography", data=data_bank,hue="Exited")


# In[ ]:


gendermap = sns.FacetGrid(data_bank,hue = 'Exited')
(gendermap.map(plt.hist,'Age',edgecolor="w").add_legend())


# ### Drop the columns which are unique for all users like IDs

# #### Removing CustomerId, RowNumber and Surname. Removing Surname is basically because One Hot Encoding will behave unusual if this column is retained(give error)

# In[ ]:


bank_data_new = data_bank.drop(['RowNumber', 'CustomerId', 'Surname'], axis =1)


# In[ ]:



bank_data_new.head()


# #### Distribution of Numerical Column

# In[ ]:


numerical_distribution = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
for i in numerical_distribution:
    plt.hist(data_bank[i])
    plt.title(i)
    plt.show()


# #### Label Encoding the Bank.csv columns to make them integers

# In[ ]:


## Label Encoding of all the columns
# instantiate labelencoder object
le = LabelEncoder()

# Categorical boolean mask
categorical_feature_mask = bank_data_new.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = bank_data_new.columns[categorical_feature_mask].tolist()
bank_data_new[categorical_cols] = bank_data_new[categorical_cols].apply(lambda col: le.fit_transform(col))
print(bank_data_new.info())


# In[ ]:



df_scaled = bank_data_new.apply(zscore)
X_columns =  df_scaled.columns.tolist()[1:10]
Y_Columns = bank_data_new.columns.tolist()[-1:]

X = df_scaled[X_columns].values
y = np.array(bank_data_new['Exited']) # Exited

print(y)
print(X)


# ### Model Training

# In[ ]:


#splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 8)


# In[ ]:


X_train.shape


# ### One Hot Encoding

# In[ ]:


#Encoding the output class label (One-Hot Encoding)
y_train=to_categorical(y_train,2)
y_test=to_categorical(y_test,2)


# ### Normalize the data

# In[ ]:


from sklearn.preprocessing import Normalizer
normalize=Normalizer(norm="l2")
X_train=normalize.transform(X_train)

print(X_train)


# In[ ]:



X_test=normalize.transform(X_test)
print(X_test)


# ### Building the Model

# In[ ]:


#Initialize Sequential Graph (model)
model = tf.keras.Sequential()


# In[ ]:


model.add(Dense(units=6, activation='relu', input_shape=(9,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# In[ ]:


history=model.fit(X_train, y_train, batch_size=45, epochs=200, validation_data=(X_test,y_test))


# 
# The above code indicates:
# 
# X_train is the independent variable portion of the data which needs to be fitted with the model.
# 
# y_train is the output portion of the data which the model needs to produce after fitting.
# 
# batch_size: How often we want to back-propogate the error values so that individual node weights can be adjusted.
# 
# epochs: The number of times we want to run the entire test data over again to tune the weights. This is like the fuel of the algorithm.
# 
# validation_split: 0.1 The fraction of data to use for validation data.

# ### Checking the Accuracy for Test

# In[ ]:


score = model.evaluate(X_test, y_test,verbose=1)

print(score)


# ### Train Accuracy

# In[ ]:



score = model.evaluate(X_train, y_train,verbose=1)

print(score)


# ### Confusion Matrix Calculation

# In[ ]:



y_pred = model.predict(X_test)


# In[ ]:


y_pred = (y_pred > 0.5)


# In[ ]:


from sklearn.metrics import confusion_matrix

confmatrx= confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


# In[ ]:


confmatrx


# ### Testing the Neural Network

# In[ ]:


plt.plot(np.array(history.history['accuracy']) * 100)
plt.plot(np.array(history.history['val_accuracy']) * 100)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'validation'])
plt.title('Accuracy over epochs')
plt.show()


# ## PRINT THE ACCURACY

# In[ ]:


print (((confmatrx[0][0]+confmatrx[1][1])*100)/(len(y_test)), '% of testing data was classified correctly')


# In[ ]:


# Checking the accuracy using accuracy_score as well to check if the above calculation is correct or not.
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# # Summary
# #### The Train and Test set are almost similar. It shows that the model did not overfit on the train set.
# 
# #### On Compiling the Neural Network, I have used optimizer as "adam" as it is very efficient to Stochastic Gradient Decent. The loss function used is "binary_crossentropy which is used within adam.
# #### The accuracy metrics which will be evaluated(minimized) by the model. The "Accuracy" is used as a criteria to improve model performance.
# 
# #### On calculation of the accuracy based on the Confusion Matrix, it came out as 86.45(APPROX). which matches with the score calculated through various EPOCH
# 
# #### We can used Optimizer as "SGD" as well to check if we get better results and accuracy.
# 
# #### HOWEVER, FOR BETTER RESULT WE CAN YOUR HYPER PARAMETER TUNING FOR BETTER RESULT.
