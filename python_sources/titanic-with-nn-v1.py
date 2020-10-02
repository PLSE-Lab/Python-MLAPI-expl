#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load the data from the input CSV files. Use relative path.
train_path = '..//input//train.csv'
test_path = '..//input//test.csv'

train_data = pd.read_csv(train_path,low_memory=False)
test_data = pd.read_csv(test_path,low_memory=False)

train_data.info()
test_data.info()


# In[ ]:


train_data.shape


# In[ ]:


# backing up survival data before merging train and test set
survival_df = train_data.filter(['PassengerId','Survived'],axis=1)


# In[ ]:


# drop survival data from train data and add a new column indicating 'train'/'test' data
train_set = train_data.drop('Survived',axis=1)
train_set['Type']='train'


# In[ ]:


# Adding the new column to test data
test_set = test_data.loc[:,:]
test_set['Type'] = 'test'


# In[ ]:


# merging train and test data for EDA
total_set = pd.concat([train_set, test_set],axis=0)


# In[ ]:


total_set=total_set.reset_index(drop=True)


# In[ ]:


#check data for errors
total_set.isnull().sum()


# Age, Fare, Cabin, Embarked - these have missing values

# In[ ]:


# The 'Fare' value is of the ticket. So, for 4 passengers on same ticket, all 4 will show the same ticket number and the total
# fare in 'Fare' column

# So, I want to calculate number of passengers on same ticket and derive fare per ticket in a new column 'MyFare'

# Here, getting the number of passengers in 'PassOnSameTicket' column
ticket_count_data = total_set.pivot_table(values='Name',index='Ticket',aggfunc='count')
ticket_count_data.rename(columns={'Name':'PassOnSameTicket'}, inplace=True)


# In[ ]:


# merging with total set
total_set = pd.merge(total_set, ticket_count_data, on='Ticket')


# In[ ]:


# Deriving 'MyFare'
total_set['MyFare'] = round((total_set['Fare']/total_set['PassOnSameTicket']),2)


# In[ ]:


# Proceeding to impute the missing 'Fare' value
total_set.loc[total_set.Fare.isnull(),:]


# In[ ]:


# Imputing with mean value of same 'Pclass', 'Embarked' and 'Sex'
total_set.loc[total_set.Fare.isnull(), ['Fare','MyFare']] =      round((total_set.loc[(total_set.Pclass==3) & (total_set.Embarked == 'S') & (total_set.Sex == 'male'),'MyFare'].mean()),2)
total_set.loc[total_set.PassengerId==1044,:]


# In[ ]:


#check data for errors
total_set.isnull().sum()


# In[ ]:


# Proceeding to impute the missing 'Embarked' value
total_set.loc[total_set.Embarked.isnull(),:]


# In[ ]:


# trying the derive the 'Embarked' value based on the 'MyFare' value for similar 'Pclass' and 'Sex' 
total_set.loc[~total_set.Embarked.isnull(),:].pivot_table(values='MyFare',index=['Pclass','Embarked','Sex'],aggfunc={'MyFare':np.mean})


# Based on above, it looks like the 'Embarked' value can be 'C'

# In[ ]:


# Imputing the missing 'Embarked' values with 'C'
total_set.loc[total_set.Embarked.isnull(),'Embarked'] = 'C'
total_set.loc[total_set.Cabin=='B28',:]


# In[ ]:


#check data for errors
total_set.isnull().sum()


# In[ ]:


# Checking the data with missing 'Age' values
total_set.loc[total_set.Age.isnull(),:]


# Unable to find a way to reasonable impute 'Age'. As it is almost 20% of total rows, it is better not to consider this column for modelling. So, deciding to drop 'Age'

# In[ ]:


# Dropping columns which are not necessary for modelling
pre_model_set = total_set.drop(columns=['Name','Age','Ticket','Fare','Cabin'])


# In[ ]:


pre_model_set


# In[ ]:


# converting the categorical columns to dummies

# creating dummies for 'Sex'
sex_df = pd.get_dummies(pre_model_set.Sex, prefix='Sex_', drop_first=False)
pre_model_set = pre_model_set.drop(columns='Sex')
pre_model_set = pd.concat([pre_model_set, sex_df], axis=1)


# In[ ]:


# creating dummies for 'Embarked'
embarked_df = pd.get_dummies(pre_model_set.Embarked, prefix='Embarked_', drop_first=False)
pre_model_set = pre_model_set.drop(columns='Embarked')
pre_model_set = pd.concat([pre_model_set, embarked_df], axis=1)


# In[ ]:


# separating the train and test data
pre_model_train_set = pre_model_set.loc[pre_model_set.Type == 'train',:].drop(columns='Type')


# In[ ]:


pre_model_test_set_with_passenger_id = pre_model_set.loc[pre_model_set.Type == 'test',:].drop(columns=['Type'])
pre_model_test_set = pre_model_test_set_with_passenger_id.drop(columns=['PassengerId'])


# In[ ]:


# merge survival data to pre_model_train_set
pre_model_train_set = pd.merge(pre_model_train_set, survival_df, on='PassengerId')
pre_model_train_set = pre_model_train_set.drop(columns='PassengerId')


# In[ ]:


'''
Training data has column "Survived" which is the Y value. So, assign all columns apart from "Survived" to X_train and assign the
"Survived" column value to Y_train
'''
X_train=pre_model_train_set.drop("Survived",axis=1).values
Y_train=pre_model_train_set["Survived"].values

Y_train = keras.utils.to_categorical(Y_train, 2)

print ('Shape of X_train >>',X_train.shape)
print ('Shape of Y_train >>',Y_train.shape)

'''
Test data are not labeled. So, assigning all to X_test
'''
X_test=pre_model_test_set.values

print ('Shape of X_test >>',X_test.shape)


# In[ ]:


# convert to float and standardize, normalize
from sklearn.preprocessing import StandardScaler

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

std_scale = StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std  = std_scale.transform(X_test)


# In[ ]:


# create model
model = Sequential()
model.add(Dense(35, input_dim=10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(21, activation = 'relu', kernel_regularizer=l2(0.01))) 
model.add(Dense(2, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


#fit model with 10% validation
history = model.fit(X_train_std, Y_train,
              batch_size=10,
              epochs=20,
              validation_split=0.1,
              shuffle=True)


# In[ ]:


# check accuracy on complete training data set
scores_train = model.evaluate(X_train_std, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))


# In[ ]:


# Making prediction on the test data set
predictions = model.predict(X_test_std)
predictions = np.argmax(predictions, axis = 1)
predictions


# In[ ]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


#write predictions of test data to file for submission
result=pd.DataFrame({"PassengerId": pre_model_test_set_with_passenger_id.PassengerId,"Survived": predictions})
#result.to_csv("/content/drive/My Drive/Colab Notebooks/mnist_kaggle/mnist_cnn_only.csv", index=False, header=True)
result.to_csv("titanil_nn_1.csv", index=False, header=True)

