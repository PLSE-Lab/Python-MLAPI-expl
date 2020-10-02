#!/usr/bin/env python
# coding: utf-8

# **Source:**
# 
# This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University. The dataset was used in the 1983 American Statistical Association Exposition.
# 
# 
# **Data Set Information:**
# 
# This dataset is a slightly modified version of the dataset provided in the StatLib library. In line with the use by Ross Quinlan (1993) in predicting the attribute "mpg", 8 of the original instances were removed because they had unknown values for the "mpg" attribute. The original dataset is available in the file "auto-mpg.data-original". 
# 
# "The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes." (Quinlan, 1993)
# 
# 
# **Attribute Information:**
# 
# 1. mpg: continuous 
# 2. cylinders: multi-valued discrete 
# 3. displacement: continuous 
# 4. horsepower: continuous 
# 5. weight: continuous 
# 6. acceleration: continuous 
# 7. model year: multi-valued discrete 
# 8. origin: multi-valued discrete 
# 9. car name: string (unique for each instance)
# 
# 
# 
# Below description can be found for this data set. 
# 
# https://archive.ics.uci.edu/ml/datasets/auto+mpg

# In[ ]:


#making the imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#reading the data set

df = pd.read_csv('../input/auto-mpg.csv')


# In[ ]:


df.head()


# In[ ]:


#checking for columns and their data types
df.info()


# In[ ]:


#checking for any nulls
df.isnull().sum()


# In[ ]:


#checking for shape
df.shape


# In[ ]:


#checking for value counts in horse power column
df['horsepower'].value_counts()


# We can see that there are 6 values with question mark (missing). 

# In[ ]:


#below rows have the horse power values missing. 
df[df['horsepower'] == '?']


# Now we can choose to replace the missing values with the mean of 'horse power' column or the mode value.  But we will eliminate the missing values. 

# In[ ]:


#removing the missing values
df = df[df['horsepower'] != '?']


# In[ ]:


#checking the shape of new df
df.shape


# In[ ]:


#converting the data type of horsepower column
df.horsepower = df.horsepower.astype('float')


# In[ ]:


#checking for data types
df.dtypes


# In[ ]:





# # Elploratory Data Analysis (EDA)

# Lets do some EDA to check for distribution of different columns.

# In[ ]:


plt.figure(figsize = (8,5))
sns.set_style('dark')
sns.distplot(df['mpg'])
plt.show()


# In[ ]:


plt.figure(figsize = (8,5))
sns.set_style('dark')
sns.distplot(df['horsepower'])
plt.show()


# In[ ]:


plt.figure(figsize = (8,5))
sns.set_style('dark')
sns.distplot(df['displacement'])
plt.show()


# In[ ]:


plt.figure(figsize = (8,5))
sns.set_style('dark')
sns.countplot(df['cylinders'])
plt.show()


# In[ ]:


#checking the pair plot for numerical columns
sns.pairplot(df.drop(['car name'], axis =1))
plt.show()


# In[ ]:


#heat map of numerical columns
temp_df = df.drop(['car name'], axis = 1)
corr = temp_df.corr()
plt.figure(figsize = (10,8))
sns.heatmap(corr, cmap='coolwarm')
plt.show()


# # Scaling the data

# In[ ]:


#define the scaling function

def scaling_func(x):
    
    y = (x - x.min())/(x.max() - x.min())
    return y


# In[ ]:


#apply the scaling to numerical columns

df['displacement'] = scaling_func(df['displacement'])
df['horsepower'] = scaling_func(df['horsepower'])
df['acceleration'] = scaling_func(df['acceleration'])
df['weight'] = scaling_func(df['weight'])
df['cylinders'] = scaling_func(df['cylinders'])
df['model year'] = scaling_func(df['model year'])
df['origin'] = scaling_func(df['origin'])


# In[ ]:


df.head()


# In[ ]:


#splitting the data into train and test with 80/20 ratio
train_dataset = df.sample(frac=0.8,random_state=0)
test_dataset = df.drop(train_dataset.index)


# In[ ]:


#separating the labels
train_labels = train_dataset.pop('mpg')
test_labels = test_dataset.pop('mpg')


# In[ ]:


#dropping the car name column
test_dataset.drop('car name', axis =1 , inplace = True)
train_dataset.drop('car name', axis = 1, inplace = True)


# In[ ]:


#converting to a np array
train_dataset = train_dataset.values
train_labels = train_labels.values

test_dataset = test_dataset.values
test_labels = test_labels.values


# ## Building the Neural Net Model

# In[ ]:


# TensorFlow and kera import
import tensorflow as tf
from tensorflow import keras


# In[ ]:


#define the model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(1))


# In[ ]:


#compile the model

model.compile(loss= 'mean_squared_error', optimizer= 'RMSprop', metrics= ['mean_absolute_error', 'mean_squared_error'])


# In[ ]:


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])


# In[ ]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[ ]:


#plotting function for MAE and MSE
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)


# As we can see that after about 100 epochs no significant improvement is observed in the Validation Error. Now we will use early stopping to stop the training if there is no significant improvement after a certain number of epochs. 

# In[ ]:


# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_dataset, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)


# ## Checking the Model performance on Test Data

# In[ ]:


loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


# ## Making the Predictions
# Lets make predictions and compare them with actual values. 

# In[ ]:


test_predictions = model.predict(test_dataset).flatten()

plt.figure(figsize = (8,6))
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])


# In[ ]:


#let's check for the error distribution

error = test_predictions - test_labels
plt.figure(figsize = (8,6))
plt.hist(error, bins = 25, color = '#6f93b7')
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")


# In **Summary** we can say that:
# 
# **Scaling** is important to bring all features on a similar scale.
# 
# When there is less data then we should use a small neural network with **less Hidden Layers** to avoid **Overfitting**.
# 
# **Early Stopping** is also a useful technique to prevent overfitting. 
# 
# 
# 
