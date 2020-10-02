#!/usr/bin/env python
# coding: utf-8

# # Simple Deep Neural Networks with Keras
# In this tutorial i am going to show how to implement Deep Neural Networks in keras and Also we will have a look on simple feature engineering to be able to classify the dataset efficiently

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# First let's start by importing our dataset into dataframes using pandas. Dataframes enables us to work easily with data usign its built in functions.

# In[ ]:


import pandas as pd
training = pd.read_csv("../input/train.csv");

x_test = pd.read_csv("../input/test.csv");


# Now let's have a look on the dataset and search for important information manually.

# In[ ]:


training.head(10)


# In[ ]:


training['n_new']= training['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[ ]:


training['n_new'].value_counts()


# In[ ]:


training['Title'] = 0


# In[ ]:


training.loc[training["n_new"] == 'Mr', 'Title'] = 1
training.loc[training["n_new"] == 'Miss', 'Title'] = 4
training.loc[training["n_new"] == 'Mrs', 'Title'] = 5
training.loc[training["n_new"] == 'Master', 'Title'] = 3
training.loc[training["n_new"] == 'Dr', 'Title'] = 2


# Now I will explore the correlations of the featues relative to the target variable.
# #### Note: not all features are listed but only numeric features.

# In[ ]:


import seaborn as sns


import matplotlib.pyplot as plt


corr = training.corr()
f, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.1, center=0,
            square=True, linewidths=.5)


# In[ ]:


corr


# I will explore the size of the dataset to compare it with the number of NANs in the dataset

# In[ ]:


print("The number of traning examples(data points) = %i " % training.shape[0])
print("The number of features we have = %i " % training.shape[1])


# I will count the number of data examples in each class in the target to determine which metric to use while evaluationg performance.

# In[ ]:


unique, count= np.unique(training["Survived"], return_counts=True)
print("The number of occurances of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )


# The numbers doesn't seem to be very far from each other, So i will use accuracy for performance eval..

# Now i will check the number of Null values. If most of a column's values are Nulls or NaNs i will drop it because filling it will not be accurate but if the number is small then i will fill it with the mean values.

# In[ ]:


training.isna().sum()


# From the results of correlation matrix and the Nan number and manual checking of dataset, I will drop "Name" ,"Ticket","Cabin" and "PassengerId".
# <lb>I will also store the label column and drop it from training.
# <lb>Then i will engineer the catagorical features and map the strings to integers to be able to use it in the model later.

# In[ ]:


training.columns


# In[ ]:


C = training.Cabin[training.Cabin.isna()]
C_not = training.Cabin[training.Cabin.notna()]
C.values[:] = 0
C_not.values[:] = 1


# In[ ]:


cabine_not = pd.concat([C, C_not]).sort_index()


# In[ ]:


np.random.seed(0)
training['sp'] = training.SibSp + training['Parch']
training['cabine_n'] = cabine_not
training.cabine_n = training.cabine_n.astype(int)
training.drop(["n_new", "Name" ,"Embarked", "PassengerId","Ticket","Cabin"], inplace = True, axis = 1 )


# In[ ]:


x_train = training
repCol9 = {1 : 3 ,   2 : 2 , 3 : 1  }


x_train['IsAlone'] = 1
x_train['IsAlone'].loc[x_train['sp'] > 0] = 0

x_train.replace({'Pclass': repCol9} , inplace = True )
x_train = pd.get_dummies(x_train)

y_train = x_train["Survived"]
x_train = x_train.drop(['Survived'], axis = 1)

print(y_train.shape )
y_train.head(20)
x_train.head()


# In[ ]:


import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=0)
x = imp.fit_transform(x_train)
x_train = pd.DataFrame(x, columns = x_train.columns)
x_train[[ 'Pclass', 'SibSp', 'Parch', 'Title', 'sp',
       'cabine_n', 'IsAlone', 'Sex_female', 'Sex_male']] = x_train[[ 'Pclass', 'SibSp', 'Parch', 'Title', 'sp',
       'cabine_n', 'IsAlone', 'Sex_female', 'Sex_male']].astype(int)


# In[ ]:


from scipy import stats
import numpy as np

z = np.abs(stats.zscore(x_train))
zee = (np.where(z > 3))[1]

print("number of data examples greater than 3 standard deviations = %i " % len(zee))
# x_train = x_train[(z < 2.5).all(axis=1)]


# I will Plot some features to see if there is any pattern in the data.

# In[ ]:


import matplotlib.pyplot as plt

import seaborn as sns
sns.pairplot(training)


# In[ ]:


x_test.head()


# Since i dropped some features from the training set I have to drop the same featurres from the test set and do the same steps of feature eng.

# In[ ]:


x_test['n_new']= x_test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[ ]:


x_test['Title'] = 0

x_test.loc[x_test["n_new"] == 'Mr', 'Title'] = 1
x_test.loc[x_test["n_new"] == 'Miss', 'Title'] = 4
x_test.loc[x_test["n_new"] == 'Mrs', 'Title'] = 5
x_test.loc[x_test["n_new"] == 'Master', "Title"] = 3
x_test.loc[x_test["n_new"] == 'Dr', 'Title'] = 2


# In[ ]:


C = x_test.Cabin[x_test.Cabin.isna()]
C_not = x_test.Cabin[x_test.Cabin.notna()]
C.values[:] = 0
C_not.values[:] = 1

x_test['sp'] = x_test.SibSp + x_test['Parch']
x_test['cabine_n'] = cabine_not
x_test.cabine_n = x_test.cabine_n.astype(int)

x_test['IsAlone'] = 1
x_test['IsAlone'].loc[x_test['sp'] > 0] = 0


x_test.drop(["Name", "Embarked", "Ticket", "n_new", "Cabin"], inplace = True, axis = 1 )
x_test.replace({'Pclass': repCol9} , inplace = True )
x_test = pd.get_dummies(x_test)
x_test.head()


# Now it is time to design the ML Pipeline. I will use Deep Neural Networks in Keras to classify the dataset. The number of layers i am using is optmized using some error analys of the results.
# <lb> I will use early stopping to stop if the error is not decreasing.

# In[ ]:


fact = y_train[y_train == 0].count() / y_train[y_train == 1].count()

class_weight = {1: fact, 0: 1.}


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import callbacks
from keras import optimizers
from keras.layers import BatchNormalization

#y_train = np_utils.to_categorical(y_train)

InputDimension = 11
print(y_train.shape )

model = Sequential()
model.add(Dense(25,input_dim=InputDimension, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))



model.add(Dense(2, activation='softmax'))


earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min')
optimizer = optimizers.Adam(lr=0.001, decay=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
history = model.fit(x_train, pd.get_dummies(y_train), epochs=2000, batch_size=60, validation_split=0.2, verbose=1, callbacks=[earlystopping], class_weight = class_weight)


# Now let's see how good is my training with respect to validation accuracies.

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])


# Now i will normalize the test set as i did before and will fill null values in the dataset as well.

# In[ ]:


id = x_test['PassengerId']

x_test.drop(['PassengerId'], inplace = True, axis = 1)

imp = IterativeImputer(max_iter=10, random_state=0)
x = imp.fit_transform(x_test)
x_test = pd.DataFrame(x, columns = x_train.columns)
x_test[[ 'Pclass', 'SibSp', 'Parch', 'Title', 'sp',
       'cabine_n', 'IsAlone', 'Sex_female', 'Sex_male']] = x_test[['Pclass', 'SibSp', 'Parch', 'Title', 'sp',
       'cabine_n', 'IsAlone', 'Sex_female', 'Sex_male']].astype(int)


# Now every thing is okay with the dataset so i will predict the output values for submission

# In[ ]:


predictions = model.predict(x_test)


# Predictions will return probability between 0 and 1 for survived or non survived so i will take the argmax() of the array to get the max index for each test example

# In[ ]:


predictions = np.argmax(predictions, axis = 1)


# It is time to make submission.

# In[ ]:


id.reset_index(drop=True, inplace=True)
out = pd.DataFrame({'PassengerId': id, "Survived": predictions})
out.to_csv('titanic-predictions.csv', index = False)
out.head(100)

