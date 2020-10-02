#!/usr/bin/env python
# coding: utf-8

# # Titanic Competition Using Keras Sequential AI for Beginners

# ## This first cell is the code that Kaggle automatically adds to competiton notebooks when you start one.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Here we import the libraries we will be using for this project. We will be building a Sequential Model using the Keras library that was added into Tensorflow recently(Yay!), with dropout implemented to combat any sort of overfitting. We will be using sklearn's Simple Imputer to fill in values that are not in the data set and LabelEncoder to label categorical values so they are numeric ones. 

# In[ ]:


from sklearn import preprocessing
from tensorflow.python import keras
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout


# ## Here we set our variables to the path of the file you want to use for training and the final set of testing data. With Kaggle competitions it is labeled to let you know what file is what pretty easily. If you dont have the files you will need to add them. You can do this by clicking the blue link that says "+ Add Data" in the top right of the sidebar and searching for Titanic Competiton. Once added, you can also get the link by extending the data option on the side bar and clicking the file you want the path to. You should then be able to see it at the bottom of your notebook and copy and paste it.

# In[ ]:


train_data = '/kaggle/input/titanic/train.csv'
test_data = "/kaggle/input/titanic/test.csv"


# ## For our next 2 cells we are going to be transforming the csv files into Panda datasets. After we load them into a Panda, we will use the head() method to check that the files were loaded properly. 

# In[ ]:


train_panda = pd.read_csv(train_data, index_col = "PassengerId")
train_panda.head()


# In[ ]:


test_panda = pd.read_csv(test_data)
test_panda.head()


# In[ ]:


train_panda.Cabin.fillna("M")


# In[ ]:


test_panda.Cabin.fillna("M")


# In[ ]:


train_panda.Cabin = train_panda.Cabin.astype('str')


# In[ ]:


test_panda.Cabin = test_panda.astype('str')


# ## Awesome! Everything looks like it has loaded properly. So we want to get rid of some of this data as it has little impact on our model and could even ruin our model entirely. Columns that have too many unique values or missing values are going to be of little to no use to us. There is a method to return all the unique values in a column, but we are just going to use common sense here and say we dont want the columns ("Name", "Cabin", "Ticket"). These are all columns that will have little to no impact on our models accuracy and even worse could overfit the data making it seem like it performs really well on our validation set but then performs poorly on our test set or in the real world. This is because the model will start learning the unique variable(such as name) and give them too heavy a weight while training. (ex. model learning Mr. Owen Harris is always dead doesnt help it predict that a new person introduced to the model survived or perished). We can achieve this by using the drop() method but we will be excluding them in a different way which I will explain in a little bit. 

# ## For our next cell we are going to use the LabelEncoder. I could have imported this above fully using "from sklearn.preprocessing import LabelEncoder", but I want to also show another way to access it. We imported preprocessing already so we can access this method with dot notation. First thing we need to do is find out what columns are Categorical (needs to be labeled), and then find out what columns are Numerical. We can do this by simply looking up at the table we printed a few cells above. For label encoding we want our categorical columns(anything that is not an int or float), so we have ("Name", "Sex", "Cabin", "Embarked"). We don't want Name or Cabin to be in the final training set so we wont bother labeling them.

# ## So lets go ahead and make ourselves a label encoder.  

# In[ ]:


encoder = preprocessing.LabelEncoder()


# ## That was pretty easy huh! Next we are gonna make a list of the features we want to encode into our data.

# In[ ]:


cat_features = ["Sex", "Embarked","Cabin"]


# ## When we go to actually encode our labels we will get thrown an error saying that Embarked is not of type str, but only for the training dataset and not the testing one. That's weird! Well that is a very easy fix with the .astype() method! So we will take the column from our training panda and transform it into a column of strings with this line of code below.

# In[ ]:


train_panda["Embarked"] = train_panda["Embarked"].astype(str)


# ## Now we encode both training and testing dataset. Why? Because we are training our model that the column "sex" is just 1s and 0s, and when we go to make predictions it wont be able to connect that "male" is 0 or "female" is 1. So we make an encoded set of data that consists of the columns we set aside for encoding (cat_features).
# 

# In[ ]:


encoded_train = train_panda[cat_features].apply(encoder.fit_transform)
encoded_test = test_panda[cat_features].apply(encoder.fit_transform)


# ## Now we have to get all of our numerical columns(except "Ticket") together in a list and we also need to get all the numerical columns(except "Ticket") for our testing data in a group together too. 

# In[ ]:


num_features = ["Survived","Pclass","Age","SibSp","Parch","Fare"]
test_features= ["Pclass","Age","SibSp","Parch","Fare"]


# ## Now this is where we make the magic happen. We will rejoin that encoded data back into our regular testing and training set. But when we do we are only going to take the columns we want (everything but Name, Cabin, and Ticket), and we already did all the work to make that happen!

# In[ ]:


training_data = train_panda[num_features].join(encoded_train)
test_data = test_panda[test_features].join(encoded_test)


# ## Lets see what our new data looks like.

# In[ ]:


training_data.head()


# ## And the testing one.

# In[ ]:


test_data.head()


# ##  Cool so now our data is in a format that our model can understand, but wait is there any missing values?! Lets check by getting the sum of all null values in our data using the isnull() method and the sum() method.

# In[ ]:


training_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# ## Looks like our data is not quite ready yet. There are many options here. We can ditch the column age, but we dont like losing all that data. We could fill it with 0, but we dont want it to throw our data off too much so lets go ahead and use imputation to fill these values in. Imputation allows you to fill in values with values such as the mean on all the ages in the list. That seems like the best bet for keeping the model accurate. 
# 
# ## We will define the imputer and transform our data sets here in the same cell.

# In[ ]:


my_imputer = SimpleImputer()
imputed_train = pd.DataFrame(my_imputer.fit_transform(training_data))
imputed_test_data = pd.DataFrame(my_imputer.fit_transform(test_data))


# ## Now lets check our data again.

# In[ ]:


imputed_train.head()


# ## Uh, oh! Looks like our column labels are gone! And so is our index label! This is fine, I am going to leave the index labels as they dont matter as much but we do need to reset the column labels. This happens everytime you impute and will need to be addressed.

# In[ ]:


imputed_train.columns = training_data.columns
imputed_test_data.columns = test_data.columns


# ## And we check our data

# In[ ]:


imputed_train.head()


# ## IGNORE Now we will normalize our test and training data, and prepare it for the model.

# In[ ]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#normalized_train = scaler.fit_transform(imputed_train)
#normalized_test = scaler.fit_transform(imputed_test_data)


# ## IGNORE We will turn them into panda DF's again, reset the columns, and check to see if they look correct.

# In[ ]:


#normalized_train_data = pd.DataFrame(data= normalized_train)


# In[ ]:


#normalized_train_data.columns = training_data.columns


# In[ ]:


#normalized_test_data = pd.DataFrame(data = normalized_test)


# In[ ]:


#normalized_test_data.columns = test_data.columns


# In[ ]:


#normalized_test_data.head()


# In[ ]:


#normalized_train_data.head()


# ## Finally! Our data is ready to be split up for our model. We have what we want to predict(y), and our predictors(X) that need to be defined still. For X we will create a Panda without "Survived", because that is what we are trying to predict. We are telling the code it can be found on axis=1 (it means columns). And we label Y as nothing but "Survived".

# In[ ]:


X = imputed_train.drop("Survived", axis = 1)
y = imputed_train["Survived"]


# ## Now we test and get the shape of our training data. 

# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


X.shape


# ## Everything looks good! Now we set our random seed. It can be any number you desire but it is recommended you set one for model consistency. I like 42 because it IS the answer to life, the universe, and everything. 

# In[ ]:


tf.random.set_seed(42)


# ## Now time to build our model! We are going to build a sequential model. This is where you can build a model from scratch. I know this is a beginner tutorial, but we wont make this too complicated. Let us start by defining the model variable. 

# In[ ]:


model = Sequential()


# ## We will set the first layer as Dense with 7 nuerons, and a "relu" activation function. It's not super important to understand all the activation fuctions for this notebook, but is recommened that you learn what they are and how they are used. The first layer needs to always have the parameter input_shape. We can look up at where we used the shape method to find out what it is. The shape is listed as (891,7), we are going to be inputing the classes so the input_shape is (7,). WE also implement a method called dropout. In a nutshell it turns off random nuerons while training to avoid overfitting. Using the parameter (0.2), we are telling it to turn off 20 percent of the nuerons.

# In[ ]:


model.add(Dense(8, activation = "relu", input_shape = (8,)))


# ## Now lets add 2 more Dense layers with 50 nuerons and relu activation.

# In[ ]:


model.add(Dense(5, activation = "relu"))


# ## Now lets add our output layer. We have 2 classes or outcomes from this dead or alive (1 or 0), so we specify that to the layer. We are using a softmax activation to classify passenger survival. Softmax will return a probabilty for each outcome and we will pick the highest probabilty as our prediction.   

# In[ ]:


model.add(Dense(2, activation = "softmax"))


# ## Time to compile all those layers together and build the model! We need to define our loss mesurement, our optimizer, and our measure of accuracy. There are many different options for these and you should absolutely play around with them. Adam is considered one of the best optimizers and your metrics for accuracy will always depend on your model and the data it is predicting. 

# In[ ]:


model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer = "adam", metrics = ['accuracy'])


# In[ ]:


callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50, restore_best_weights = True, verbose = 1)


# ## Time to fit(train) our model! We pass in our X and y data we made earlier. Our batch size is going to be 1 since we arent looking at more than 1 passenger at a time. Epochs is the number of training cycles it will go through. We will split off 20 percent of our data for validation. You can change verbose to 1 if you want to watch the training in action. I find it exciting, but I am sure others feel like they are watching paint dry if they try to do so. 

# In[ ]:


model.fit(X, y,
          batch_size=1,
          epochs=1000,
          callbacks = [callback],
          validation_split = 0.2,
          verbose = 1)


# ## An hour later......

# ## Now lets use that model we just built to make predictions! 

# In[ ]:


preds = model.predict(imputed_test_data)


# ## Let's see what it looks like 

# In[ ]:


print(preds)


# ## Thats not what we want! Remember how I said Softmax will return a probabilty for each answer. Well we need to take the one with the highest probability and make that our prediciton. We will do this with the argmax() method. 

# In[ ]:


predictions = np.array(preds).argmax(axis=1)


# ## Check it out.

# In[ ]:


print(predictions)


# ## Alright this is better, but still have a few tweaks we need to make to the panda before we can upload it for submission. 

# In[ ]:


passenger_id=test_panda["PassengerId"]
results=passenger_id.to_frame()
results["Survived"]=predictions


# ## Looking again.

# In[ ]:


results.head()


# ## Perfect! Now we upload it for submission. 

# In[ ]:


results.to_csv("Titanic_ai_model.csv", index=False)


# # Congrats on building an AI model from scratch for Titanic survival predictions!
