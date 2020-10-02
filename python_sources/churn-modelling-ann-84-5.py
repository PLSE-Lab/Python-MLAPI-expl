#!/usr/bin/env python
# coding: utf-8

# ![https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2Fanalytics-vidhya%2Fwhy-not-deploy-a-machine-learning-model-in-excel-fc95fe4e0629&psig=AOvVaw3DgAFzBKP4bn3txibHPXXb&ust=1594202293350000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCOjShsLwuuoCFQAAAAAdAAAAABAD](http://)

# Hlo Everyone and welcome to another exciting notebook in which we are going to do the churn modelling in which based on the data provided we are going to detect wheather the customer will stay or not with the bank in the upcomming months/years.
# 
# This approach can apllied through various techniques like Decision Tree, Logistic Regression,etc.
# But i m here to do it with little suprising way which is obsivously not new but i think it the most applied method when comes to such problems:)
# 
# OK let's get started!!!

# In[ ]:


## importing main libaries requied:)
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Since we know that the first three coloums of the dataset are not at all requried.
# 1.**The First one Row Number(ID)** is just like the row no which will not add any value to our model
# 2.**The customer unique id** which is just to distinguish the customer will not help our model to track any pattern or information.
# 3.**The Surname** which is again will be unique and have no impact on the model
# 
# **GOlDEN RULE:
# > > > > Always remmember if the data is not neccesary please remove that because that will sometimes affect our model accuracy !!****

# So in X by using the slicing from the dataframe I have left leaving 3 coloums from the start
# and in Y we take only last coloumn..:)

# In[ ]:


def get_data_cleaned(df: pd.DataFrame):
    X = df.iloc[:, 3:-1].values
    y = df.iloc[:, -1].values
    return X, y


# Now since we have categorical data we need to change that to numeric so here i have used two encoder:
# **1.Label encoder
# 2.one hot encoder****

# In[ ]:


def get_procsed_dependent_variable(depen_var):
    le = LabelEncoder()
    depen_var[:, 2] = le.fit_transform(depen_var[:, 2])
    return depen_var


# In[ ]:


def get_procssed_independent_variable(depen_var):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    depen_var = np.array(ct.fit_transform(depen_var))
    return depen_var


# Then comes the most important thing of ay deep learning model in which we need to scale our data!!
# 
# **GOLDEN RULE:**
# Always remmember to scale the data when dealing with deep learning models as without scaling u model nevers perfor well tasks.
# 
# Scalability matters in machine learning because:
# 
# Training a model can take a long time.
# 
# A model can be so big that it can't fit into the working memory of the training device.
# Even if we decide to buy a big machine with lots of memory and processing power, it is going to be somehow more expensive than using a lot of smaller machines. In other words, vertical scaling is expensive.

# In[ ]:


def get_splitted_data_and_feature_scaling(depen_var, indepen_var):
    X_train, X_test, y_train, y_test = train_test_split(depen_var, indepen_var, test_size=0.25, random_state=32)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test


# Many of you would have doubt that why **fit_transform()** for train and only **transform()** for the test ??
# 
# Well the answer is right here We use fit_transform() on the train data so that we learn the parameters of scaling on the train data and in the same time we scale the train data. We only use transform() on the test data because we use the scaling paramaters learned on the train data to scale the test data.

# In[ ]:


def final_making_of_model(X_train, X_test, y_train, y_test):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=7, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=7, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train, batch_size=32, epochs=20)
    y_pred = ann.predict(X_test)
    y_pred = (y_pred > 0.5)
    print("Results")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1),y_test.reshape(len(y_test), 1)), 1))
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("confusion_matrix")
    print(cm)


# Ok guys a final tip from my side before we end up always try to write in methods because those will help you very much in the upcomming life as if you wanna change anything just go to that method andd change no need for full code correction and also thats add a beauty to your work.
# 
# If you see my main code its just matters of some method calling which help the user to read very easily and also the end user do not need ur codes and all that its just output that matters!!!!:):):)
# 
# With this hope lets meet in next notebook!!
# 
# **Happy learning:)**

# In[ ]:


def main():
    df = pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")
    depen_var, indepen_var = get_data_cleaned(df)
    depen_var = get_procsed_dependent_variable(depen_var)
    depen_var = get_procssed_independent_variable(depen_var)
    X_train, X_test, y_train, y_test = get_splitted_data_and_feature_scaling(depen_var, indepen_var)
    final_making_of_model(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()


# **The results shows us that 1st coloumn is for the predicted values and the other coloumn is for the actual values**

# If you like my work please do **UPVOTE**** for me that will add more potential to work 
# Thank You:)
