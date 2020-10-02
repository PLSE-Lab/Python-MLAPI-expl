#!/usr/bin/env python
# coding: utf-8

# # What's cooking kernel !

# In[ ]:


import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer 


# Load the dataset

# In[ ]:


df = pd.read_json("../input/train.json")
testset = pd.read_json("../input/test.json")


# In[ ]:


df.head()


# In[ ]:


testset.head()


# Check for any null values.

# In[ ]:


df.isnull().sum()


# In[ ]:


testset.isnull().sum()


# Check different types of cuisines

# In[ ]:


df.cuisine.unique()


# # Text Data processing

# Convert the ingredients to string.

# In[ ]:


df.ingredients = df.ingredients.astype('str')
testset.ingredients = testset.ingredients.astype('str')


# In[ ]:


df.ingredients[0]


# In[ ]:


testset.ingredients[0]


# Lets remove those unnecessary symbols, which might be problem when tokenizing and lemmatizing

# In[ ]:


df.ingredients = df.ingredients.str.replace("["," ")
df.ingredients = df.ingredients.str.replace("]"," ")
df.ingredients = df.ingredients.str.replace("'"," ")
df.ingredients = df.ingredients.str.replace(","," ")


# In[ ]:


testset.ingredients = testset.ingredients.str.replace("["," ")
testset.ingredients = testset.ingredients.str.replace("]"," ")
testset.ingredients = testset.ingredients.str.replace("'"," ")
testset.ingredients = testset.ingredients.str.replace(","," ")


# In[ ]:


df.ingredients[0]


# In[ ]:


testset.ingredients[0]


# Convert everything to lower ( I think they are already in lower case, but to be on safe side).

# In[ ]:


df.ingredients = df.ingredients.str.lower()
testset.ingredients = testset.ingredients.str.lower()


# Lets TOKENIZE the data now. (the processing of splitting into individual words)

# In[ ]:


df.ingredients = df.ingredients.apply(lambda x: word_tokenize(x))
testset.ingredients = testset.ingredients.apply(lambda x: word_tokenize(x))


# Lets LEMMATIZE the data now (Since i believe that dataset might have different representation of same words, like the olives and olive, tomatoes and tomato, which represent the same word)

# In[ ]:


lemmatizer = WordNetLemmatizer()


# In[ ]:


def lemmat(wor):
    l = []
    for i in wor:
        l.append(lemmatizer.lemmatize(i))
    return l


# In[ ]:


df.ingredients = df.ingredients.apply(lemmat)
testset.ingredients = testset.ingredients.apply(lemmat)


# In[ ]:


df.ingredients[0]


# In[ ]:


testset.ingredients[0]


# Observe that olives converted to olive, tomatoes to tomato etc, many words are now in their root form.

# In[ ]:


type(df.ingredients[0])


# Lemmatization converted it back to list, so change to str again and remove the unncessary words.

# In[ ]:


df.ingredients = df.ingredients.astype('str')
df.ingredients = df.ingredients.str.replace("["," ")
df.ingredients = df.ingredients.str.replace("]"," ")
df.ingredients = df.ingredients.str.replace("'"," ")
df.ingredients = df.ingredients.str.replace(","," ")


# In[ ]:


testset.ingredients = testset.ingredients.astype('str')
testset.ingredients = testset.ingredients.str.replace("["," ")
testset.ingredients = testset.ingredients.str.replace("]"," ")
testset.ingredients = testset.ingredients.str.replace("'"," ")
testset.ingredients = testset.ingredients.str.replace(","," ")


# In[ ]:


type(df.ingredients[0])


# In[ ]:


df.ingredients[0]


# Now our data looks good for vectorization.

# In[ ]:


#vect = HashingVectorizer ()
vect = TfidfVectorizer()


# In[ ]:


features = vect.fit_transform(df.ingredients)


# In[ ]:


features


# So, now our features has 2826 features, which are created by the process of vectorization.

# Lets visualize some random features.

# In[ ]:


#vect.get_feature_names()


# Lets vectorize our testset as well, we only tranform it with already fitted model

# In[ ]:


testfeatures = vect.transform(testset.ingredients)


# In[ ]:


testfeatures


# Lets create our labels now, which is obviously cuisine column. Lets labelencode it so that they convert to numerical lables, which usually might give better prediction results. Not a necessary step tho

# In[ ]:


encoder = LabelEncoder()
labels = encoder.fit_transform(df.cuisine)


# Lets split the dataset into training and testing parts

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


# Check the shapes, to make sure.

# In[ ]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# # Data Modeling

# In[ ]:


#logreg = LogisticRegression(C=10,solver='lbfgs', multi_class='multinomial',max_iter=400)
#logreg.fit(X_train,y_train)


# In[ ]:


#print("Logistic Regression accuracy",logreg.score(X_test, y_test))


# In[ ]:


#logreg.predict(X_test)


# In[ ]:


#from sklearn import linear_model
#sgd = linear_model.SGDClassifier()
#sgd.fit(X_train, y_train)


# In[ ]:


#print("SGD classifier accuracy",sgd.score(X_test, y_test))


# In[ ]:


#from sklearn.svm import LinearSVC
#linearsvm = LinearSVC(C=1.0,random_state=0,multi_class='crammer_singer',dual = False, max_iter = 1500)
#linearsvm.fit(X_train, y_train)


# In[ ]:


#print("Linear SVM accuracy", linearsvm.score(X_test, y_test))


# Now, lets try our luck with neural networks.

# # NEURAL NETWORK'S

# I have tried both Keras and tensorflow (Of course the backend is same), but Keras code looks simpler and clear.

# For Neural Networks we need to have the dense array's as inputs and preferably one hot encoding for lables.
# So, lets create lables.

# In[ ]:


#labelsNN = df.cuisine


# Convert it to one hot formatting, there are many ways to do, i prefer to do this way.

# In[ ]:


#labelsNN = pd.get_dummies(labelsNN)


# Convert it to arrays, you can do by values method or np.array() both are same

# In[ ]:


#labelsNN = labelsNN.values


# Here's how the one hot encoding looks like.

# In[ ]:


#labelsNN[0]


# Our labels are ready, now we need the features, we have already created the features above but it was sparse matrix, which neural network doesnt like, so convert to dense arrays.

# In[ ]:


#from scipy.sparse import csr_matrix
#sparse_dataset = csr_matrix(features)
#featuresNN = sparse_dataset.todense()


# Here's how the features look like.

# In[ ]:


#featuresNN[0]


# Split the dataset.

# In[ ]:


#X_trainNN, X_testNN, y_trainNN, y_testNN = train_test_split(featuresNN, labelsNN, test_size=0.2)


# In[ ]:


#print(X_trainNN.shape, X_testNN.shape, y_trainNN.shape, y_testNN.shape)


# In[ ]:


#numfeat = X_trainNN.shape[1]


# # KERAS

# In[ ]:


#import keras
#from keras.layers import *


# A sequential NN with 300,500 and 400 nodes in first,second and third layers resp.

# The loss is categorical cross entropy and the optimizer is adam with default learning rate.
# We can tweak a lot of parameters like the no of nodes, epochs, batchsize etc to improve accuracy.

# In[ ]:


#model = keras.models.Sequential()
#model.add(Dense(300,input_dim = numfeat,activation = 'relu'))
#model.add(Dense(500,activation = 'relu'))
#model.add(Dense(400,activation = 'relu'))
#model.add(Dense(20,activation='softmax'))
#model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['categorical_accuracy'])
#model.fit(X_trainNN,y_trainNN,epochs=50,shuffle=True, verbose =2,batch_size=500)


# In[ ]:


#print("Accuracy with KERAS" ,model.evaluate(X_testNN,y_testNN)[1])


# I have trained with KERAS on my pc for few times and achieved max accuracy of 0.81.

# Now, we have achieved almost similar accuracies in all the above models, I dont prefer NN's on this data as it is computationally very expensive.

# # PREDICTION

# I prefer just using the logisticRegression or linearsvm for predictions, but linearSVC also has almost same results.
# I'm not predict using Keras or Tensorflow, since it needs an extra two steps to convert the labels, which I dont want to waste my time on.

# In[ ]:


#linearsvmfinal = LinearSVC(C=1.0,random_state=0,multi_class='crammer_singer',dual = False, max_iter = 1500)
#linearsvmfinal.fit(features,labels)


# In[ ]:


import lightgbm as lgb


# In[ ]:


gbm = lgb.LGBMClassifier(objective="mutliclass",n_estimators=10000,num_leaves=512)
gbm.fit(X_train,y_train,verbose = 300)


# In[ ]:


pred = gbm.predict(testfeatures)


# In[ ]:


#pred = linearsvmfinal.predict(testfeatures)


# In[ ]:


predconv = encoder.inverse_transform(pred)


# In[ ]:


sub = pd.DataFrame({'id':testset.id,'cuisine':predconv})


# In[ ]:


output = sub[['id','cuisine']]


# In[ ]:


output.to_csv("outputfile.csv",index = False)


# # END

# # NOTES:
# 1) You can achieve better accuracy by ensembling the model, i will update this very soon.
# 2) Neural Network has even scored an accuracy of 0.81 but the computation is very time taking.
# 3) I have not used my time on visualizing the dataset.(which is not needed for this submission).
# 4) Please comment for any questions, doubts or suggestions.
# 
#  THANK YOU
#  
# # please UPVOTE, if you like.

# In[ ]:




