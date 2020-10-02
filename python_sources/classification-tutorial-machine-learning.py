#!/usr/bin/env python
# coding: utf-8

# # Description of iris data
# The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.
# 
# It includes three iris species with <b>50 samples</b> each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
# 
# The columns in this dataset are:
# -  Id
# - SepalLengthCm
# - SepalWidthCm
# - PetalLengthCm
# - PetalWidthCm
# - Species
# Sepal Width vs. Sepal Length
# 
# 
# 
# -  Ref1: https://www.kaggle.com/uciml/iris
# - Ref2: http://archive.ics.uci.edu/ml/index.php
# - Ref3: http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv('../input/Iris.csv')

# First and last five observations or row


# In[ ]:


data.head(5)


# In[ ]:


data.tail(5)


# ### Classes in Species

# In[ ]:


# Our target is 'species' So need to check how many of them
print("Species")
print(data['Species'].unique())


# ### Description 

# In[ ]:


data.describe()


# ### Data Classes from original dataset

# In[ ]:


import seaborn as sns
sns.FacetGrid(data, hue="Species", size=6)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()

plt.show()


# ### Preprocessing Steps: Separating Independent Features and Dependant Feature(Response)

# In[ ]:


# Preprocessing
# Let Separate Features and Target for machine Learning
# Step1 


features = list(data.columns[1:5])            # SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	
target = data.columns[5]                      # Species

print('Features:',features)
print('Target:',target)

# store feature matrix in "X"
X = data.iloc[:,1:5]                          # slicing: all rows and 1 to 4 cols

# store response vector in "y"
y = data.iloc[:,5]                            # slicing: all rows and 5th col


print(y.shape)
print(X.shape)


# ### Converting Response Variable 'Species' to numbers to train model and add column

# In[ ]:


# Read more: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y=le.transform(y)

# new col
data['EncodedSpecies'] = y

print('Classes:',le.classes_)
print('Response variable after encoding:',y)
data.tail(10)


# ### Classification Models
# 
# ### 1: K-nearest neighbors (KNN) classification
# - Pick a value for K.
# - Search for the K observations in the training data that are "nearest" to the measurements of the unknown iris.
# - Use the most popular response value from the K nearest neighbors as the predicted response value for the unknown iris.

# ### KNN Model

# In[ ]:


#Step2: Model

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)


# ### Prediction for some feature values: 3, 5, 4, 2 

# In[ ]:


#Step3: Prediction for new observation
value = knn.predict([[3, 5, 4, 2]])
print('prediction value:',value)

print('Predicted Class' , data.loc[data['EncodedSpecies'] == 2, 'Species'].values[0])
#data.loc[data['EncodedSpecies'] == 2, 'Species'].values[0]


# In[ ]:


# more predictions for other rows

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]                        # Consider them as two new rows of input features in X
knn.predict(X_new)


# ### KNN for K=5

# In[ ]:


# Different value of K 
# instantiate the model (using the value K=5)

knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
print(knn.predict(X_new))

kypred = knn.predict(X)


# ### How to check the best value for K?

# In[ ]:


# For an optimal value of K for KNN

from sklearn import metrics
v=[]




k_range = list(range(1, 50))
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    # fit the model with data
    knn.fit(X, y)
    k_pred = knn.predict(X)
    v.append( metrics.accuracy_score(y, k_pred))


import matplotlib.pyplot as plt
plt.plot(k_range,v,c='Orange',)
plt.show()


# # Logistic Regression

# In[ ]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response values for the observations in column: [3, 5, 4, 2]
logreg.predict([[3, 5, 4, 2]]) # Col vector # See previous result


y_pred = logreg.predict(X)

print(y_pred)


# # Classification accuracy #1:

# In[ ]:


# 1 KNN ACCURACY

# compute classification accuracy for the KNN model
from sklearn import metrics
print(metrics.accuracy_score(y, kypred))


# In[ ]:


# 2 
# compute classification accuracy for the logistic regression model
from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))


# # Evaluation procedure #2: Train/test split

# In[ ]:


# STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# # Logistic Regression Model with Train Test split

# In[ ]:


logres = LogisticRegression()
logres.fit(X_train,y_train) # train data


# predict from test
log_pred = logres.predict(X_test)

# check accuracy
import sklearn.metrics as mt
mt.accuracy_score(log_pred,y_test)


# # KNN with Train Test

# In[ ]:



from sklearn import metrics
v=[]


k_range = list(range(1, 50))
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    # fit the model with data
    knn.fit(X_train, y_train)
    k_pred = knn.predict(X_test)
    v.append( metrics.accuracy_score(y_test, k_pred))



import matplotlib.pyplot as plt
plt.plot(k_range,v)

plt.show()


# In[ ]:


print('from above the best value is near:',10)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=12)
# fit the model with data
knn.fit(X_train, y_train)
k_pred = knn.predict(X_test)

metrics.accuracy_score(y_test, k_pred)


# # TO BE CONTINUED

# In[ ]:




