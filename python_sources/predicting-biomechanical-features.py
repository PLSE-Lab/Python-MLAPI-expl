#!/usr/bin/env python
# coding: utf-8

# # Predicting whether a patient's biomechanical features are normal or abnormal
# by Kevin Young
# 
# I will be using the dataset provided by UCI Machine Learning.
# 
# The data contains 310 instances of patients' features with six biomechanical attributes which come from the shape and orientation of the pelvis and lumbar spine:
# 
# - pelvic incidence
# - pelvic tilt
# - lumbar lordosis angle
# - sacral slope
# - pelvic radius
# - grade of spondylolisthesis

# ## Preparing the data
# First I import the data file into a Pandas dataframe. Fortunatley, there are no missing values in the data set, so we won't have to do any cleaning.

# In[ ]:


import pandas as pd

patients_data = pd.read_csv('../input/column_2C_weka.csv')
patients_data.head()


# Now I convert the dataframes into numpy arrays to be used by scikit_learn. We have one array that contains the class, another array with the feature data and another array with the feature name labels.

# In[ ]:


all_features = patients_data[['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']].values

all_classes = patients_data['class'].values

feature_names = ['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']

all_features


# Now I will need to normalise the input data.

# In[ ]:


from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)
all_features_scaled


# ## Logistic Regression
# 
# Given this is just a binary classification problem, I will first try logistic regression and see how high this accuracy is.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

clf = LogisticRegression()
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
cv_scores.mean()


# ## Decision Trees
# 
# Before using the DecisionTreeClassifier, I will create a train/test split of the data - 80% for training and 20% for testing.

# In[ ]:


import numpy
from sklearn.model_selection import train_test_split

numpy.random.seed(1234)

(training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(all_features_scaled, all_classes, train_size= 0.8, random_state = 1)


# Now I fit a DecisionTreeClassifier to the training data.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=1)

clf.fit(training_inputs, training_classes)


# Measuring the accuracy of the decision tree model using the test data.

# In[ ]:


clf.score(testing_inputs, testing_classes)


# Now I am trying K-Fold cross validation to further help avoid overfitting (K=10).

# In[ ]:


clf = DecisionTreeClassifier(random_state=1)

cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)

cv_scores.mean()


# Now I will also try a RandomForestClassifier to see if that accuracy is any better.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, random_state=1)
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)

cv_scores.mean()


# ## Support Vector Machines
# 
# svm.SVC has different kernels which may vary in performance. I will try linear, rbf, sigmoid and poly and see which results in the highest accuracy.

# In[ ]:


from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
cv_scores.mean()


# In[ ]:


svc = svm.SVC(kernel='rbf', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
cv_scores.mean()


# In[ ]:


svc = svm.SVC(kernel='sigmoid', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
cv_scores.mean()


# In[ ]:


svc = svm.SVC(kernel='poly', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
cv_scores.mean()


# ## K-Nearest Neighbours
# 
# In attempt to find the best value of K, I will use a for loop to iterate through different values of K from 1 to 50 and compare the resulting accuracy from each value of K.

# In[ ]:


from sklearn import neighbors

for i in range(1, 51):
    clf = neighbors.KNeighborsClassifier(n_neighbors= i)
    cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
    print(i, ":", cv_scores.mean())


# ## Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

scaler = preprocessing.MinMaxScaler()
all_features_minmax = scaler.fit_transform(all_features)

clf = MultinomialNB()
cv_scores = cross_val_score(clf, all_features_minmax, all_classes, cv=10)

cv_scores.mean()


# ## Neural Networks

# In[ ]:


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
cv_scores = cross_val_score(estimator, all_features_scaled, all_classes, cv=10)
cv_scores.mean()


# ## Conclusion
# 
# It seems the best machine learning model was the model using a SVM with a linear kernel hyperparameter. It yielded the highest accuracy being 83.5%.
