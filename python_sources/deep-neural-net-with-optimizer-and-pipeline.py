#!/usr/bin/env python
# coding: utf-8

# Maryam Ashoori 
# 
# _Requirements: Scikit-learn 0.20_
# 
# This notebook provides an easy implementation of a deep neural net for survival prediction

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import tensorflow as tf

# Feature Engineering
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Restrating the tensor graph
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)    
reset_graph()

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.head()


# The test data does not contain survival labels: your goal is to train the best model you can using the training data, then make your predictions on the test data and upload them to Kaggle to see your final score. This code gives you slightly more than 80% accuracy.
# 
# We use the validation set to test out the performance of the training models. The simplest way to create a validation set is to use train_test_split function and specify the percentage of data that you would want to allocate for validation. This will splits the training data into two sets of training and validation. For example, the code below creates a validation set with 20% of the data.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_data, train_data["Survived"], test_size=0.20, random_state=42)


# In[ ]:


# I created one preprocessing pipelines for processing both numeric and categorical data.
numeric_features = ['Age','Fare', 'SibSp', 'Parch']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) #Please note all the numerical values are scaled using StandardScaler().
])

categorical_features = ['Sex', 'Pclass', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='S')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[ ]:


X_train_prepared = preprocessor.fit_transform(X_train)

X_test_prepared = preprocessor.fit_transform(X_test)


# # Deep Neural Net

# In[ ]:


feature_columns = [tf.feature_column.numeric_column("X", shape=(X_train_prepared.shape[1],1))]


# In[ ]:


dnn_clf = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 9, 10, 10, 10],
    n_classes=2,
    activation_fn=tf.nn.elu,
    batch_norm=False,
    dropout=0.5,
    optimizer=tf.train.ProximalGradientDescentOptimizer(
        learning_rate=0.01,
        l1_regularization_strength=0.1,
        l2_regularization_strength=0.1),                                      
#     optimizer=tf.train.MomentumOptimizer(
#       learning_rate=0.1,
#       momentum=0.2,
#       use_nesterov=True)
)


# In[ ]:


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": np.array(X_train_prepared)},
    y=np.array(y_train.values.reshape((len(y_train),1))),
    num_epochs=100, 
    shuffle=True
)
dnn_clf.train(input_fn=train_input_fn)


# In[ ]:


test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": np.array(X_test_prepared)},
    y=np.array(y_test.values.reshape((len(y_test),1))),
    num_epochs=1, #We just want to use one epoch since this is only to score.
    shuffle=False  #It isn't necessary to shuffle the cross validation 
)


# In[ ]:


# Evaluate accuracy.
accuracy_score = dnn_clf.evaluate(input_fn=test_input_fn)
print("\nTest Accuracy: {0:f}\n".format(accuracy_score['accuracy']))


# In[ ]:


test_set_prepared = preprocessor.fit_transform(test_data)


# In[ ]:


prediction_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": np.array(test_set_prepared)},
    y=None,
    num_epochs=1, #We just want to use one epoch since this is only to score.
    shuffle=False  #It isn't necessary to shuffle the cross validation 
)
pred = dnn_clf.predict(input_fn=prediction_input_fn)


# In[ ]:


predictions = np.array([])
pred_list = list(pred)
for p in pred_list:
    predictions = np.append(predictions,p['class_ids'][0])
#cast from string to integer
predictions = predictions.astype(int)


# In[ ]:


result = pd.DataFrame(columns=["PassengerId", "Survived"])
result["PassengerId"] = test_data['PassengerId']
result["Survived"] = predictions
result.to_csv("Submission-tf1.csv", index=False)

