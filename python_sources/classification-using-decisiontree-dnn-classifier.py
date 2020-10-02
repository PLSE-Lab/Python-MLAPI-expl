#!/usr/bin/env python
# coding: utf-8

# # Sloan Digital Sky Survey

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")


# In[ ]:


data.head()


# In[ ]:


# drop the object id columns, they are of no use in the analysis
data.drop(['objid','specobjid'], axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


sns.countplot(x=data['class'])


# In[ ]:


def change_category_to_number(classCat):
    if classCat=='STAR':
        return 0
    elif classCat=='GALAXY':
        return 1
    else:
        return 2


# In[ ]:


# assign a numerical value to the categorical field of class, by using the above function
data['classCat'] = data['class'].apply(change_category_to_number)


# In[ ]:


data.head()


# In[ ]:


sns.pairplot(data[['u','g','r','i']])


# In[ ]:


data.drop(['run','rerun','camcol','field','class'],axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


X = data.drop('classCat', axis=1)
y = data['classCat']


# ### Perform train and test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=128)


# ## Decision Tree Classifier

# In[ ]:


dtClassifer = DecisionTreeClassifier(max_leaf_nodes=15,random_state=0)


# In[ ]:


dtClassifer.fit(X_train, y_train)


# ### Prediction on test data

# In[ ]:


prediction = dtClassifer.predict(X_test)


# In[ ]:


prediction[:10]


# In[ ]:


y_test[:10]


# ### Measure accuracy of the classifier

# In[ ]:


accuracy_score(y_true=y_test, y_pred=prediction)


# ## Logistic Regression Classifier

# In[ ]:


lrClassifier = LogisticRegression()


# In[ ]:


lrClassifier.fit(X_train,y_train)


# ### Prediction on test data

# In[ ]:


prediction = lrClassifier.predict(X_test)


# In[ ]:


prediction[:10]


# In[ ]:


y_test[:10]


# ### Measure accuracy of the model

# In[ ]:


accuracy_score(y_true=y_test, y_pred=prediction)


# # Neural Network using Tensorflow and Keras

# In[ ]:


featcols = [tf.feature_column.numeric_column('ra'),
            tf.feature_column.numeric_column('dec'),
            tf.feature_column.numeric_column('u'),
            tf.feature_column.numeric_column('g'),
            tf.feature_column.numeric_column('r'),
            tf.feature_column.numeric_column('i'),
            tf.feature_column.numeric_column('z'),
            tf.feature_column.numeric_column('redshift'),
            tf.feature_column.numeric_column('plate'),
            tf.feature_column.numeric_column('mjd'),
            tf.feature_column.numeric_column('fiberid')
           ]


# ### Linear Classifier 

# In[ ]:


model = tf.estimator.LinearClassifier(n_classes=3,
                                      optimizer=tf.train.FtrlOptimizer(l2_regularization_strength=0.1,learning_rate=0.01),
                                     feature_columns=featcols)


# In[ ]:


data.head()


# In[ ]:


def get_input_fn(num_epochs,n_batch,shuffle):
    return tf.estimator.inputs.pandas_input_fn(
        x=X_train,
        y=y_train,
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle
    )


# In[ ]:


model.train(input_fn=get_input_fn(100,128,True),steps=1000)


# In[ ]:


def evaluate_fn(num_epochs,n_batch,shuffle):
    return tf.estimator.inputs.pandas_input_fn(
        x=X_test,
        y=y_test,
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle
    )


# In[ ]:


model.evaluate(input_fn=evaluate_fn(100,128,True),steps=1000)


# ### DNN Classifier

# In[ ]:


dnn_model = tf.estimator.DNNClassifier(n_classes=3,
                                       feature_columns=featcols,
                                       hidden_units=[1024,512,256,32,3],
                                       activation_fn=tf.nn.relu,
                                       optimizer='Adam',
                                       dropout=0.2,
                                      )


# In[ ]:


dnn_model.train(input_fn=get_input_fn(100,128,True),steps=1000)


# In[ ]:


dnn_model.evaluate(input_fn=evaluate_fn(100,128,True),steps=1000)


# ### DNN Linear Combined Classifier

# In[ ]:


dnnlcc_model = tf.estimator.DNNLinearCombinedClassifier(n_classes=3,dnn_activation_fn='relu',dnn_dropout=0.2,dnn_hidden_units=[1024,512,256,32,3],dnn_optimizer='Adam',dnn_feature_columns=featcols,linear_feature_columns=featcols)


# In[ ]:


dnnlcc_model.train(input_fn=get_input_fn(100,128,True),steps=1000)


# In[ ]:


dnnlcc_model.evaluate(input_fn=evaluate_fn(100,128,True),steps=1000)


# For Deep Neural Network to work better, you need more dataset.

# In[ ]:




