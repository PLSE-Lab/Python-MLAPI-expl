#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


# In[ ]:


exo_train = pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')
exo_test = pd.read_csv('../input/kepler-labelled-time-series-data/exoTest.csv')


# In[ ]:


exo_train.head()


# In[ ]:


# checking to see if there is any row with nulll values
exo_train.isnull().values.any()


# In[ ]:


exo_train['LABEL'].value_counts()


# In[ ]:


## since, the data is unbalanced we will use decision tree, random forest and gradient boosting as they perform better with unbalanced data and we will use F1-Score as a 


# In[ ]:


x_train = exo_train.drop('LABEL', axis=1)
y_train = exo_train['LABEL']


# ## Implementing decision tree

# In[ ]:


decisiontree_model = DecisionTreeClassifier()


# In[ ]:


decisiontree_model.fit(x_train,y_train)


# In[ ]:


x_test = exo_test.drop('LABEL', axis=1)
y_test = exo_test['LABEL']


# In[ ]:


y_predict = decisiontree_model.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_predict)


# In[ ]:


confusion_matrix(y_test, y_predict)


# In[ ]:


f1_score(y_test, y_predict)


# In[ ]:





# ## Implementing Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


randomforest_model = RandomForestClassifier(n_estimators=200,bootstrap=True,max_features='sqrt')


# In[ ]:


randomforest_model.fit(x_train,y_train)


# In[ ]:


yrandomforest_predict = randomforest_model.predict(x_test)


# In[ ]:


accuracy_score(y_test,yrandomforest_predict)


# In[ ]:


confusion_matrix(y_test, yrandomforest_predict)


# In[ ]:


f1_score(y_test, yrandomforest_predict)


# <p>Both the accuracy and f1 score has been improved compraed to decision tree</p>

# ## Implementing Greadient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


learning_rates = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
for l_rate in learning_rates:
    gradient_model = GradientBoostingClassifier(n_estimators=200, learning_rate=l_rate, max_features=10, max_depth=2, random_state=60616)
    gradient_model.fit(x_train,y_train)
    ygradientboost_predict = gradient_model.predict(x_test)
    
    print('learning rate', l_rate)
    print('Accuracy is',accuracy_score(y_test,ygradientboost_predict))
    print('f1 score is',f1_score(y_test, yrandomforest_predict))
    print('\n')


# <p>we can see that the accuracy has been improved with gradient boost but f1 score remains the same as compared to random forest</p>

# ## TensorFlow boosted trees estimator

# In[ ]:


import tensorflow as tf
tf.random.set_seed(60616)


# In[ ]:


#converting the headers to list
numeric_column_headers = x_train.columns.values.tolist()


# In[ ]:


#converting the features into rwo  features around the mean as that TFBT estimator only takes bucketed features
# https://medium.com/ml-book/demonstration-of-tensorflow-feature-columns-tf-feature-column-3bfcca4ca5c4 (Good link)
bc_fn = tf.feature_column.bucketized_column
nc_fn = tf.feature_column.numeric_column
bucketized_features = [bc_fn(source_column=nc_fn(key=column),
                             boundaries=[x_train[column].mean()])
                       for column in numeric_column_headers]


# In[ ]:


all_features = bucketized_features


# In[ ]:


# converting label 2 to 1 and 1 to 0 as labels should be less than num_class - 1 for tf.estimator.BoostedTreesClassifier
y_train_new = pd.Series(np.where(y_train.values == 2, 1, 0),y_train.index)
y_test_new = pd.Series(np.where(y_test.values == 2, 1, 0),y_test.index)


# In[ ]:


batch_size = 32

def make_input_fn(dataframe, y, shuffle=True, n_epochs=None, batch_size=32):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), y))
        if shuffle:
            dataset = dataset.shuffle(len(dataframe))
        
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        
        # In memory training doesn't use batching.
        dataset = dataset.batch(batch_size)
        return dataset
    
    return input_fn


train_input_fn = make_input_fn(x_train, y_train_new)
test_input_fn = make_input_fn(x_test, y_test_new, shuffle=False, n_epochs=1)


# In[ ]:


n_trees = 100
tbts_model = tf.estimator.BoostedTreesClassifier(feature_columns=all_features, n_trees=n_trees, n_batches_per_layer=batch_size)


# In[ ]:


n_steps = 100
tbts_model.train(input_fn=train_input_fn,steps=n_steps)


# In[ ]:


results = tbts_model.evaluate(input_fn=test_input_fn)


# In[ ]:


print(pd.Series(results))


# <p>Accuracy is similar here too</p>

# In[ ]:




