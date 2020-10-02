#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dftrain = pd.read_csv('../input/titanic_train (1).csv')


# In[4]:


dftrain.head()


# In[5]:


print(dftrain.info())


# In[6]:


dftrain.isnull().values.any()


# In[7]:


dftrain.columns


# In[8]:


dftrain['embark_town'].unique()


# In[9]:


dftrain['deck'].unique()


# In[10]:


def finding_unknown(df):
    for col in df.columns:
        number = 0
        for i in range(0,len(df[col])):
            if i != None:
                if df[col][i] == 'unknown':
                    number+=1
        print(col,'=',number,'\n')


# In[11]:


finding_unknown(dftrain)


# In[12]:


#as the number of unknown values in deck is huge, it is very unlikely that it is going to help in predicting the output
dftrain = dftrain.drop('deck',axis=1)


# In[13]:


dftrain.head()


# In[14]:


dftrain.loc[dftrain['embark_town'] == 'unknown']


# In[15]:


#as we have just one row, its better to remove the unknown data
dftrain = dftrain.drop(dftrain.index[48])


# In[16]:


dftrain.info()


# In[17]:


dftrain.head()


# In[18]:


X = dftrain.drop(['survived','n_siblings_spouses','parch'],axis=1)


# In[19]:


y = dftrain['survived']


# In[20]:


#creating numeric feature columns
feature_columns = []
feature_columns.append(tf.feature_column.numeric_column('age',dtype=tf.float32))
feature_columns.append(tf.feature_column.numeric_column('fare',dtype=tf.float32))


# In[21]:


#creating categorical feature columns
categorical_columns = ['sex','class','embark_town','alone']
tc = tf.feature_column
def create_cat_featcol(cat_cols):
    for feature_name in cat_cols:
        vocab = dftrain[feature_name].unique()
        feature_columns.append(tc.indicator_column(tc.categorical_column_with_vocabulary_list(feature_name,vocab)))


# In[22]:


create_cat_featcol(categorical_columns)


# In[23]:


feature_columns


# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)


# In[25]:


#creating an input function
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=5,shuffle=True)


# In[26]:


#defining a classifier
classifier = tf.estimator.DNNClassifier(hidden_units=[20,20,20,20], n_classes=2,feature_columns=feature_columns)


# In[27]:


#training the classifier on the input function
classifier.train(input_fn=input_func,steps=50)


# In[28]:


#creating a prediction function
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


# In[29]:


#creating a list of the predictions
note_predictions = list(classifier.predict(input_fn=pred_fn))


# In[30]:


#a sample prediction in the list
note_predictions[0]


# In[31]:


final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])


# In[32]:


from sklearn.metrics import classification_report,confusion_matrix


# In[33]:


print(classification_report(y_test,final_preds))


# In[34]:


#using Boosted Tress as a classifier
#adding L2 regularisation and decreasing the num of tress immensely improved the results 
#because the data is very less and it is very probable of the model to overfit
classifier_2 = tf.estimator.BoostedTreesClassifier(feature_columns = feature_columns,n_batches_per_layer=1,l2_regularization=0.1,n_trees=50)


# In[35]:


classifier_2.train(input_fn=input_func,steps=50)


# In[36]:


note_predictions_2 = list(classifier_2.predict(input_fn=pred_fn))
    


# In[37]:


final_preds_2  = []
for pred in note_predictions_2:
    final_preds_2.append(pred['class_ids'][0])


# In[38]:


print(classification_report(y_test,final_preds_2))

