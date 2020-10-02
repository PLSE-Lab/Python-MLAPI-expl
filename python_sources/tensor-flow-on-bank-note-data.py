#!/usr/bin/env python
# coding: utf-8

# ## Data
# The data consists of 5 columns:
# 
# * variance of Wavelet Transformed image (continuous)
# * skewness of Wavelet Transformed image (continuous)
# * curtosis of Wavelet Transformed image (continuous)
# * entropy of image (continuous)
# * class (integer)
# 
# Where class indicates whether or not a Bank Note was authentic.
# 

# ## Get the Data
# 
# ** Use pandas to read in the bank_note_data.csv file **

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv("../input/bank_note_data.csv")


# ** Check the head of the Data **

# In[ ]:


data.head()


# ## EDA
# 
# We'll just do a few quick plots of the data.
# 
# ** Import seaborn and set matplolib inline for viewing **

# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Create a Countplot of the Classes (Authentic 1 vs Fake 0) **

# In[ ]:


sns.countplot(x='Class',data=data)


# ** Create a PairPlot of the Data with Seaborn, set Hue to Class **

# In[ ]:


sns.pairplot(data,hue='Class')


# ## Data Preparation 
# 
# When using Neural Network and Deep Learning based systems, it is usually a good idea to Standardize your data, this step isn't actually necessary for our particular data set, but let's run through it for practice!
# 
# ### Standard Scaling
# 
# ** 

# In[ ]:


from sklearn.preprocessing import StandardScaler


# **Create a StandardScaler() object called scaler.**

# In[ ]:


scaler = StandardScaler()


# **Fit scaler to the features.**

# In[ ]:


scaler.fit(data.drop('Class',axis=1))


# **Use the .transform() method to transform the features to a scaled version.**

# In[ ]:


scaled_features = scaler.fit_transform(data.drop('Class',axis=1))


# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[ ]:


df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()


# ## Train Test Split
# 
# ** Create two objects X and y which are the scaled feature values and labels respectively.**

# In[ ]:


X = df_feat


# In[ ]:


y = data['Class']


# ** Use the .as_matrix() method on X and Y and reset them equal to this result. We need to do this in order for TensorFlow to accept the data in Numpy array form instead of a pandas series. **

# In[ ]:


X = X.as_matrix()
y = y.as_matrix()


# ** Use SciKit Learn to create training and testing sets of the data as we've done in previous lectures:**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# # Contrib.learn
# 
# ** Import tensorflow.contrib.learn.python.learn as learn**

# In[ ]:


import tensorflow as tf
import tensorflow.contrib.learn.python

from tensorflow.contrib.learn.python import learn as learn


# ** Create an object called classifier which is a DNNClassifier from learn. Set it to have 2 classes and a [10,20,10] hidden unit layer structure:**

# In[ ]:


#classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2)
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
classifier = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=10)


# ** Now fit classifier to the training data. Use steps=200 with a batch_size of 20. You can play around with these values if you want!**
# 
# *Note: Ignore any warnings you get, they won't effect your output*

# In[ ]:


tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")


# In[ ]:


classifier.fit(X_train, y_train, steps=200, batch_size=20)


# ## Model Evaluation
# 
# ** Use the predict method from the classifier model to create predictions from X_test **

# In[ ]:


note_predictions = classifier.predict(X_test)


# ** Now create a classification report and a Confusion Matrix. Does anything stand out to you?**

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


lst = list(note_predictions)


# In[ ]:


print(confusion_matrix(y_test,lst))


# In[ ]:


print(classification_report(y_test,lst))


# In[ ]:


# !pip install -q tf-nightly-2.0-preview
# # Load the TensorBoard notebook extension
# %load_ext tensorboard


# In[ ]:


# import tensorflow as tf
# import datetime, os

# logs_base_dir = "./logs"
# os.makedirs(logs_base_dir, exist_ok=True)
# %tensorboard --logdir {logs_base_dir}


# In[ ]:


# %load_ext tensorboard.notebook
# %tensorboard --logdir logs
# %reload_ext tensorboard.notebook


# ## Optional Comparison
# 
# ** You should have noticed extremely accurate results from the DNN model. Let's compare this to a Random Forest Classifier for a reality check!**
# 
# **Use SciKit Learn to Create a Random Forest Classifier and compare the confusion matrix and classification report to the DNN model**

# ## 1.RandomForest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


rfc_preds = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,rfc_preds))


# In[ ]:


rfc_score_train = rfc.score(X_train, y_train)
print("Training score: ",rfc_score_train)
rfc_score_test = rfc.score(X_test, y_test)
print("Testing score: ",rfc_score_test)


# In[ ]:


print(confusion_matrix(y_test,rfc_preds))


# ** It should have also done very well, but not quite as good as the DNN model. Hopefully you have seen the power of DNN! **

# ## 2.logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
logis.fit(X_train, y_train)
logis_score_train = logis.score(X_train, y_train)
print("Training score: ",logis_score_train)
logis_score_test = logis.score(X_test, y_test)
print("Testing score: ",logis_score_test)


# ## 3.Decision tree

# In[ ]:


#decision tree
from sklearn.ensemble import RandomForestClassifier
dt = RandomForestClassifier()
dt.fit(X_train, y_train)
dt_score_train = dt.score(X_train, y_train)
print("Training score: ",dt_score_train)
dt_score_test = dt.score(X_test, y_test)
print("Testing score: ",dt_score_test)


# In[ ]:





# In[ ]:


#Model comparison
models = pd.DataFrame({
        'Model'          : ['Logistic Regression',  'Decision Tree', 'Random Forest'],
        'Training_Score' : [logis_score_train,  dt_score_train, rfc_score_train],
        'Testing_Score'  : [logis_score_test, dt_score_test, rfc_score_test]
    })
models.sort_values(by='Testing_Score', ascending=False)


# In[ ]:




