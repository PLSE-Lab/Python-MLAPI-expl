#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Get the Data
# 
# Use pandas to read Dataset_spine.csv as a dataframe called train.
# 

# In[ ]:


train = pd.read_csv("../input/Dataset_spine.csv")


# Check out the info(), head(), and describe() methods on the data.

# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.columns


# # Exploratory Data Analysis
# 
# Let's do some data visualization!
# 

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Class_att',data=train,palette='RdBu_r')


# Data is not balanced.

# In[ ]:


plt.figure(figsize=(20,10))
c=train.corr()
sns.heatmap(c,cmap="BrBG",annot=True)


# # Train Test Split
# 
# Now its time to split our data into a training set and a testing set!
# 
# Use sklearn to split your data into a training set and a testing set as we've done in the past.
# 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X= train [['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6',]]
y= train['Class_att']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, 
                                                    random_state=101)


# # Training a Decision Tree Model
# 
# Let's start by training a single decision tree first!
# 
# **Import DecisionTreeClassifier**
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# **Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.**

# In[ ]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# **Predictions and Evaluation of Decision Tree**
# 
# Create predictions from the test set and create a classification report and a confusion matrix.
# 

# In[ ]:


predictions = dtree.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# # Training the Random Forest model
# 
# Now its time to train our model!
# 
# Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=600)


# In[ ]:


rfc.fit(X_train,y_train)


# **Predictions and Evaluation**
# 
# Let's predict off the y_test values and evaluate our model.
# 
#  Predict on the X_test data.
# 

# In[ ]:


predictions = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# Show the classification report and Confusion Matrix for the predictions.

# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# **What performed better the random forest or the decision tree?**

# Random Forest performed slightly better.
