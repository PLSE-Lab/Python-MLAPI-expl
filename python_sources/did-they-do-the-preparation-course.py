#!/usr/bin/env python
# coding: utf-8

# ![](http://)** Predict Using a Random Forest Classifier**
# 
# 
# We are using scikit-learn Random Forest Classifier to predict, if a particular student has already completed **test preparation course** .
# 
# * So given how well they did in the course, we predict if they did the preparation course before doing the course.

# Import libraries

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as snb


# In[ ]:


# read csv
df = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")


# * Read top few rows from the file using the head() method of Pandas.

# In[ ]:


df.head()


# * Check missing values
# * .<code>isnull()</code> and <code>sum()</code> is used to find whether there are any missing values in the CSV file.

# In[ ]:


df.isnull().sum()


# * Explore the target
# 

# * get test preparation course values count

# In[ ]:



df['test preparation course'].value_counts()


# * Turn categorical values of the target into number

# In[ ]:


mapping = {"none" : 0, "completed" : 1}
df['test preparation course'] = df['test preparation course'].map(mapping)
df.head()


# * plot the pairplot graph of the original dataset using the target 'test preparation course' as the hue of the graph
# 

# In[ ]:


import seaborn as sns
sns.pairplot(data=df,hue='test preparation course',plot_kws={'alpha':0.2},palette='Set1')


# # Categorical data
# panda.get_dummies is to edit fake variables of strings that you want to put in a tree or random forest.

# In[ ]:


df = pd.get_dummies(df, columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch'],drop_first = True)


# In[ ]:


df.columns


# In[ ]:


df.head()


# # Create x and y
# For column-based operation
#     

# In[ ]:


#Set the value of the axis parameter to 1
X= df.drop("test preparation course", axis = 1)
y = df["test preparation course"]


# # Split train and test 

# In[ ]:


# split train test data set
from sklearn.model_selection import train_test_split
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y)


# ## Random Forest Classifier

# <p>A random forest is* a meta estimator* that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.</p>
# 
# <code>n_estimators</code> <i>integer, optional (default=10)</i>  The number of trees in the forest.<br/>
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(n_estimators = 100, max_features= 10) 


# **Fit X_Train and y_Train**

# In[ ]:


RandomForest.fit(X_Train, y_Train) 


#  * Score of the training data 

# In[ ]:


print("Accuracy train:", RandomForest.score(X_Train, y_Train))


# * Score of the testing data 
# 

# In[ ]:


print("Accuracy test:",RandomForest.score(X_Test, y_Test))


# ****Looks like the model is not able to generalise very well on unseen data. Let's investigate more evaluation metrics

# In[ ]:


predictions = RandomForest.predict(X_Test)


# * Classification report of your true labels y_test compared to the predictions

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_Test,predictions))


# In[ ]:


#from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_Test,predictions)
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 


# In[ ]:


n_features = X.shape[1]
plt.barh(range(n_features),RandomForest.feature_importances_)
plt.yticks(np.arange(n_features),df.columns[1:])


# 1. # Another approach - Logistic Regression and polynomial features

# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:





# In[ ]:


df["mean_grade"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
df["math score_squared"] = df["math score"] * df["math score"]
df["reading score_squared"] = df["reading score"] * df["reading score"]
df["writing score_squared"] = df["writing score"] * df["writing score"]


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


#X= df[['math score', 'reading score','writing score', 'gender_male','mean_grade', 'math score_squared', 'reading score_squared','writing score_squared']]
X = df.drop("test preparation course", 1)
y = df["test preparation course"]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
# split train test data set

from sklearn.model_selection import train_test_split
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() 
model.fit(X_Train, y_Train) 

print (model.score(X_Train, y_Train))#score of train
print (model.score(X_Test, y_Test))#score to test


# In[ ]:





# In[ ]:




