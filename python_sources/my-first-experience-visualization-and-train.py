#!/usr/bin/env python
# coding: utf-8

# <br>
# <center> <b> ******Data Visualization****** </b> </center>
# <br>

# In[ ]:


# Module import
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/heart.csv')
data.head(5)


# ## Dataset Columns (Features)
# 

# - Age (age in years)
# - Sex (1 = male; 0 = female)
# - CP (chest pain type)
# - TRESTBPS (resting blood pressure (in mm Hg on admission to the hospital))
# - CHOL (serum cholestoral in mg/dl)
# - FPS (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# - RESTECH (resting electrocardiographic results)
# - THALACH (maximum heart rate achieved)
# - EXANG (exercise induced angina (1 = yes; 0 = no))
# - OLDPEAK (ST depression induced by exercise relative to rest)
# - SLOPE (the slope of the peak exercise ST segment)
# - CA (number of major vessels (0-3) colored by flourosopy)
# - THAL (3 = normal; 6 = fixed defect; 7 = reversable defect)
# - TARGET (1 or 0)

# ### Concise summary of a Data

# In[ ]:


data.info()


# ### Missing values detection

# In[ ]:


data.isnull().sum()


# <br>
# ## <center>  <b> **Visualization** </b> </center>
# <br>

# **Age**

# In[ ]:


plot = data[data.target == 1].age.value_counts().sort_index().plot(kind = "bar", figsize=(15,4), fontsize = 15)
plot.set_title("Age distribution", fontsize = 20)


# **Sex**

# In[ ]:


male = len(data[data.sex == 1])
female = len(data[data.sex == 0])
plt.pie(x=[male, female], explode=(0, 0), labels=['Male', 'Female'], autopct='%1.2f%%', shadow=True, startangle=90)
plt.show()


# **CP (chest pain type)**

# In[ ]:


x = [len(data[data['cp'] == 0]),len(data[data['cp'] == 1]), len(data[data['cp'] == 2]), len(data[data['cp'] == 3])]
plt.pie(x, data=data, labels=['CP(1) typical angina', 'CP(2) atypical angina', 'CP(3) non-anginal pain', 'CP(4) asymptomatic'], autopct='%1.2f%%', shadow=True,startangle=90)
plt.show()


# **TRESTBPS (resting blood pressure (in mm Hg on admission to the hospital))**

# In[ ]:


plot = data[data.target == 1].trestbps.value_counts().sort_index().plot(kind = "bar", figsize=(15,4), fontsize = 15)
plot.set_title("Resting blood pressure", fontsize = 20)


# **Chol (serum cholestoral in mg/dl)**

# In[ ]:


plt.hist([data.chol[data.target==0], data.chol[data.target==1]], bins=20,color=['blue', 'red'], stacked=True)
plt.legend(["Haven't Disease", "Have Disease"])
plt.title('Heart Disease Frequency for cholestoral ')
plt.ylabel('Frequency')
plt.xlabel('Chol in mg/dl')
plt.plot()


# **FPS (fasting blood sugar > 120 mg/dl)**

# In[ ]:


sizes = [len(data[data.fbs == 0]), len(data[data.fbs==1])]
labels = ['No', 'Yes']
plt.pie(x=sizes, labels=labels, explode=(0.1, 0), autopct="%1.2f%%", startangle=90,shadow=True)
plt.show()


# **Restecg (resting electrocardiographic results)**

# In[ ]:


sizes = [len(data[data.restecg == 0]), len(data[data.restecg==1]), len(data[data.restecg==2])]
labels = ['Normal', 'ST-T wave abnormality', 'definite left ventricular hypertrophy by Estes criteria']
plt.pie(x=sizes, labels=labels, explode=(0, 0, 0), autopct="%1.2f%%", startangle=90,shadow=True)
plt.show()


# **THALACH (maximum heart rate achieved)**

# In[ ]:


plt.hist([data.thalach[data.target==0], data.thalach[data.target==1]], bins=20,color=['blue', 'red'], stacked=True)
plt.legend(["Haven't Disease", "Have Disease"])
plt.title('Heart Disease Frequency for maximum heart rate achieved')
plt.ylabel('Frequency')
plt.xlabel('Heart rate')
plt.plot()


# **Exang**

# In[ ]:


sizes = [len(data[data.exang == 0]), len(data[data.exang==1])]
labels = ['No', 'Yes']
plt.pie(x=sizes, labels=labels, explode=(0.1, 0), autopct="%1.2f%%", startangle=90,shadow=True)
plt.show()


# **Slope**

# In[ ]:


sizes = [len(data[data.slope == 0]), len(data[data.slope==1]), len(data[data.slope==2])]
labels = ['Upsloping', 'Flat', 'Downssloping']
plt.pie(x=sizes, labels=labels, explode=(0, 0, 0), autopct="%1.2f%%", startangle=90,shadow=True)
plt.show()


# **Thal (thalassemia)**

# In[ ]:


sns.countplot('thal', data=data)
plt.title('Frequency for thal')
plt.ylabel('Frequency')
plt.show()


# **Data Preprocessing**
# 
# 'cp', 'thal' and 'slope' are categorical variables

# In[ ]:


cp = pd.get_dummies(data['cp'], prefix = "cp", drop_first=True)
thal = pd.get_dummies(data['thal'], prefix = "thal" , drop_first=True)
slope = pd.get_dummies(data['slope'], prefix = "slope", drop_first=True)


# Removing the first level.

# In[ ]:


new_data = pd.concat([data, cp, thal, slope], axis=1)
new_data.head(3)


# We don't need 'cp', 'thal', 'slope' columns so we will drop them

# In[ ]:


new_data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)
new_data.head(3)


# removing target columns from dataset

# In[ ]:


X = new_data.drop(['target'], axis=1)
y = new_data.target


# In[ ]:


print(X.shape)


# **Normalize the data**

# In[ ]:


X = (X - X.min())/(X.max()-X.min())
X.head(3)


# Split our Data. 80% - train, 20% - test data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# <br>
# ## <center>  <b> **Train** </b> </center>
# <br>

# **Logistic Regression**

# In statistics, the logistic model (or logit model) is a widely used statistical model that, in its basic form, uses a logistic function to model a binary dependent variable; many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model; it is a form of binomial regression. 

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# In[ ]:


from sklearn.model_selection import GridSearchCV


# Setting parameters for GridSearchCV

# In[ ]:


params = {'penalty':['l1','l2'],
         'C':[0.01,0.1,1,10,100],
         'class_weight':['balanced',None]}
lr_model = GridSearchCV(lr,param_grid=params,cv=10)


# In[ ]:


lr_model.fit(X_train,y_train)
lr_model.best_params_


# In[ ]:


lr = LogisticRegression(C=1, penalty='l2')
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# In[ ]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, lr.predict(X_test))
sns.heatmap(cm, annot=True)
plt.plot()


# Result (Logistic Regression) - 0.9016393442622951.
# 
# Our model (Logistic Regression) is giving good result.

# **K Nearest Neighbor**

# In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:
# 
# * In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
# * In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.
# 
# k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. The k-NN algorithm is among the simplest of all machine learning algorithms.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test)


# In[ ]:


for i in range(1,11):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print("k : ",i ,"score : ",knn.score(X_test, y_test), end="\n" )


# In[ ]:


#Confusion Matrix
cm = confusion_matrix(y_test, knn.predict(X_test))
sns.heatmap(cm, annot=True)
plt.plot()


# Result (K Nearest Neighbor) - 0.8688524590163934.

# **Decision Tree Classifier**

# In computer science, Decision tree learning uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modeling approaches used in statistics, data mining and machine learning. Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, y_train)
dt.score(X_test, y_test)


# In[ ]:


#Confusion Matrix
cm = confusion_matrix(y_test, dt.predict(X_test))
sns.heatmap(cm, annot=True)
plt.plot()


# Result (Decision Tree Classifier) - 0.8032786885245902.

# **Gradient Boosting Classifier**

# Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc.score(X_test, y_test)


# In[ ]:


#Confusion Matrix
cm = confusion_matrix(y_test, gbc.predict(X_test))
sns.heatmap(cm, annot=True)
plt.plot()


# **Gaussian NB**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb.score(X_test, y_test)


# In[ ]:


#Confusion Matrix
cm = confusion_matrix(y_test, nb.predict(X_test))
sns.heatmap(cm, annot=True)
plt.plot()


# **Random Forest Classifier**

# Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
for i in range(1, 20):
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train, y_train)
    print('estimators : ', i, "score : ", rfc.score(X_test, y_test), end="\n")


# In[ ]:


for i in range(1, 10):
    rfc = RandomForestClassifier(n_estimators=100, max_depth=i)
    rfc.fit(X_train, y_train)
    print('max_depth : ', i, "score : ", rfc.score(X_test, y_test), end="\n")


# In[ ]:


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)


# In[ ]:


#Confusion Matrix
cm = confusion_matrix(y_test, rfc.predict(X_test))
sns.heatmap(cm, annot=True)
plt.plot()


# **Support Vector Machine**

# In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). A SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

# In[ ]:


from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
svc.score(X_test, y_test)


# In[ ]:


#Confusion Matrix
cm = confusion_matrix(y_test, svc.predict(X_test))
sns.heatmap(cm, annot=True)
plt.plot()


# **All Score**

# * Logistic Regression - 0.9016393442622951
# * K Nearest Neighbor - 0.8688524590163934
# * Decision Tree Classifier - 0.8032786885245902
# * Gradient Boosting Classifies - 0.8852459016393442
# * Gaussian NB - 0.9344262295081968
# * Random Forest Classifier - 0.9180327868852459
# * Support Vector Machine - 0.9016393442622951
# 
# The best option shows the Gaussian NB model
