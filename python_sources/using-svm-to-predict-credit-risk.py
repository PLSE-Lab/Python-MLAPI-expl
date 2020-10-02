#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# *I am doing this course: [Machine Learning Practical](https://eylearning.udemy.com/machine-learning-practical) on Udemy, and I think the best way is to practice on another dataset with similar problem (classification).*
# 
# The dataset contains information about loan applicants with a German bank, where many characteristics are collected and based on that the risk (good or bad) will be categorized. 
# 
# Detailed description and source of the data can be found [here](https://www.kaggle.com/uciml/german-credit). Helpful comments are welcome.

# **Step 0: Import relevant Packages**

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as  plt # background package for seaborn
import seaborn as sns # visualisation package based on plt
import sklearn # machine learning package
import os
print(os.listdir("../input"))


# **Step 1: Import and Clean the Data**

# * Using methods df.head(), df.info or df.describe() (*describe only works with numeric variables*) or df.nunique() allows us to take a glimpse at the data such as type of varibales, how many have missing data.
# * Our dataset contains: 
#     * dependent variable (outcome): Risk (good/bad)
#     * independent variables (features): Age, Sex, Job (0->3), Housing (own/rent/free), Saving accounts, Checking account, Credit amount, Duration (month) and Purpose
# * As "Saving accounts" and "Checking account" have missing values and they are not quantitative/numeric but categorical, hence we will fill these missing values with another category as "Not Available"

# In[2]:


credit_df = pd.read_csv("../input/german_credit_data.csv",index_col=0)
credit_df.head(10)


# In[3]:


credit_df.info()


# In[4]:


credit_df = credit_df.fillna(value="not available")
credit_df.info()


# In[5]:


credit_df.describe()


# In[6]:


credit_df.nunique()


# I realise as I move along the analyse that categorical variables cause inconsistency to the plot and the model. And since our categorical variables donot vary much, for example, saving accounts have 5 categories. I will transform categorical variables (including Sex, Housing, Saving and Checking accounts, Purpose) into numerics.
# * Sex: male = 1, female = 2
# * Housing: own = 1, rent = 2, free = 3
# * Saving account/Checking account: Not available = 0, litte = 1, moderate = 2, quite rich = 3, rich = 4
# * Purpose: car = 1, furniture/equipment = 2, radio/TV = 3, domestic appliances = 4, repairs = 5, education = 6, business = 7, vacation/others = 8
# 
# I saw that sklearn.preprocessing has this class 'LabelEncoder' which will fit_transform the categorical data into numeric values, however, I find it hard to have a control understanding on to which value they will convert what. So I will do it manually with pandas. I am also aware of pandas having this method 'get_dummies' to get one-hot encodings but I am not using it either.

# In[7]:


credit_df.Sex = credit_df.Sex.map({ 'male' : 1, 'female' : 2})


# In[8]:


credit_df.Housing = credit_df.Housing.map({ 'own' : 1, 'rent' : 2, 'free' : 3})


# In[9]:


credit_df['Saving accounts'] = credit_df['Saving accounts'].map({ 'not available' : 0, 'little' : 1, 'moderate' : 2, 'quite rich': 3, 'rich': 4})


# In[10]:


credit_df['Checking account'] = credit_df['Checking account'].map({ 'not available' : 0, 'little' : 1, 'moderate' : 2, 'quite rich': 3, 'rich': 4})


# In[11]:


credit_df['Purpose'] = credit_df['Purpose'].map({ 'car':1, 'furniture/equipment':2, 'radio/TV':3, 'domestic appliances':4, 'repairs':5, 'education':6, 'business':7, 'vacation/others':8})


# In[12]:


credit_df.head(10)


# **Step 2: Visualisation**

# First, let us see the distribution of the target variable 'Risk'. We have pretty skewed data, which has more "good" than "bad". This lead to the fact that the probablity that our model predict "good" better than "bad", or outcome will be correctly guessed as "good".

# In[13]:


sns.countplot(credit_df['Risk'], label = "Count") 


# In[14]:


ax = sns.pairplot(credit_df, hue = 'Risk')


# **Step 3: Model Training**

# In[15]:


# Create set of only independant variables by dropping Risk
X = credit_df.drop(['Risk'], axis=1)
X.head()


# In[16]:


# Create a series of outcome variable only
y = credit_df['Risk']
y.head()


# In[17]:


# split datasets into training and test subsets for both X and y using sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)


# In[18]:


print(X_train.shape)
print(X_test.shape)


# In[19]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
# train the model
svc_model = SVC()
svc_model.fit(X_train, y_train)


# **Step 4: Evaluate the Model**

# AS we can see, our model can only predict well with good clients since there are more good clients in our data than bad clients. Even though the accuracy is at 72%, as seen in the confusion matrix, it didnt score anything for the "bad" class, just exactly as we expected. 
# 
# Accuracy is to measure how well a binary classification test correctly identify the results in both classes. Accuracy is not a reliable metric for the real performance of a classifier, because it will yield misleading results if the data set is unbalanced (that is, when the numbers of observations in different classes vary greatly, which is in our case). Therefore, in our case it is better if we can enhance our model in the sense that more "bad" cases are predicted correctly even if we have to trade off the lower general acccuracy. These are reflected in confusion matrix as well as in other metrics for "bad" class.
# 

# In[20]:


y_pred = svc_model.fit(X_train, y_train).predict(X_test)


# In[21]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)


# In[22]:


print(classification_report(y_test, y_pred))


# In[39]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# **Step 5: Improving the Model**

# **# We try to improve the model in 2 steps:**
# * First step, we will normalise our data to see if a normalised data yields better result
# * Secondly, we will try to optimise the parameters in SVM function (C, gamma)

# 1. Normalise data

# By rescaling data into the range of 0-1, we expect to remove the unit of measurements to make our data more consistent. 
# 
# The formular is:  x_new = (x - x_min) / (x_max - x_min)

# In[23]:


min_train = X_train.min()
min_train


# In[24]:


range_train = (X_train - min_train).max()
range_train


# In[40]:


X_train_scaled = (X_train - min_train)/range_train
X_train_scaled.head()


# In[26]:


sns.scatterplot(x = X_train['Credit amount'], y = X_train['Duration'], hue = y_train)


# In[27]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[28]:


svc_model1 = SVC()
svc_model1.fit(X_train_scaled, y_train)


# In[29]:


y_predict = svc_model1.predict(X_test_scaled)
cm1 = confusion_matrix(y_test, y_predict)

sns.heatmap(cm1,annot=True,fmt="d")


# In[41]:


accuracy_score(y_test, y_predict)


# 2. Optimisation of SVM Parameters

# parameters gamma and C of the Radial Basis Function (RBF) kernel SVM: 
# 
# * C: trade off between misclassification and smoothness of decision boundary. Larger C (high margin) is to penalise when misclassification, smaller C (soft margin) is to be gentle to misclassfication in order to obtain a smoother boundary surface.
# * Gamma: can be said to adjust the curvature of the decision boundary. Lower gamma makes the decision region broader due to low curve of the decision boundary.

# In[30]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[43]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[33]:


grid.fit(X_train_scaled,y_train)


# In[34]:


grid.best_params_


# In[35]:


grid.best_estimator_


# In[36]:


grid_predictions = grid.predict(X_test_scaled)


# In[37]:


cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)


# In[38]:


print(classification_report(y_test,grid_predictions))


# In[42]:


accuracy_score(y_test, grid_predictions)


# As we have selected the best parameters for SVM model, we are able to perform better in correcly classifying more "bad" creditors, this is for the bank better than correctly classifying "good" creditors, since "good creditors" will pay the debt anyway, and identifying "bad creditors" will help create better risk adjustment methods on such clients in terms of extra deposit or another guarantee. 
# 
# Food for thought: applying one model is not enough, we have to compare different models to see which one scores better, find enhancements on those models and compare them again. Furthermore, maybe it is not necessary to include all features in the model, but to select those are relevant?
