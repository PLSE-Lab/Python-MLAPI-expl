#!/usr/bin/env python
# coding: utf-8

# ## Telecom Dataset for Customer Churn

# ### 1. Dataset Description  <a id='titanic'>
# 
# The dataset provided is from a telecom company which has a record of customer information and their churn.  <br>
# Our goal is to predict the customers who might stop using their services.  <br>
# 
# **Column Description :** <br>
# - Customer ID : Unique ID of customer
# - gender : Two categories Male and Female
# - Senior Citizen : Two categories 0 or 1
# - Partner : Yes or No
# - Dependents : Yes or No
# - tenure : How long have they been with the company
# - Phone Service : Yes or No
# - Multiple Lines : Yes, No or No phone service
# - Internet Service : DSL, Fiber Optics or No
# - Online Security : Yes, No or No internet service 
# - Online Backup : Yes, No or No internet service
# - Device Protection : Yes, No or No internet service
# - Tech Support : Yes, No or No internet service
# - Streaming TV : Yes, No or No internet service
# - Streaming Movies : Yes, No or No internet service
# - Contract : Month-to-month, One year, Two year
# - Paperless Biling : Yes or No
# - Payment Method : Electronic check, Mailed check, Bank Transfer(automatic), Credit Card(automatic)
# - Monthly Charges : Numeric value 
# - Total Charges : Numeric value
# - Churn : Yes or No
# 
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### 2. Importing the packages and dataset  <a id='packages'>

# In[2]:


telco_df = pd.read_excel("../input/Telco-Customer-Churn.xlsx")


# In[3]:


# Importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import seaborn as sns


# ### 3. Exploring the dataset  <a id='explore'>

# In[ ]:


telco_df.head()


# In[ ]:


telco_df.info()


# There are 7043 rows and 21 columns. <br>
# There are only 3 numeric features and the rest are all categorical. 
# 

# In[6]:


telco_df.isna().sum()


# There are no missing values.

# **Descriptive Statistics**

# In[7]:


telco_df.describe()


# There is a lot of difference in standard deviations. We would need to scale the features. But Tree models do not need feature scaling

# In[8]:


corr = telco_df.corr()


# In[9]:


corr


# In[10]:


sns.heatmap(corr, annot=True)


# There is absolutely no correlation between any 2 variables.

# In[11]:


telco_df.columns


# **Exploratory Data Analysis**

# In[12]:


sns.countplot(x='gender', data=telco_df)


# In[13]:


sns.factorplot(x='gender', col='Churn', kind='count', data=telco_df);


# No relation between gender and churn. From the graph it is visible that both male and female are equally distributed. Therefore no impact of gender on churn. 

# In[14]:


sns.factorplot(x='DeviceProtection', col='Churn', kind='count', data=telco_df);


# When it comes to device protection, it doesn't give any insight about customer who are staying back. As for cutomer who are staying back, the device protection is not the only reasin for stay. <br>
# But it do provide information (right side graph) about the customer who are leaving the telecom provider, so there device protections plays a role. Therefore device protection is important but not that important.<br>
# 
# From the no internet service, we can further look into the streaming movies. As we can say that having no internet service means no streaming movies.

# In[15]:


sns.factorplot(x='StreamingMovies', col='Churn', kind='count', data=telco_df);


# From, the graph we can say that streaming movies and tv is not the deciding factor for the customer to decide whether to churn or not. 

# In[16]:


sns.factorplot(x='StreamingTV', col='Churn', kind='count', data=telco_df);


# In[17]:


sns.factorplot(x='InternetService', col='Churn', kind='count', data=telco_df);


# From the left side graph, customer who are staying for them fiber optics is not so important. But for those who are leaving (right side graph), customers especially companies, startups who needs higher bandwith. For them fiber optics internet service is important. Hence, more Fiber optic InternetService users are leaving. 

# In[18]:


sns.factorplot(x='SeniorCitizen', col='Churn', kind='count', data=telco_df);


# In[19]:


sns.distplot(telco_df['tenure'], color = 'green')
plt.title('Customer tenure with Telecom Provider')


# Most of the datapoint are on the leftside. As there is not much differenc between mean and median. So the graph is pretty much flat at the centre. But due to high variance at min and max and high standard deviation. We noticed a right and left tail

# In[20]:



sns.distplot(telco_df['MonthlyCharges'], color = 'green')
plt.title('Monthly Charges')


# In[21]:


sns.factorplot(x='PhoneService', col='Churn', kind='count', data=telco_df);


# In[22]:


sns.factorplot(x='MultipleLines', col='Churn', kind='count', data=telco_df);


# In[23]:


sns.factorplot(x='OnlineBackup', col='Churn', kind='count', data=telco_df);


# In[24]:


sns.factorplot(x='TechSupport', col='Churn', kind='count', data=telco_df);


# In[25]:


sns.factorplot(x='Contract', col='Churn', kind='count', data=telco_df);


# In[26]:


sns.factorplot(x='PaperlessBilling', col='Churn', kind='count', data=telco_df);


# In[27]:


plt.rcParams['figure.figsize'] = (18, 5)
sns.factorplot(x='PaymentMethod', col='Churn', kind='count', data=telco_df);
plt.xticks(rotation =70)

plt.show


# ### 4. Feature Engineering  <a id='fe'>

# In[28]:


sns.countplot(x="Churn", data=telco_df)
telco_df.Churn.value_counts()


# As we can see that the dataset is imbalanced. The number of customer churn is aproximately 1/4 of the dataset to the customer didnot churn is approximately 3/4 of the dataset. 

# Lets separate the categorical and numeric columns.

# In[29]:


cat_df = telco_df[['gender', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'Churn']]
cat_df.shape


# Dummification of categorical columns.

# In[30]:


cat_cols = pd.get_dummies(cat_df, drop_first=True)
cat_cols.head()


# In[31]:


cat_cols.shape


# In[32]:


num_df = telco_df.drop(['gender', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'Churn'], axis=1)
num_df.shape


# In[33]:


num_df.info()


# Merging dummified and categorical columns.

# In[34]:


dataset = pd.concat([num_df,cat_cols], axis=1 )


# In[35]:


dataset.shape


# In[36]:


dataset.info()


# In[37]:


dataset['TotalCharges']==' '


# **Taking care of object dtype**

# In[38]:


dataset['TotalCharges'][dataset['TotalCharges']==' ']


# In[39]:


dataset = dataset.drop(labels = list(dataset.TotalCharges[dataset.TotalCharges == " "].index))


# In[40]:


dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'])


# ## 5. Decision Tree  <a id='dt'>

# In[41]:


y = dataset["Churn_Yes"].values

X = dataset.drop(['Churn_Yes','customerID'], axis=1)


# In[42]:


# Stratified sampling

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101,stratify=y)


# In[43]:


# Importing the packages for Decision Tree Classifier

from sklearn import tree
my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=101, min_samples_leaf=3, class_weight="balanced")  #, class_weight="balanced"
my_tree_one


# ### 5.1 Training Decision Tree Model   <a id=tdt>

# In[44]:


# Fit the decision tree model on your features and label

my_tree_one = my_tree_one.fit(X_train, y_train)


# In[45]:


# The feature_importances_ attribute make it simple to interpret the significance of the predictors you include

list(zip(X_train.columns,my_tree_one.feature_importances_))


# We can see that 'Contract_Two year' is a important parameter in the model while predicting whether customer stay or churn away from the existing telecom provider.

# In[46]:



# The accuracy of the model on Train data

print(my_tree_one.score(X_train, y_train))


# The accuracy of the model on Test data

print(my_tree_one.score(X_test, y_test))


# We can see that the model is not over fitted because the accuracy score of train and test data are close to each other. Where as, the model is performing better with test data because it's accuracy score is better than the train data.  

# In[47]:


# Visualize the decision tree graph

with open('tree.dot','w') as dotfile:
    tree.export_graphviz(my_tree_one, out_file=dotfile, feature_names=X_train.columns, filled=True)
    dotfile.close()
    
# You may have to install graphviz package using 
# conda install graphviz
# conda install python-graphviz

from graphviz import Source

with open('tree.dot','r') as f:
    text=f.read()
    plot=Source(text)
plot   


# ### 5.2. Predictions of Decision Tree model   <a id= pdt>

# In[48]:


y_pred = my_tree_one.predict(X_test)


# ### 5.3. Evaluation of Decision Tree  <a id=edt>

# In[49]:


#Print Confusion matrix on Train Data
from sklearn.metrics import confusion_matrix, classification_report

pred = my_tree_one.predict(X_test)
df_confusion = confusion_matrix(y_test, pred)
df_confusion


# In[50]:


cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(df_confusion,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,
            fmt='d')


# In[51]:


# Remove few features and train


# ### 5.4 Parameter Tuning   <a id='ptdt'>

# #### What happen when we change the tree depth?

# In[52]:


# Setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two

my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 101, class_weight='balanced')
my_tree_two = my_tree_two.fit(X_train, y_train)

#Print the score of both the decision tree

print("New Decision Tree Accuracy: ",my_tree_two.score(X_train, y_train))
print("Original Decision Tree Accuracy",my_tree_one.score(X_train,y_train))


# We have improved our model by fine tuning the parameters. This is called hyperparameters tuning.

# In[53]:


# Making predictions on our Test Data 

pred = my_tree_two.predict(X_test)


# In[54]:


print("New Decision Tree Accuracy on test data: ",my_tree_two.score(X_test, y_test))


# In[55]:


# The accuracy of the model on Train data

print(my_tree_two.score(X_train, y_train))


# The accuracy of the model on Test data

print(my_tree_two.score(X_test, y_test))


# As we can see that the variation between the train and test data is significant. So, we can infer that our model is suffering from the overfitting. Though the accuracy score is better than the previous model ( my tree one; .665). It mainly because of the hyperparmater i.e. the increase in the max_depth, min_sample split due to which it become too much attached to training data and increases the level of complexity. Instead of being generic model it become more centric to specific conditions as we increases the number of levels. <br>
# 
# For example it is similar to survey where keeping the question minimum and generic can be utilise in other area of studies. Whereas, increasing the number of question will increase the level of complexity and direct the questions to specifice requirement. Therefore it will no longer remain a generic and become more specific to particular conditions.

# In[56]:


# Building confusion matrix of our improved model

df_confusion_new = confusion_matrix(y_test, pred)
df_confusion_new


# In[57]:


cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(df_confusion_new, cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,
            fmt='d')


# If we look back at our previous model (my tree one, TP =542) it is more accurate than the above model (my tree two, TP =457) to predict the customer churning. Therefore the recall and F1 score dropped in this model. 

# ## 6. Random Forest   <a id=rf>

# In[58]:


# Building and fitting Random Forest

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion = 'gini',  n_estimators = 100, max_depth = 10,random_state = 101, class_weight="balanced")


# ### 6.1. Training Random Forest Model  <a id=trf>

# In[59]:


# Fitting the model on Train Data

my_forest = forest.fit(X_train, y_train)


# In[60]:


# Print the accuracy score of the fitted random forest

print(my_forest.score(X_train, y_train))

print(my_forest.score(X_test, y_test))


# As we can see the model is suffering from overfitting. The main reason for this is having a 10 level of max depth, which seems to be huge and result in having a complex decision tree. We can reduce the overfitting by reducing the maxdepth.

# ### 6.2. Prediction from Random Forest Model   <a id=prf>

# In[61]:


# Making predictions

pred = my_forest.predict(X_test)


# In[62]:


list(zip(X_train.columns,my_forest.feature_importances_))


# ### 6.3 Evaluation of Random Forest Model  <a id=erf>

# In[63]:


df_confusion_rf = confusion_matrix(y_test, pred)
df_confusion_rf


# In[64]:


cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(df_confusion_rf, cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,
            fmt='d')


# ### 7. Appendix  <a id = 'appendix'>
# 
# **Grid Search**
# 
# Grid-searching is the process of scanning the data to configure optimal parameters for a given model.  <br>
# Let's apply Grid Search on a Decision Tree Model. It has many parameters like tree depth, criterion... etc. We will build different models with different combinations of these parameters and come up with the best model. 
# 

# In[65]:


# Different parameters we want to test

max_depth = [5,10,15] 
criterion = ['gini', 'entropy']
min_samples_split = [5,10,15]


# In[66]:


# Importing GridSearch

from sklearn.model_selection import GridSearchCV


# In[67]:


# Building the model

my_tree_three = tree.DecisionTreeClassifier(class_weight="balanced")

# Cross-validation tells how well a model performs on a dataset using multiple samples of train data
grid = GridSearchCV(estimator = my_tree_three, cv=3, 
                    param_grid = dict(max_depth = max_depth, criterion = criterion, min_samples_split=min_samples_split), verbose=2)


# In[68]:


grid.fit(X_train,y_train)


# In[69]:


# Best accuracy score

print('Avg accuracy score across 54 models:', grid.best_score_)


# In[70]:


# Best parameters for the model

grid.best_params_


# In[71]:


# Building the model based on new parameters

my_tree_three = tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 10, random_state=42, class_weight="balanced")


# In[72]:


my_tree_three.fit(X_train,y_train)


# In[73]:


# Accuracy Score for new model

my_tree_three.score(X_train,y_train)


# **Observation:** Our accuracy score improve from 0.73 to 0.82

# **Randomized Search**
# 
# Using Randomized Search, we can define a grid of hyperparameters and randomly sample from the grid to get the best combination of values. <br>
# Lets apply Randomized search on Random Forest model. This model has plenty of parameters like number of trees, depth of trees...etc. We will evaluate models with different parameters and come up with the best model.

# In[74]:


# Different parameters we want to test

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[75]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[76]:


# Importing RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV


# In[77]:


forest_two = RandomForestClassifier(class_weight="balanced")

# Fitting 3 folds for each of 100 candidates, totalling 300 fits
rf_random = RandomizedSearchCV(estimator = forest_two, param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose=2, random_state=42)


# In[78]:


rf_random.fit(X_train,y_train)


# In[81]:


rf_random.best_params_


# A Random Forest model built with this hyperparameters will provide best accuracy

# In[83]:


rf_random.best_score_


# **Observation:** The original Random forest algorithm gave 0.87 accuracy whereas after Randomized Grid Search we have only got 0.79. This is the best random combination of parameters the algorithm has choosen which produced the highest accuracy.

# ### The End
