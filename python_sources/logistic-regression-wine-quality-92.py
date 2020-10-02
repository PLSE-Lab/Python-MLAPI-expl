#!/usr/bin/env python
# coding: utf-8

# ### INTRODUCTION
# 
# #### Predicting the quality of wine mainly through logistic regression
# #### Wine quality was classified into two categories  good(0) and bad(1)
# #### Steps taken in preprocessing includes Data cleaning, Outliers Removal, Standardization etc
# 
# ### SIDE NOTE
# #### You can leave your question about any unclear part in the comment section
# #### Any correction will be highly welcomed

# ### LOADING THE DATASET

# In[ ]:


# Importing the neccesary libraries we are going to need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sns.set()


# In[ ]:


path = '/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv'

df = pd.read_csv(path)


# In[ ]:


df.head(5)


# ### DEALING WITH MISSING VALUES 

# In[ ]:


df.info()
#From the result we see that the dataset is clean i.e no misssing values


# ### DATA CLEANING

# #### The first thing we will want to to do is categories our target variable that is 'quality' into 'good' or 'bad'. In this case if the quality is greater than 6.5  the quality is good value less or equal to 6.5 is represented as bad
# #### NOTE that 'good' is represented by 1 while 'bad' by 0

# In[ ]:


grade = [] #Declaring a new list
for i in df['quality']: 
    if i > 6.5:
        i = 1
        grade.append(i)
    else:
        i = 0
        grade.append(i)
df['grade'] = grade # A new column to hold our already categoried quality 


# In[ ]:


df.head(10)


# In[ ]:


df.drop('quality', axis = 1, inplace = True) #Dropping the quality column since we won't be needing it anymore


# ### OUTLIERS

# ####  plotting the distribution of our numerical variables will help us to detect outliers and any other abnormalities

# In[ ]:


df.describe() #shows describption for only numerical variables


# In[ ]:


sns.distplot(df['fixed acidity']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['fixed acidity'].quantile(0.99)
df = df[df['fixed acidity'] < q]

sns.distplot(df['fixed acidity'])


# In[ ]:


sns.distplot(df['volatile acidity']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['volatile acidity'].quantile(0.99)
df = df[df['volatile acidity'] < q]

sns.distplot(df['volatile acidity'])


# In[ ]:


sns.distplot(df['citric acid']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['citric acid'].quantile(0.99)
df = df[df['citric acid'] < q]

sns.distplot(df['citric acid'])


# In[ ]:


sns.distplot(df['residual sugar']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['residual sugar'].quantile(0.99)
df = df[df['residual sugar'] < q]

sns.distplot(df['residual sugar'])


# In[ ]:


sns.distplot(df['chlorides']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['chlorides'].quantile(0.99)
df = df[df['chlorides'] < q]

sns.distplot(df['chlorides'])


# In[ ]:


sns.distplot(df['free sulfur dioxide']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['free sulfur dioxide'].quantile(0.99)
df = df[df['free sulfur dioxide'] < q]

sns.distplot(df['free sulfur dioxide'])


# In[ ]:


sns.distplot(df['total sulfur dioxide']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['total sulfur dioxide'].quantile(0.99)
df = df[df['total sulfur dioxide'] < q]

sns.distplot(df['total sulfur dioxide'])


# In[ ]:


sns.distplot(df['density']) #we can see those few outliers shown by the longer left tail of the distribution


# In[ ]:


#Removing the bottom 1% of the observation will help us to deal with the outliers
q = df['density'].quantile(0.01)
df = df[df['density'] > q]

sns.distplot(df['density'])


# In[ ]:


sns.distplot(df['pH']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['pH'].quantile(0.99)
df = df[df['pH'] < q]

sns.distplot(df['pH'])


# In[ ]:


sns.distplot(df['sulphates']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['sulphates'].quantile(0.99)
df = df[df['sulphates'] < q]

sns.distplot(df['sulphates'])


# In[ ]:


sns.distplot(df['alcohol']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['alcohol'].quantile(0.99)
df = df[df['alcohol'] < q]

sns.distplot(df['alcohol'])


# ### CHECKING OLS ASSUMPTIONS

# #### Let's check that our dataset are not violating any of this assumptions which includes:
# #### 1. No Endogeneity
# #### 2. Normality and Homoscedasticity
# #### 3.No Autocorrelation
# #### 4.NO multicollinearity: making sure our independents variables are not strongly related(correlated) with each other
# 
# ####  We are not violating  assumptions 1 through 3 but for NO multicollinearity we need to check

# In[ ]:


df.columns.values


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# the target column (in this case 'grade') should not be included in variables
#Categorical variables may or maynot be added if any
variables = df[['fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',]]
x = add_constant(variables)
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(x.values,i) for i in range (x.shape[1])]
vif['features'] = x.columns
vif

#Using 10 as the minimum vif values i.e any independent variable 10 and above will have to be dropped
#From the results all independent variable are below 10


# ### Standardization

# #### Standardizing helps to give our independent varibles a more standard and relatable numeric scale, it also helps in improving model accuracy

# In[ ]:


#Declaring independent variable i.e x
#Declaring Target variable i.e y
x = df.drop('grade', axis =1 )
y = df['grade']


# In[ ]:


scaler = StandardScaler()
scaler.fit(x)
scaled_x = scaler.transform(x)


# ### LOGISTIC REGRESSION

# In[ ]:


#Splitting our data into train and test dataset
x_train, x_test, y_train, y_test = train_test_split(scaled_x, y , test_size = 0.2, random_state  = 365)


# In[ ]:


reg = LogisticRegression() #select the algorithm
reg.fit(x_train,y_train) # we fit the algorithm with the training data and the training output


# In[ ]:


y_hat = reg.predict(x_test) # y_hat holding the prediction made with the algorithm using x_test


# In[ ]:


acc = metrics.accuracy_score(y_hat,y_test)# To know the accuracy
acc


# In[ ]:


reg.intercept_ # Intercept of the regression


# In[ ]:


reg.coef_ # coefficients of the variables / features 


# In[ ]:


result = pd.DataFrame(data = x.columns, columns = ['Features'])
result['weight'] = np.transpose(reg.coef_)
result['odds'] = np.exp(np.transpose(reg.coef_))
result


# #### Remember we standardized all independents variables so the odds values have no direct interpretation
# #### Nevertheless using acohol as an example we can say for one standard deviation increase in acohol it is twice more likely to cause a change in our target variables

# ### CONFUSION MATRIX

# In[ ]:


cm = confusion_matrix(y_hat,y_test)
cm


# In[ ]:


# Format for easier understanding
cm_df = pd.DataFrame(cm)
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df


# #### Our model predicted '0' correctly 250 times while predicting '0' incorrectly 18 times
# #### Also it predicted  '1'  correctly 10 times while predicting '1' incorrectly 4 times

# ###  USING OTHER MODELS

# In[ ]:


from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours


# In[ ]:


dd = DecisionTreeClassifier()
dd.fit(x_train,y_train)
y_1 = dd.predict(x_test)
acc_1 = metrics.accuracy_score(y_1,y_test)
acc_1


# In[ ]:


sv = svm.SVC() #select the algorithm
sv.fit(x_train,y_train) # we train the algorithm with the training data and the training output
y_2 = sv.predict(x_test) #now we pass the testing data to the trained algorithm
acc_2 = metrics.accuracy_score(y_2,y_test)
acc_2


# In[ ]:


knc = KNeighborsClassifier()
knc.fit(x_train,y_train)
y_3 = knc.predict(x_test)
acc_3 = metrics.accuracy_score(y_3,y_test)
acc_3


# #### After comparison to some other models, LogisticRegression still gives us the highest (~92%)

# #### If you find this notebook useful don't forget to upvote. #Happycoding

# In[ ]:




