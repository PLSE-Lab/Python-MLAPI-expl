#!/usr/bin/env python
# coding: utf-8

#  1. In this problem we have to use 30 different columns and we have to predict the Stage of Breast Cancer M (Malignant)  and B (Bengin)
#  2. This analysis has been done using Basic Machine Learning Algorithm with detailed explanation
#  3. This is good for beginners like as me Lets start.
#  
# 4.Attribute Information:
# 
# 1) ID number
# 
# 2) Diagnosis (M = malignant, B = benign)
# 
# -3-32.Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter)
# 
# b) texture (standard deviation of gray-scale values)
# 
# c) perimeter
# 
# d) area
# 
# e) smoothness (local variation in radius lengths)
# 
# f) compactness (perimeter^2 / area - 1.0)
# 
# g). concavity (severity of concave portions of the contour)
# 
# h). concave points (number of concave portions of the contour)
# 
# i). symmetry
# 
# j). fractal dimension ("coastline approximation" - 1)
# 
# 5  here 3- 32 are divided into three parts first is Mean (3-13),  Stranded Error(13-23) and  Worst(23-32) and each contain 10 parameter (radius, texture,area, perimeter, smoothness,compactness,concavity,concave points,symmetry and fractal dimension) 
# 
#  6. Here Mean means the means of the all cells,  standard Error of all cell and worst means the worst  cell 

# In[ ]:


# here we will import the libraries used for machine learning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model
# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then insteda of model_selection we can use cross_validation


# **Import data **

# In[ ]:


data = pd.read_csv("../input/data.csv",header=0)# here header 0 means the 0 th row is our coloumn 
                                                # header in data


# In[ ]:


# have a look at the data
print(data.head(2))# as u can see our data have imported and having 33 columns
# head is used for to see top 5 by default i used 2 so it will print 2 rows
#rows If we will use
#print(data.tail(2))# it will print last 2 rows in data


# In[ ]:


# now lets look at the type of data we have. We can use 
data.info()


# *As I said I m beginner so here I am explaining every thing in detail 
#  So lets decribe what these data type means
#  e.g 5 radius_mean 569 non-null float64 here it means the radius_mean have 569 float type value
# now u can see Unnamed:32 have 0 non null object it means the all values are null in this column so we cannot use this column for our analysis*

# In[ ]:


# now we can drop this column Unnamed: 32
data.drop("Unnamed: 32",axis=1,inplace=True) # in this process this will change in iour data itself 
# if you want to save your old data then you can use below code
# data1=data.drop("Unnamed:32",axis=1)
# here axis 1 means we are droping the column


# In[ ]:


# here you can check the column has been droped
data.columns # this gives the column name which are persent in our data no Unnamed: 32 is now there


# In[ ]:


# like this we also donot want the Id columfor our analysis
data.drop("id",axis=1,inplace=True)


# In[ ]:


# As I said above the data can be divided into three parts.lets divied the features according to their category
features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:31])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)


# In[ ]:


# lets now start with features_mean 
# now as u know our diagnosis column is a object type so we can map it to integer value
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})


# ## Explore the Data now
# 

# In[ ]:


data.describe() # this will describe the all statistical function of our data


# In[ ]:


# lets get the frequency of cancer stages
sns.countplot(data['diagnosis'],label="Count")


# In[ ]:


# from this graph u can see that there is a more number of bengin stage of cancer which can be cure


# ## Data Analysis a little feature selection

# In[ ]:


# now lets draw a correlation graph so that we can remove multi colinearity it means the columns are
# dependenig on each other so we should avoid it because what is the use of using same column twice
# lets check the correlation between features
# now we will do this analysis only for features_mean then we will do for others and will see who is doing best
corr = data[features_worst].corr() # .corr is used for find corelation
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm') # for more on heatmap you can visit Link(http://seaborn.pydata.org/generated/seaborn.heatmap.html)


# *observation*
# * the radius, parameter and area  are highly correlated as expected from their relation*
# * so from these we will use anyone of them *
# *compactness_mean, concavity_mean and concavepoint_mean are highly correlated so we will use compactness_mean from here *
# * so selected Parameter for use is perimeter_mean, texture_mean, compactness_mean, symmetry_mean*

# In[ ]:





# In[ ]:


prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
# now these are the variables which will use for prediction


# In[ ]:


#now split our data into train and test
train, test = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and tset
# we can cjheck there shape
print(train.shape)
print(test.shape)



# In[ ]:


train_X = train[prediction_var]# taking the training data input only for prediction model
print(train_X.shape) # to confirm the shape
train_y=train.diagnosis# this is our output variable
print(train_y.shape)
# same we have to do for test
test_X= test[prediction_var] # this is data for test and these are the input values
test_y =test.diagnosis   #this is the test ouput value which we will predict usin our model and test input values


# In[ ]:


model=RandomForestClassifier(n_estimators=100)# a simple random forest model


# In[ ]:


model.fit(train_X,train_y)# now fit our model for traiing data


# In[ ]:


prediction=model.predict(test_X)# predict for the test data
# prediction will contain the predicted value by our model dor test input 


# In[ ]:


metrics.accuracy_score(prediction,test_y) # to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values


# * Here the Accuracy for our model is 91 % which seems good*

# In[ ]:


# lets now try with SVM


# In[ ]:


model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


# **SVM is giving only 0.85 which we can improve by using different techniques** 
# **i will improve it till then beginners can understand how to model a data and they can have a overview of ML**
