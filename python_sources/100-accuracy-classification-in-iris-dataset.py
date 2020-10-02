#!/usr/bin/env python
# coding: utf-8

# # IRIS DATASET 
# Exploring Iris dataset through different methods. 
# 
# 1. Plotting various features against their categories
# 3. Linear Regression 
# 4. Correlation 
# 4. Decision Tree Classification 
# 5. Accuracy Measurement

# Importing necessary libraries 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# Creating a Pandas DataFrame from a CSV file<br></p>
# 
# 

# In[ ]:


data = pd.read_csv("../input/Iris.csv")
data.shape


# In[ ]:


data.head(10)


# In[ ]:


data.tail(10)


# ## Data Cleaning 
# Let's look for any null values to find out whether there needs to be null values. 

# In[ ]:


data.isnull().any().any()


# In[ ]:


# Dropping the ID column
data.drop('Id',inplace=True,axis=1)


# In[ ]:


data.columns
cols = list(data.columns)
cols


# As we can see, there are no null values in this data. Hence, we do not need to perform data cleaning. 

# ## Scaling the values ( Data Normalisation) 

# We will use the min - max method to scale the values down 

# In[ ]:


range = data["SepalLengthCm"].max() - data["SepalLengthCm"].min()
range


# In[ ]:


data["SepalLengthCm"] = (data["SepalLengthCm"] - data["SepalLengthCm"].min())/range
data.head()


# In[ ]:


data["SepalLengthCm"] = data["SepalLengthCm"] / data["SepalLengthCm"].max()


# In[ ]:


data.head()


# ## Reverting 
# I reverted back to the same values to avoid major scale down and maintain some variance to allow for distinguishable features. 

# In[ ]:


data = pd.read_csv("../input/Iris.csv")
data.shape


# In[ ]:


data.head()


# ## Plotting 
# 
# Let's plot scatter plots to see the differences in the different features across different Species 

# In[ ]:


group_names = data['Species'].unique().tolist()
group_names


# Importing necessary libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt


# ### Scatterplots

# #### SepalLength vs SepalWidth 

# In[ ]:


sns.scatterplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = data, hue = 'Species')
plt.title('Sepal Length vs Sepal Width')
plt.show()


# In[ ]:


data['SepalLengthCm'].corr(data['SepalWidthCm'])


# > **Analysis**: We can see that there is a lot of correlation in Iris-Setosa category when it comes to Sepal Length and SepalWidth but not a similar distinction for the other two categories. The correlation does not seem very strong either. 

# #### PetalLength vs PetalWidth

# In[ ]:


sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data = data ,hue ='Species')
plt.title('Petal Length vs Petal Width')
plt.show()


# In[ ]:


data['PetalLengthCm'].corr(data['PetalWidthCm'])


# > **Analysis**: This is a very insightful graph. It tells us that we can use PetalLength to predict PetalWidth and it's category because of the proper clustrering of data points. The correlation is also very high. We can use these features for regression analysis later. **

# #### PetalLength vs SepalLength

# In[ ]:


sns.scatterplot(x = 'PetalLengthCm', y = 'SepalLengthCm', data = data ,hue ='Species')
plt.title('Petal Length vs Sepal Length')
plt.show()


# In[ ]:


data['PetalLengthCm'].corr(data['SepalLengthCm'])


# > **Analysis**: There is high correlation but we still do not get a very linear graph which is important. There is good  clustering but let's see if we can get something better 

# #### PetalWidth vs SepalWidth

# In[ ]:


sns.scatterplot(x = 'PetalWidthCm', y = 'SepalWidthCm', data = data ,hue ='Species')
plt.title('Petal Length vs Sepal Length')
plt.show()


# In[ ]:


data['PetalWidthCm'].corr(data['SepalWidthCm'])


# > **Analysis**: There isn't a lot of correlation. There is good clustering but let's see if we can get something better 

# ### Boxplot

# In[ ]:


sns.boxplot(x = "Species", y = "PetalLengthCm", data = data)


# 

# ## Correlation Heat Map
# Let's make a correlation heatmap to understand the correlation between different species.

# In[ ]:


no_id_data = data.copy()
no_id_data.drop("Id", axis = 1, inplace = True)
sns.heatmap(data = no_id_data.corr(), annot = True)
plt.show()


# We can see that there is a very high correlation between Petal Length and Petal Width. 

# ## Linear Regression ( Extra Work : Skip ahead to the 100% accuracy classification ) 
# Let's try and predict the Petal Width from the Petal Length

# In[ ]:


x_values = data['PetalLengthCm'].copy()
y_values = data['PetalWidthCm'].copy()


# In[ ]:


x_train, x_test, y_train1, y_test1 = train_test_split(x_values, y_values, test_size = 0.33, random_state = 3)


# Regression Libraries

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold">
# Convert to a Classification Task <br></p>
# 

# ### Adding Dummy Variables 
# 

# In[ ]:


species_dummy = pd.get_dummies(data["Species"])
species_dummy.head()


# In[ ]:


assigned_data = data.copy()


# In[ ]:


assigned_data = pd.concat([data, species_dummy], axis = 1)
assigned_data.head()


# Classification 

# In[ ]:


assigned_data.drop(["Id"], inplace = True, axis = 1)
assigned_data.head()


# In[ ]:


target = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
features = cols[0:4]
print(target)
print(features)


# In[ ]:


y = assigned_data[target].copy()
X = assigned_data[features].copy()


# Let's divide the data into training and test sets

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)


# In[ ]:


print(X_train.describe())
X_train.head()


# In[ ]:


y_train.head(10)


# We can see now that we have quite a randomized group of values for y_train and X_train. Let's build our classifier model. 

# In[ ]:


iris_classifier = DecisionTreeClassifier(max_leaf_nodes = 4, random_state = 0)
iris_classifier.fit(X_train, y_train)


# <p style="font-family: Arial; font-size:1.75em; color:blue; font-style:bold"><br>
# 
# Predict on Test Set 
# 
# <br><br></p>
# 

# In[ ]:


y_prediction = iris_classifier.predict(X_test)


# In[ ]:


y_prediction[0 : 10]


# In[ ]:


y_test[0:10]


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# Measure Accuracy of the Classifier
# <br><br></p>
# 

# In[ ]:


accuracy_score(y_true = y_test, y_pred = y_prediction)


# ## Conclusion 
# <br> 
# I want to know  the reason for this 100% accuracy. Is it because these are highly correlated variables? 
# 
# Things to note:- 
# 1. If I decrease the size of the training set, I can reduce the accuracy which is obvious but still mentioning. **Is it a bad practice to take 0.1 as the test size? **
# 2. Even if I do not scale the values, I can get a 100% result. The scaling was for my own knowledge. 

# In[ ]:




