#!/usr/bin/env python
# coding: utf-8

# **What is Logistic Regression?**
# 
# * Logistic Regression is a type of superviersed Machine Learning algorithm used to solve binary classification problems ( *1 or 0, Yes or No, True or False*).  We use sigmoid function to classify the dependent variable based on the given independent variables.  Here the dependent varible has to be categorical. Alternatively, we can also say that logistic regression predicts the probability of an event. 
# 
# **Sigmoid Function:** $$\frac{1}{1 + e^{-y }}$$
# 
# y = w*x + b
# Here x is the independent variable that we want to transform and w & b are weight and bias respectively

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = np.arange(-100, 100, 1)
w = 0.1
b = 0.2

plt.title("Sigmoid Function");
plt.scatter(x,  1 / (1 + np.exp(-w*x -b))) 


# # Simple Logistic Regression Application

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression() 
from sklearn.metrics import accuracy_score


# In[ ]:


application = [[ 0.56,  1],
            [ 0.17,  0 ],
            [ 0.34,  0],
            [ 0.20,  0 ],
            [ 0.70,  1 ]]

x = np.array(application)[:, 0:1] 
y = np.array(application)[:, 1]
model.fit(x, y) 


# In[ ]:


model.predict(x)  # this model is not giving us the perfect classification, we can validate this by using other metrics


# In[ ]:


accuracy_score(y, model.predict(x)) # Accuracy of our model is only 60%. 
# We just wanted to show how to use Logistic Regression algorithm.


# # Logistic Regression on Abalone Dataset

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/abalone.csv") 


# In[ ]:


df.sample(5)


# In[ ]:


df.describe() 


# In[ ]:


df[df.Height == 0] # there are 2 columns with 0 height. We can remove this data as 0 height does not make sense.


# In[ ]:


df = df[df.Height != 0] # removing rows with 0 height.
df.describe() 


# In[ ]:


df.isna().sum() # Finding null values


# In[ ]:


df.info() # We, have one categorical values and we shall change that to countinuous variable.


# In[ ]:


sns.countplot(df.Sex) 


# In[ ]:


new_col = pd.get_dummies(df.Sex)
df[new_col.columns] = new_col


# In[ ]:


df.columns # new columns has been added M, F & I


# In[ ]:


sns.pairplot(df.drop(['F','I', 'M'], axis=1))


# Most of the analysis has already been done in https://www.kaggle.com/suprabhatsk/abalone-eda-simple-regression-analysis. 
# So, we will just continue from there to solve this using classification algorithms

# In[ ]:


#  Our job is to predict the age of the Ring on the given feature. So, let look at the Ring in detail.

plt.figure(figsize=(12, 10))

plt.subplot(2,2,1)
sns.countplot(df.Rings)

plt.subplot(2,2,2)
sns.distplot(df.Rings)

plt.subplot(2,2,3)
stats.probplot(df.Rings, plot=plt)

plt.subplot(2,2,4)
sns.boxplot(df.Rings) 

plt.tight_layout()

#It seems that the label value is skewed after 15 years of age. We will deal with that in a latter.df.describe()  


# In[ ]:


# As we can see that the data we have at disposal is great for predicting the Rings between 3 to 15 years.

new_df = df[df.Rings < 16]
new_df = new_df[new_df.Rings > 2]
new_df = new_df[new_df.Height < 0.4]


# In[ ]:


plt.figure(figsize=(12,10))

plt.subplot(3,2,1)
sns.boxplot(data= new_df, x = 'Rings', y = 'Diameter')

plt.subplot(3,2,2)
sns.boxplot(data= new_df, x = 'Rings', y = 'Length')

plt.subplot(3,2,3)
sns.boxplot(data= new_df, x = 'Rings', y = 'Height')

plt.subplot(3,2,4)
sns.boxplot(data= new_df, x = 'Rings', y = 'Shell weight')

plt.subplot(3,2,5)
sns.boxplot(data= new_df, x = 'Rings', y = 'Whole weight')

plt.subplot(3,2,6)
sns.boxplot(data= new_df, x = 'Rings', y = 'Viscera weight')
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(12, 10))

plt.subplot(2,2,1)
sns.countplot(new_df.Rings)

plt.subplot(2,2,2)
sns.distplot(new_df.Rings)

plt.subplot(2,2,3)
stats.probplot(new_df.Rings, plot=plt)

plt.subplot(2,2,4)
sns.boxplot(new_df.Rings)

plt.tight_layout()


# Data Preprocssing

# In[ ]:


from sklearn.preprocessing import StandardScaler
convert = StandardScaler()

feature = new_df.drop(['Sex', 'Rings'], axis = 1)
label = new_df.Rings

feature = convert.fit_transform(feature)


# In[ ]:


from sklearn.model_selection import train_test_split
f_train, f_test, l_train, l_test = train_test_split(feature, label, random_state = 23, test_size = 0.2)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=23)
model.fit(f_train, l_train)


# In[ ]:


y_predict = model.predict(f_train)


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


# In[ ]:


accuracy_score(l_train, y_predict) 


# In[ ]:


# So, this is accuracy score of Logistic Regression. We can uses other algorithms to get the better prediction.

