#!/usr/bin/env python
# coding: utf-8

# This is one of my firsts here. Please feel free to review the code and leave constructive criticism. 

# ### Importing all the basic libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# Reading the file into iris_df

# In[ ]:


iris_df = pd.read_csv("../input/Iris.csv")
iris_df.head()
#iris_df.info()


# In[ ]:


iris_df.describe()


# I've described the table. I'll try to show some univariate plots.

# In[ ]:


iris_df.groupby('Species').count()


# Violin Plots could help me visualise the description table better.

# In[ ]:


def plotvio(p,i):
    plt.subplot(2,2,i)
    g = sns.violinplot(y=p, x='Species', data=iris_df, inner = 'quartile')
    #plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
plotvio('SepalLengthCm',1)
plotvio('SepalWidthCm',2)
plotvio('PetalLengthCm',3)
plotvio('PetalWidthCm',4)


# In[ ]:


#palette = {'red': 'Iris-setosa','blue': 'Iris-versicolor', 'green': 'Iris-virginica' }

#pd.plotting.scatter_matrix(iris_df, figsize = (10,10))
sns.pairplot(iris_df,hue='Species',diag_kind='kde')


# * Most of the plots show a linear relationship
# * Also, these form clusters according to their Species Type.
# 

# * The 'Id' column won't really help in classifying a particular instance, so I drop(or delete) it.

# In[ ]:


iris_df.drop('Id',axis=1,inplace = True)
iris_df.head()


# A correlation map would help us see how strongly 2 parameters are related to each other which I might use later for training the model.

# In[ ]:


#iris_df.corr().head()
sns.heatmap(iris_df.corr(),annot=True)


# * SepalWidth is almost independent of SepalLength but loosely dependent(-ve corr) on the other 2 parameters
# * Both the Petal parameters are very strongly dependent on each other, and the SepalLength.
# 
# This being a classification problem, I don't really see the need for any regression curves for data visualization and hence, the scatter plots along with the heatmap seems enough for identifying how the parameters vary with each other.
# 
# 
# ## Choosing an Algorithm
# We first import all the required modules and packages from scikit-learn

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# I create seperate the 'Species' (target variable) and rest of the columns (features). 

# In[ ]:


iris_df_pre = iris_df.drop('Species', axis=1)
iris_Spe = iris_df['Species']


# #### *Splitting the data into training set and test set so that we can train using one set and test using another.*

# In[ ]:


train_p, test_p, train_t, test_t = train_test_split(iris_df_pre, iris_Spe, test_size = 0.333)


# ## 1. K Nearest Neighbours

# In[ ]:


N = [] #No. of Neighbours
A = [] #Accuracy Score

for k in range(1,30):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_p, train_t)
    y_pred = model.predict(test_p)
    A.append(accuracy_score(test_t, y_pred))
    N.append(k)
    
plt.grid(True)
plt.plot(N,A)


# ## 2. Support Vector Machine

# In[ ]:


model = SVC(gamma = 'scale')
model.fit(train_p, train_t)
y_pred = model.predict(test_p)
accuracy_score(test_t, y_pred)


# ## 3. Decision Tree

# In[ ]:


model = DecisionTreeClassifier()
model.fit(train_p, train_t)
y_pred = model.predict(test_p)
accuracy_score(test_t, y_pred)


# ## 4. Random Forest

# In[ ]:


N = [] #No. of Neighbours
A = [] #Accuracy Score

for k in range(1,20):
    model = RandomForestClassifier(n_estimators=k)
    model.fit(train_p, train_t)
    y_pred = model.predict(test_p)
    A.append(accuracy_score(test_t, y_pred))
    N.append(k)
    
plt.grid(True)
plt.plot(N,A)

