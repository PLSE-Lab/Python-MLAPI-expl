#!/usr/bin/env python
# coding: utf-8

# **IRIS Flower Classifier using Random Forest** <br></br><br></br>
# In this kernel, I will be working with the "Hello World" data of Data Science and Machine Learing, the IRIS dataset. I will attempt to make a classifier model using Random Forest. Let's get started..

# In[ ]:


#Import the needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# **Loading the Data** <br></br>
# Iris data is readily available is scikit learn. To load the iris flower dataset, import the load_iris library form sklearn.datasets

# In[ ]:


from sklearn.datasets import load_iris


# In[ ]:


#Create an instance of the load_iris. Here, I'm naming it flower_data but you can name it whatever you want.
flower_data = load_iris()

#Calling the flower_data will give you a dictionary containing the information of the iris data (e.g data, target, target_names, DESCR,
# feature_names and filename)
flower_data


# Each key in the dictionary can be called using the **.** notation. To get the data, you can just type flower_data.data. <br></br>
# This will give you the entire feature data which are the measurements of the sepal length, sepal width, petal length and petal width

# In[ ]:


flower_data.data


# In[ ]:


flower_data.feature_names


# The flower_data.target contains target data which is just a numerical representation of the IRIS flower species (i.e **setosa**, **versicolor**, **virginaca**)

# In[ ]:


flower_data.target


# In[ ]:


flower_data.target_names


# In[ ]:


flower_data.feature_names


# **Convert to Pandas DataFrame**

# In[ ]:


#Converting the data to a pandas dataframe
df = pd.DataFrame(flower_data.data, columns=flower_data.feature_names)

#adding another column for the target and we will call it 'species'
df['species'] = flower_data.target

#Since target is a numerical representation. We will map the corresponding target_names
df['species'] = df['species'].apply(lambda x: flower_data.target_names[x])

#displaying the first top 5 data
df.head()


# Looking at the result by calling the dataframe info method, it shows that there are a total of 150 entries. No missing data on the <br></br> data columns

# In[ ]:


df.info()


# In[ ]:


# Descibe method will show the mead, standard deviation, min, max and percentile
df.describe()


# It can be observe that the iris dataset is a balanced dataset. A total of 50 samples for each flower species.

# In[ ]:


df['species'].value_counts()


# **Exploratory Data Analysis** <br></br><br></br>
# **Exploratory Data Analysis** (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.
# 
# source: https://en.wikipedia.org/wiki/Exploratory_data_analysis

# Let's take a look at how data is distributed based on sepal length and sepal width.

# In[ ]:


sns.set_context(font_scale=2, rc={'font.size':8, 'axes.labelsize':9})
sns.set_style('darkgrid')
sns.FacetGrid(df, hue='species', height=5).map(plt.scatter, 'sepal length (cm)', 'sepal width (cm)').add_legend()

plt.show()


# **Observations** <br></br>
# 1. Using the features sepal length ang sepal width, we can easily classify the setosa flower variety from the 2 other species <br></br>
# 2. It is much harder to classify versicolor and virginica as there is a clearly overlap between data

# A similar scatter plot is shown below based on petal length and petal width 

# In[ ]:


sns.set_style('darkgrid')
sns.FacetGrid(df, hue='species', height = 5).map(plt.scatter, 'petal length (cm)', 'petal width (cm)').add_legend()

plt.show()


# **Onservations**
# 1. Again, the setosa can easily be determine using the petal lenght and petal width features <br></br>
# 2. There is still data overlap between versicolor and virgina but it is not that much using the petal length and petal width features

# **Pairplot**
# Since our iris data contains 4 features or dimensions, it is difficult to visualize in a data in multiple dimension. A solution to this is using a pairplot. A pairplot is simply a 2D plots used to understand the relationship or pattern between two variables or dimensions in our dataset

# In[ ]:


sns.pairplot(df, hue='species', height=3)


# From the above plots, it can be seen that the petal length and petal width are clustered together in fairly well defined groups as compared to sepal length and sepal width 

# **Data Preparation** <br></br>
# From the above **EDA** we can say that it's easier to classify the flowers using the petal length and petal width. We can drop the other features and use only the features that are useful

# In[ ]:


#import the train_test_split
from sklearn.model_selection import train_test_split

#split the data: 70% training and 30% test
features_train, features_test, labels_train, labels_test = train_test_split(
    df.drop(['sepal length (cm)','sepal width (cm)', 'species'], axis=1), flower_data.target, 
    test_size=0.30, random_state=42)


# In[ ]:


#import the RandomForest Classifier from the ensemble
from sklearn.ensemble import RandomForestClassifier

#initialize the classifier
rfc = RandomForestClassifier()

# train the classifier using the training data
rfc.fit(features_train, labels_train)


# In[ ]:


#make predictions
predictions = rfc.predict(features_test)


# In[ ]:


#import the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(labels_test, predictions))


# In[ ]:


print(confusion_matrix(labels_test, predictions))


# **The Random Forest Classifier yeilded a result that is 100% accurate**

# In[ ]:




