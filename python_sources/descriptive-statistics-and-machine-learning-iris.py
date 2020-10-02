#!/usr/bin/env python
# coding: utf-8

# This report aims at applying descriptive statistics and Machine learning techniques to find trends in IRIS data

# <b>Load the Libraries</b>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')
print(check_output(["ls", "../input"]).decode("utf8"))


# <b>Load the Dataset </b>

# In[ ]:


iris_df = pd.read_csv("../input/Iris.csv")


# <b>Descriptive Statistics</b>

# Information about the dataset

# In[ ]:


iris_df.describe()


# In[ ]:


iris_df.info()


# Shape Function to list the records and the features

# In[ ]:


iris_df.shape


# A look at the data using the head function

# In[ ]:


iris_df.head()


# In[ ]:


# The different categories of Species
iris_df.Species.unique()


# Number of Records per species

# In[ ]:


iris = iris_df.groupby('Species',as_index= False)["Id"].count()
iris


# <b>Some visualizations for understanding the spread of these features</b>

# Relationship between the Sepal Length and Width using scatter plot

# In[ ]:


ax = iris_df[iris_df.Species=='Iris-setosa'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris_df[iris_df.Species=='Iris-versicolor'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='white', label='versicolor',ax=ax)
iris_df[iris_df.Species=='Iris-virginica'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=ax)
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_title("Relationship between Sepal Length and Width")


# Similarly for Petal using the seaborn function

# In[ ]:


sns.FacetGrid(iris_df, hue="Species", size=6)    .map(plt.scatter, "PetalLengthCm", "PetalWidthCm")    .add_legend()
plt.title("Relationship between Petal Length and Width")


# <b>Coorelation between the features</b>

# In[ ]:


cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
corr_matrix = iris_df[cols].corr()
heatmap = sns.heatmap(corr_matrix,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols,cmap='Dark2')


# From the above matrix it is seen that Petal Length and Width show a strong coorelation 
# whereas the Sepal Length and Width show weak correlations, it indicates that the Species can be identified 
# better using Petal compared to Sepal,we will verify the same using Machine Learning

# <b>Machine Learning with IRIS data</b>

# In[ ]:


petal = np.array(iris_df[["PetalLengthCm","PetalWidthCm"]])
sepal = np.array(iris_df[["SepalLengthCm","SepalWidthCm"]])

key = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
Y = iris_df['Species'].map(key)


# <b>Creating the Training and testing datasets</b>

# In[ ]:


from sklearn.cross_validation import train_test_split

X_train_S, X_test_S, y_train_S, y_test_S = train_test_split(sepal,Y,test_size=0.2,random_state=42)

X_train_P, X_test_P, y_train_P, y_test_P = train_test_split(petal,Y,test_size=0.2,random_state=42)


# <b>Standardizing and Scaling the features</b>

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train_S)
X_train_std_S = scaler.transform(X_train_S)
X_test_std_S = scaler.transform(X_test_S)

scaler.fit(X_train_P)
X_train_std_P = scaler.transform(X_train_P)
X_test_std_P = scaler.transform(X_test_P)


print('Standardized features for Sepal and Petal \n')
print("Sepal\n\n" +str(X_train_std_S[:2]))
print("\nPetal\n\n" +str(X_train_std_P[:2]))


# <b>Decision Tree Classifier</b>

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini',max_depth=4,presort=True)
model.fit(X_train_std_S,y_train_S)
print('Training Accuracy Sepal = {}'.format(model.score(X_train_std_S, y_train_S)))
print('Testing  Accuracy Sepal = {}'.format(model.score(X_test_std_S, y_test_S)))

model.fit(X_train_std_P,y_train_P)
print('\nTraining Accuracy Petal = {}'.format(model.score(X_train_std_P, y_train_P)))
print('Testing  Accuracy Petal = {}'.format(model.score(X_test_std_P, y_test_P)))


# <b>Logistic Reggression</b>

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_std_S,y_train_S)
print('Training Accuracy Sepal = {}'.format(model.score(X_train_std_S, y_train_S)))
print('Testing  Accuracy Sepal = {}'.format(model.score(X_test_std_S, y_test_S)))

model.fit(X_train_std_P,y_train_P)
print('\nTraining Accuracy Petal = {}'.format(model.score(X_train_std_P, y_train_P)))
print('Testing  Accuracy Petal = {}'.format(model.score(X_test_std_P, y_test_P)))


# <b>k- Nearest Neighbours</b>

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_std_S,y_train_S)
print('Training Accuracy Sepal = {}'.format(model.score(X_train_std_S, y_train_S)))
print('Testing  Accuracy Sepal = {}'.format(model.score(X_test_std_S, y_test_S)))

model.fit(X_train_std_P,y_train_P)
print('\nTraining Accuracy Petal = {}'.format(model.score(X_train_std_P, y_train_P)))
print('Testing  Accuracy Petal = {}'.format(model.score(X_test_std_P, y_test_P)))


# <b>Ensemble Learning: Random Forests ( n Decision trees)</b>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=2)
model.fit(X_train_std_S,y_train_S)
print('Training Accuracy Sepal = {}'.format(model.score(X_train_std_S, y_train_S)))
print('Testing  Accuracy Sepal = {}'.format(model.score(X_test_std_S, y_test_S)))

model.fit(X_train_std_P,y_train_P)
print('\nTraining Accuracy Petal = {}'.format(model.score(X_train_std_P, y_train_P)))
print('Testing  Accuracy Petal = {}'.format(model.score(X_test_std_P, y_test_P)))


# <b>Support Vector Machines</b>

# In[ ]:


from sklearn.svm import LinearSVC

model = LinearSVC(C=10)
model.fit(X_train_std_S,y_train_S)
print('Training Accuracy Sepal = {}'.format(model.score(X_train_std_S, y_train_S)))
print('Testing  Accuracy Sepal = {}'.format(model.score(X_test_std_S, y_test_S)))

model.fit(X_train_std_P,y_train_P)
print('\nTraining Accuracy Petal = {}'.format(model.score(X_train_std_P, y_train_P)))
print('Testing  Accuracy Petal = {}'.format(model.score(X_test_std_P, y_test_P)))


# <b>As stated earlier using the correlation scores,  the Petal Length and Width are the best features to identify the species of IRIS</b>
