#!/usr/bin/env python
# coding: utf-8

# Just forked the script  
# 
# a flower is three petals and three sepals
# the surface of the flower is the radius of the sepals-length^2 x 3.14  
# the coverance of the flower is the surface of the sepalsx3+surface of petals x3  / surface of sepal-length^2  3.14
# 

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

# Shape Function to list the records and the features

# A look at the data using the head function

# In[ ]:


iris_df.head()


# In[ ]:


# The different categories of Species
iris_df.Species.unique()
temp=((iris_df.PetalLengthCm*iris_df.PetalWidthCm*3+iris_df.SepalLengthCm*iris_df.SepalWidthCm*3)/ ( iris_df.SepalLengthCm*iris_df.SepalLengthCm*3.14))
iris_df['Sepaldiv'] = temp.values
temp = iris_df.PetalLengthCm*iris_df.PetalWidthCm
iris_df['Petalm'] = temp.values


# Number of Records per species

# <b>Some visualizations for understanding the spread of these features</b>

# Relationship between the Sepal Length and Width using scatter plot
# 
# **now after looking a few seconds attendfully to the graph you see that they describe a surface of the Petals (width x length)  and the Sepals tend to be reverse smaller versus the length**
# 
# The scatterplot xy graph nicely splits up in three parts

# In[ ]:


ax = iris_df[iris_df.Species=='Iris-setosa'].plot.scatter(x='Sepaldiv',y='Petalm',color='blue', label='Setosa')
iris_df[iris_df.Species=='Iris-versicolor'].plot.scatter(x='Sepaldiv',y='Petalm',color='green', label='versicolor',ax=ax)
iris_df[iris_df.Species=='Iris-virginica'].plot.scatter(x='Sepaldiv',y='Petalm',color='red', label='virginica', ax=ax)
ax.set_xlabel("Sepaldivision or Surface covering of the flower")
ax.set_ylabel("petalsurface")
ax.set_title("Relationship between Sepal Length and Width")


# Similarly for Petal using the seaborn function

# In[ ]:


# Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
# 
# From the pairplot, we'll see that the Iris-setosa species is separataed from the other
# two across all feature combinations
sns.pairplot(iris_df.drop("Id", axis=1), hue="Species", size=3)


# <b>Coorelation between the features</b>

# In[ ]:


cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm','Sepaldiv','Petalm']
corr_matrix = iris_df[cols].corr()
heatmap = sns.heatmap(corr_matrix,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols,cmap='Dark2')


# From the above matrix it is seen that Petal Length and Width show a strong coorelation 
# whereas the Sepal Length and Width show weak correlations, it indicates that the Species can be identified 
# better using Petal compared to Sepal,we will verify the same using Machine Learning

# <b>Machine Learning with IRIS data</b>

# In[ ]:


petal = np.array(iris_df[["PetalWidthCm","PetalLengthCm"]])
sepal = np.array(iris_df[["Petalm","PetalWidthCm"]])

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


# <b>As stated earlier using the correlation scores,  the Petal Length and Width are the best features to identify the species of IRIS</b>As we can conclude simply reasoning on the data, and looking at flowers you can discover faster relations then doing utterly difficult math
