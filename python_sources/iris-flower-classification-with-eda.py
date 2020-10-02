#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from pandas.tools import plotting
from sklearn.model_selection import train_test_split
from plotly import tools
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlin', 'inline')


# In[ ]:


#loading dataset
iris = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")


# In[ ]:


#printing how many features?
print("the dataset has {} rows and {} features".format(iris.shape[0],iris.shape[1]))


# In[ ]:


#visualising data
table = ff.create_table(iris.head())
py.iplot(table,filename='jupyter-table1')


# In[ ]:


#printing info about dataset
print(iris.info())


# **The data types are correct and we can see that there is no null values in the dataset**

# In[ ]:


#printing statistics about the dataset
print(iris.describe())
#py.iplot(ff.create_table(iris.describe()),filename='describe_table')


# In[ ]:


#printing coefficient of Quartile deviation
qd = pd.DataFrame((iris.describe().loc['75%']-iris.describe().loc['25%'])/(iris.describe().loc['75%']+iris.describe().loc['25%']),columns=['COQD'])
print(qd)


# In[ ]:


#how many datapoints for each class are presents?
print(iris["species"].value_counts())


# In[ ]:


#which categories of flowers we have?
print(" we have {} types of species in this dataset".format(iris.species.nunique()))
print("the names of this species are:",iris.species.unique())


# In[ ]:


#2-D scatter plots
iris.plot(kind='scatter', x='sepal_length', y='sepal_width') ;
plt.show()


# In[ ]:


iris.plot(kind='scatter', x='petal_length', y='petal_width') ;
plt.show()


# * cannot make much sense out it.
# * What if we color the points by thier class-label/flower-type.

# In[ ]:


# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style("whitegrid")
sns.FacetGrid(data=iris,hue="species",size=6)   .map(plt.scatter,"sepal_length","sepal_width")   .add_legend()
plt.show()


# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(data=iris,hue="species",size=6)   .map(plt.scatter,"petal_length","petal_width")   .add_legend()
plt.show()


# * Using sepal_length and sepal_width features, we can distinguish Setosa flowers from others.
# * Seperating Versicolor from Viginica is much harder as they have considerable overlap.

# In[ ]:


#pair-plot
sns.set_style("whitegrid")
sns.pairplot(data=iris,hue="species",size=3)   .add_legend()
plt.show()


# * petal_length and petal_width are the most useful features to identify various flower types.
# * While Setosa can be easily identified (linearly seperable), Virnica and Versicolor have some overlap (almost linearly seperable).
# * We can find "lines" and "if-else" conditions to build a simple model to classify the flower types.

# In[ ]:


#histograms
iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(15,6)
plt.show()


# In[ ]:


#Andrews curves are a method for visualizing multidimensional data by mapping each observation onto a function.
#https://glowingpython.blogspot.com/2014/10/andrews-curves.html
plt.figure(figsize=(10,8))
plotting.andrews_curves(iris,'species')
#In the plot below, the each color used represents a class
#we can easily note that the lines that represent samples from the same class have similar curves.


# In[ ]:


#distplot

sns.FacetGrid(data=iris, hue="species", size=5)    .map(sns.distplot, "petal_length")    .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(data=iris, hue="species", size=5)    .map(sns.distplot, "petal_width")    .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(data=iris, hue="species", size=5)    .map(sns.distplot, "sepal_length")    .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(data=iris, hue="species", size=5)    .map(sns.distplot, "sepal_width")    .add_legend()
plt.show()


# In[ ]:


#Let's check if we have any correlation between features
#https://seaborn.pydata.org/generated/seaborn.heatmap.html
plt.figure(figsize=(7,4)) 
sns.heatmap(iris.corr(),annot=True,fmt="f",cmap="YlGnBu")
plt.show()


# In[ ]:


#Plot CDF of petal_length

iris_setosa = iris[iris['species']=='Iris-setosa']
print(iris_setosa.head())
counts, bin_edges= np.histogram(iris_setosa['petal_length'],bins=10, density= True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)
print(cdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()


# In[ ]:


#Plot CDF of petal_width

iris_setosa = iris[iris['species']=='Iris-setosa']
print(iris_setosa.head())
counts, bin_edges= np.histogram(iris_setosa['petal_width'],bins=10, density= True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)
print(cdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()


# In[ ]:


#Box-plot with whiskers: another method of visualizing the  1-D scatter plot more intuitivey.
sns.boxplot(x='species',y='petal_length', data=iris)
plt.show()


# In[ ]:


sns.boxplot(x='species',y='petal_width', data=iris)
plt.show()


# In[ ]:


sns.boxplot(x='species',y='sepal_length', data=iris)
plt.show()


# In[ ]:


sns.boxplot(x='species',y='sepal_width', data=iris)
plt.show()


# In[ ]:


# A violin plot combines the benefits of the previous two plots 
#and simplifies them

# Denser regions of the data are fatter, and sparser ones thinner 
#in a violin plot

sns.violinplot(x="species", y="petal_length", data=iris, size=8)
plt.show()


# In[ ]:


sns.violinplot(x="species", y="petal_width", data=iris, size=8)
plt.show()


# In[ ]:


sns.violinplot(x="species", y="sepal_length", data=iris, size=8)
plt.show()


# In[ ]:


sns.violinplot(x="species", y="sepal_length", data=iris, size=8)
plt.show()


# In[ ]:


#splitting of data
X_train, X_test, y_train,y_test = train_test_split(iris.iloc[:,:4],iris.species,stratify=iris.species,test_size = 0.3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


#random samples of training data
X_train.head()


# In[ ]:


#training svc model
SVC = SVC()
SVC.fit(X_train,y_train)
y_predict = SVC.predict(X_test)
print("the accuracy of the Support Vector Machine Classifier model is :",accuracy_score(y_test,y_predict))
confusion_matrix(y_predict,y_test)


# In[ ]:


#training logistic regression
LG = LogisticRegression()
LG.fit(X_train,y_train)
y_predict = LG.predict(X_test)
print("the accuracy of the Logistic Regresssion Classifier model is :",accuracy_score(y_predict,y_test))
confusion_matrix(y_predict,y_test)


# In[ ]:


#training decision tree classifier model
DTC = DecisionTreeClassifier(max_leaf_nodes = 3)
DTC.fit(X_train,y_train)
y_predict = DTC.predict(X_test)
print("the accuracy of the Decision Tree Classifier model is :",accuracy_score(y_test,y_predict))
confusion_matrix(y_predict,y_test)


# In[ ]:


#training extra tree classfier model
ETC=ExtraTreesClassifier()
ETC.fit(X_train,y_train)
ETC_prediction=ETC.predict(X_test)
print('The accuracy of the Extra Trees Classifier model is',accuracy_score(ETC_prediction,y_test))
confusion_matrix(ETC_prediction,y_test)


# In[ ]:


#training KNN model
KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train,y_train)
KNN_prediction=KNN.predict(X_test)
print('The accuracy of the KNeighborsClassifier model is',accuracy_score(KNN_prediction,y_test))
confusion_matrix(KNN_prediction,y_test)


# In[ ]:


#training gaussian naive bayes model
GNB=GaussianNB()
GNB.fit(X_train,y_train)
GNB_prediction=GNB.predict(X_test)
print('The accuracy of the GaussionNB model is',accuracy_score(GNB_prediction,y_test))
confusion_matrix(GNB_prediction,y_test)


# In[ ]:


#training random forest classfier model
RFC=RandomForestClassifier()
RFC.fit(X_train,y_train)
RFC_prediction=RFC.predict(X_test)
print('The accuracy of the RandomForestClassifier model is',accuracy_score(RFC_prediction,y_test))
confusion_matrix(RFC_prediction,y_test)


# In[ ]:


#training XGboost model
XGB=XGBClassifier()
XGB.fit(X_train,y_train)
XGB_prediction=XGB.predict(X_test)
print('The accuracy of the XGBClassifier model is',accuracy_score(XGB_prediction,y_test))
confusion_matrix(XGB_prediction,y_test)


# In[ ]:


#lets try to build a Deep learning model with keras
from sklearn.preprocessing import StandardScaler, LabelBinarizer
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris["species"]

X = StandardScaler().fit_transform(X)
y = LabelBinarizer().fit_transform(y)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[ ]:


model = Sequential()
model.add(Dense( 4, input_dim=4, activation = 'relu'))
model.add(Dense( units = 10, activation= 'relu'))
model.add(Dense( units = 3, activation= 'softmax'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[ ]:


model_training = model.fit(x_train, y_train, epochs = 150, validation_data = (x_test, y_test))


# In[ ]:




