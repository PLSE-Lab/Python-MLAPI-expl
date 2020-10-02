#!/usr/bin/env python
# coding: utf-8

# **    Analysis of the Iris Data Set**
# 
# Iris is the name of a flower.It is located in the Northern hemisphere zones, from Europe to Asia and across North America.The dataset that we have covers 3 types of Iris species,
# Iris Setosa
# Iris Virginica
# Iris Versicolor
# The dataset has an intersting history,I was able to learn about this from wiki page,
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis".
# It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species.
# 
# 

#                     **General Introduction**
# 
# In any form of analysis involving data,the main goal of it will be to solve a problem,therefore there is a pipeline to reach the goal.The starting point being gathering required data ,cleaning,visualising and so on.
# 
# Last step being building machine learning models and validating them.
# Have a look at this article,
# https://towardsdatascience.com/a-beginners-guide-to-the-data-science-pipeline-a4904b2d8ad3
# 

# **Problem statement**
# 
# **To predict the type of Iris species based on other features in the dataset pertaining to sepal length,width,petal length and width.
# 
# Having said that we need to know that there are different class of machine learning problems
# 1.Supervised - Those which has a pre defined target variable.For every row in the dataset there is a labeled outcome which would be used as a reference to train the model.
# 2.Unsupervised Learning - There is no pre defined target variable. There are no prior labellings but we have to predict,group and collate the features,so that they are meaningful.
# 
# Our dataset here has a pre defined target variable Species,which qualifies this one to be a supervised learning problem.
# 
# More over our target to be predicted is categorical and hence this is a classification problem.
# **"Be kind,fork and upvote the kernel,if you find it useful"**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler, LabelBinarizer
# auxiliary function
from sklearn.preprocessing import LabelEncoder
def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


##loading the data
iris = pd.read_csv("../input/Iris.csv") 
##Getting a gist of the dataset in hand
iris.describe()


# In[ ]:


#Glancing features
iris.info()


# In[ ]:


#First few observations
iris.head(20)


# We could see that the dataset has 1 Identifier,4 numerical variables & 1 Categorical Variable.
# Our task is to predict the Species with the help of all other dependent variables.
# Therefore,our assumption here is that Species is the dependent variable or the variable to be predicted.
# All other varaibles ID,SepalLengthCm,SepalWidthCm,PetalLengthCm & PetalWidthCm are Independent  variables.
# 
# 

# The data that we see in the real world contains noise and are generally not ready for instant analysis.Therefore we need to cleanse them for making it usable.

# In[ ]:


iris.isnull()


# Looks like an illusion that we do not have any null values!Great,time saver!
# Should check though.

# In[ ]:


#One more way to quickly check if your assumption about the widths and lengths of Sepal and Petal 
#differs for different species
Pivot = iris.pivot("Species","Id")
print(Pivot)
sns.heatmap(Pivot,cbar = True)


# ![image.png](attachment:image.png)
# 
# The heatmap above has grouped the species and indicates that our assumption about the Sepal and Petal lengths,widths are true.They all vary significantly for every speices.
# Iris - Setosa has **smaller** length & width for both Petal and Sepal.
# Iris - Versicolor has **Mediocre** length & width for both Petal and Sepal.
# Iris - Virginica has **bigger** length & width for both Petal and Sepal.

# Now some visualisations,basic ones.

# 

# In[ ]:


iris.hist()
plt.show()


# In[ ]:


##scatter plot

scatter_matrix(iris)
plt.show()


# In[ ]:


iris.Species.value_counts().plot(kind='pie')
#Pie chart shows that all 3 species are distributed equally.


# The diagonal grouping of the variables suggests that there is a relationship between the objects.This is also called as correlation of the varaibles.High correlation indicates high predictability with fewer features.We will talk about prediction later.

# In[ ]:


iris['Species']=iris['Species'].astype('category')
iris.dtypes


# In the code above,the Species is converted to Categorical data type.

# In[ ]:


plt.figure()
fig,ax=plt.subplots(1,2,figsize=(17, 19))##
iris.plot(x="SepalLengthCm",y="SepalWidthCm",kind="scatter",ax=ax[0],sharex=False,sharey=False,label="sepal",color='g')
iris.plot(x="PetalLengthCm",y="PetalWidthCm",kind="scatter",ax=ax[1],sharex=False,sharey=False,label="petal",color='y')
ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()
##Sepal comparison shows that the range of the sepal width is between 2 to 4.5,most of the data points are between 2.5 to 3.5.
#Petal Comparision shows that the range of the petal width is between 0.5 to 2.5,intersting observation here is that there are no points in the scatterplot between 0.5 & 1.
#There is a gradual increase in the petal length from 1 to 2.5
#From this we can conclude that there is a finite pattern in the sepal and petal widths,lengths.


# In[ ]:


##I don't recommend this part of the code for beginners
setosa=iris[iris['Species']=='Iris-setosa']
versicolor =iris[iris['Species']=='Iris-versicolor']
virginica =iris[iris['Species']=='Iris-virginica']

print(setosa.describe())
print(versicolor.describe())
print(virginica.describe())

##using the data frame to plot a scatter plot for comparing different species
##Thanks Abhishek Gupta for this method:)



plt.figure()
fig,ax=plt.subplots(1,2,figsize=(21, 10))##1,2 indicates the visual will be fitted in a row and 2 columns.

setosa.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter",ax=ax[0],label='Setosa',color='r')
virginica.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter",ax=ax[0],label='Virginica',color='g')
versicolor.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter",ax=ax[0],label='Versicolor',color='b')

setosa.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter",ax=ax[1],label='Setosa',color='r')
versicolor.plot(x="PetalLengthCm",y="PetalWidthCm",kind="scatter",ax=ax[1],label='Versicolor',color='b')
virginica.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter", ax=ax[1], label='Virginica', color='g')

ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()
#As we have seen in the previous scatterplot,this one confirms it.


# **Correlation Coefficients
# **I have already mentioned that there is a high level of predictability with the available variables.One needs to examine the correlation between the variables.
# Correlation is applicable only for numerical data.

# In[ ]:


iris_1=iris.drop(['Id'],axis=1)
iris_1.head()
iris_1.corr()


# Interpreting the correlation matrix,
# 1.SepalLengthCm - PetalLengthCm,PetalWidthCm are highly correlated positively.
# 2.SepalWidthCm - Has negative relationship with all the other varaibles.
# 3.PetalLengthCm - SepalLengthCm & PetalWidthCm are highly correlated.
# 4.PetalWidthCm - SepalLengthCm & PetalWidthCm are highly correlated.

# The visual above shows that all the three categories of species are of the same proportion.

# In[ ]:


iris.plot(kind = "scatter",x = "SepalLengthCm",y = "SepalWidthCm")


# In[ ]:



pd.crosstab(iris.SepalLengthCm,iris.Species).plot(kind='bar')
plt.title('Sepal Length vs Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.subplots_adjust(bottom=0.1, right=1.9, top=1.5)

pd.crosstab(iris.SepalWidthCm,iris.Species).plot(kind = 'bar')
plt.title('Sepal Width vs Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width')
plt.subplots_adjust(bottom=0.1, right=1.9, top=1.5)


# From this histogram we could see that there are few observations where the Sepal lengths are almost the same.
# We coudld also see that the extreme points have distinct species captured.
# We can confirm that there is a distinction between the Species with respect to Sepal Lengths & Widths.
# 
# 
# 

# In[ ]:



pd.crosstab(iris.PetalLengthCm,iris.Species).plot(kind='bar')
plt.title('Petal Length vs Species')
plt.xlabel('Species')
plt.ylabel('Petal Length')
plt.subplots_adjust(bottom=0.1, right=1.9, top=1.5)

pd.crosstab(iris.PetalWidthCm,iris.Species).plot(kind = 'bar')
plt.title('Petal Width vs Species')
plt.xlabel('Species')
plt.ylabel('Petal Width')
plt.subplots_adjust(bottom=0.1, right=1.9, top=1.5)


# Petal Length and Petal Width also shows uniqueness for different species.
# Iris Setosa and Iris Virginica have clear distinction of Petal Lengths and Widths.
# Some of them have overlapping features.

# Above visual shows the max and min values of Sepal Length for different Species and some more interesting things.

# In[ ]:


sns.boxplot(x = "Species",y = "SepalLengthCm",data = iris)


# In[ ]:


sns.pairplot(iris, hue='Species', size=2.5)
#The pair plot also shows some overlap.


# In[ ]:


x = iris.PetalLengthCm
y = iris.PetalWidthCm
plt.scatter(x, y ,s = 10*x,data = iris,cmap = "plasma",c = iris.PetalLengthCm)
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.subplots_adjust(bottom=0.1, right=1.9, top=1.5)


# In[ ]:


x =iris.SepalLengthCm
y =iris.SepalWidthCm
plt.scatter(x,y,s = 10*x,data = iris,cmap = "plasma",c = iris.SepalLengthCm)
plt.xlabel("SepalLength")
plt.ylabel("SepalWidth")
plt.subplots_adjust(bottom=0.1, right=1.9, top=1.5)


# Model Building

# #This is another way
# #train=iris.sample(frac=0.8,random_state=200)
# #test=iris.drop(train.index)
# 
# train,test = train_test_split(iris,test_size = 0.2)
# train.info();test.info()

# In[ ]:


##Learned this encoding technique from Ranjeet Jain's Iris notebook
x = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = iris['Species']
encoder = LabelEncoder()
y = encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)


# Machine learning  to predict the type of Iris flower based on other parameters.
# Since this is a classification problem we will explore related algorithms.
# 

#                                                                               **Logistic Regression **                     

# In[ ]:


lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
lr_predict = lr_model.predict(x_test)
print('Logistic Regression - ',accuracy_score(lr_predict,y_test))


# In[ ]:


print("Confusion Matrix",confusion_matrix(y_test,lr_predict))


# In[ ]:


print("Accuracy :" ,accuracy_score(y_test,lr_predict)*100)


# In[ ]:


print("Report :" ,classification_report(y_test,lr_predict))


# **                                                                                      Decision Tree**

# In[ ]:


d_tree = DecisionTreeClassifier(max_depth = 4)
d_tree_fit = d_tree.fit(x_train,y_train)
d_tree_pred = d_tree.predict(x_test)
print("Decision Tree Accuracy",accuracy_score(d_tree_pred,y_test))


# In[ ]:


print("Confusion Matrix: ", 
        confusion_matrix(y_test, d_tree_pred)) 
      
print ("Accuracy : ", 
    accuracy_score(y_test,d_tree_pred)*100) 
      
print("Report : ", 
    classification_report(y_test, d_tree_pred))


# **                                                                               Randomforest Model**

# In[ ]:


rf_model = RandomForestClassifier(max_depth = 3)
rf_fit = rf_model.fit(x_train,y_train)
fr_tree_pred = rf_fit.predict(x_test)


# In[ ]:


print("Confusion Matrix:",confusion_matrix(y_test,fr_tree_pred))


# In[ ]:


print("Accuracy Score:",accuracy_score(y_test,fr_tree_pred)*100)


# In[ ]:


print("EvaluationReport : ",classification_report(y_test,fr_tree_pred))


# Which is the best model?

# Inorder to decide which model is the best model,we can consider the accuracy scores.
# You can see that the Accuracy score for Random Forest is higher compared to all other models.
# One more observation with respect to confusion matrix here is ,Random forest had 0 mis- calssification.All other models had misclassification to some degree.
# 

# In[ ]:


from sklearn.cluster import KMeans
# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(x_train,y_train)
# Getting the cluster labels
labels = kmeans.predict(x_test,y_test)
# Centroid values
centroids = kmeans.cluster_centers_
print(centroids)


# In[ ]:


colormap = np.array(['purple', 'black', 'yellow'])
plt.subplot(1, 2, 1)
plt.subplots_adjust(right = 2,left = 0.3)
plt.scatter(x_test.PetalLengthCm, x_test.PetalWidthCm, c=colormap[y_test], s=40)
plt.title('K Mean Classification Test Data')
plt.subplot(1, 2, 2)
plt.scatter(x_train.PetalLengthCm, x_train.PetalWidthCm, c=colormap[y_train], s=40)
plt.title('K Mean Classification on Train Data')


# The scatter plot depicts the Species of Iris.There are no miss classifications in the dataset.We can be assured of this as there are no overlapping dots in the graph for test set.
