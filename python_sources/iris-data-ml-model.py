#!/usr/bin/env python
# coding: utf-8

# # IRIS
# 
# The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.
# 
# It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
# 
# The columns in this dataset are:
# 
# - Id
# - SepalLengthCm
# - SepalWidthCm
# - PetalLengthCm
# - PetalWidthCm
# - Species
# 
# <img src = "https://miro.medium.com/max/361/0*1lgB-Yqej6VPER00" alt="Drawing" width="400">
# <figcaption> <h6> Iris Flower : Getting started with Machine Learning since 1988 .</h6>

# In[ ]:


#for genral opeartions
import pandas as pd

#for visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#for warnings
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.simplefilter('ignore')


# In[ ]:


#reading the data from csv file
data = pd.read_csv("../input/iris/Iris.csv")
data.head(3)


# In[ ]:


#dropping the id column
data.drop('Id', axis = 1, inplace = True)
data.head(3)


# In[ ]:


#checking the null values in dataset
data.isnull().any()


# In[ ]:


#spliting the dataset into X and y variables as independent and target feature

y=data.Species #the target feature

X=data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] #the independent features
X.describe()# for the brief statistics of the dataset


# In[ ]:


#plotting pairplot to check the target feature spread with respect to indpendent variables
sns.set_style("ticks")
sns.pairplot(data,hue = 'Species',diag_kind = "kde",kind = "scatter",palette = "husl")
plt.show()


# In[ ]:


sns.set(style="dark", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
#sns.despine(left=True)

sns.swarmplot(x="Species", y="PetalLengthCm", data=data, ax=axes[0, 0])
sns.swarmplot(x="Species", y="PetalWidthCm", data=data, ax=axes[0, 1])
sns.swarmplot(x="Species", y="SepalLengthCm", data=data, ax=axes[1, 0])
sns.swarmplot(x="Species", y="SepalWidthCm", data=data, ax=axes[1, 1])
plt.xticks()
plt.tight_layout()


# In[ ]:


sns.lmplot(x="PetalLengthCm", y="SepalLengthCm",data=data)


# In[ ]:


sns.lmplot(x="PetalWidthCm", y="SepalWidthCm",data=data)


# In[ ]:


sns.lmplot(x="PetalLengthCm", y="PetalWidthCm",data=data)
fig=plt.gcf()


# In[ ]:


sns.lmplot(x="SepalLengthCm", y="SepalWidthCm",data=data)


# In[ ]:


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
#sns.despine(left=True)

sns.distplot(data['SepalLengthCm'], hist = True, color="b", kde_kws={"shade": True}, ax=axes[0, 0])
sns.distplot(data['SepalWidthCm'], hist=True, rug=False, color="r", ax=axes[0, 1])
sns.distplot(data['PetalLengthCm'], hist=True, color="g", kde_kws={"shade": True}, ax=axes[1, 0])
sns.distplot(data['PetalWidthCm'], color="m", ax=axes[1, 1])

plt.xticks()
plt.tight_layout()


# In[ ]:


#individual distribuition of length and width
data.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# In[ ]:


#Species wise variation of length and width
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=data)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=data)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=data)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=data)


# In[ ]:


#checking the outliers and data quartiles
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(x='Species',y='PetalLengthCm',data=data)
plt.subplot(2,2,2)
sns.boxplot(x='Species',y='PetalWidthCm',data=data)
plt.subplot(2,2,3)
sns.boxplot(x='Species',y='SepalLengthCm',data=data)
plt.subplot(2,2,4)
sns.boxplot(x='Species',y='SepalWidthCm',data=data)


# In[ ]:


#checking on the correaltion between features
plt.figure(figsize=(7,4)) 
sns.heatmap(data.corr(),annot=True,cmap='cubehelix_r')
plt.show()

sns.set_style('darkgrid')
f,axes=plt.subplots(2,2,figsize=(15,15))

k1=sns.boxplot(x="Species", y="PetalLengthCm", data=data,ax=axes[0,0])
k2=sns.violinplot(x='Species',y='PetalLengthCm',data=data,ax=axes[0,1])
k3=sns.stripplot(x='Species',y='SepalLengthCm',data=data,jitter=True,edgecolor='gray',size=8,palette='winter',orient='v',ax=axes[1,0])
#axes[1,1].hist(iris.hist,bin=10)
axes[1,1].hist(data.PetalLengthCm,bins=100)
#k2.set(xlim=(-1,0.8))
plt.show()
# ### Building Machine Learning Models

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#defining logistic regression model
def logreg(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_val)
    print("Train Data Accuracy {0} ".format(lr.score(X_train, y_train)))
    print("Test Data Accuracy {0} ".format(lr.score(X_val, y_val)))


# In[ ]:


#defining random forest model
def randfrst(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    rfc= RandomForestClassifier()
    rfc.fit(X_train,y_train)
    y_pred = rfc.predict(X_val)
    print("Train Data Accuracy {0} ".format(rfc.score(X_train, y_train)))
    print("Test Data Accuracy {0} ".format(rfc.score(X_val, y_val)))


# In[ ]:


#defining decision tree model
def destree(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    dtc= RandomForestClassifier()
    dtc.fit(X_train,y_train)
    y_pred = dtc.predict(X_val)
    print("Train Data Accuracy {0} ".format(dtc.score(X_train, y_train)))
    print("Test Data Accuracy {0} ".format(dtc.score(X_val, y_val)))


# In[ ]:


#defining k nearest neighbour model
def knn_mod(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    a_index=list(range(1,11))
    a=pd.Series()
    x=[1,2,3,4,5,6,7,8,9,10]
    for i in list(range(1,11)):
        model=KNeighborsClassifier(n_neighbors=i) 
        model.fit(X_train,y_train)
        prediction=model.predict(X_val)
        a=a.append(pd.Series(metrics.accuracy_score(prediction,y_val)))
    plt.plot(a_index, a)
    plt.xticks(x)
    print("Train Data Accuracy {0} ".format(model.score(X_train, y_train)))
    print("Test Data Accuracy {0} ".format(model.score(X_val, y_val)))


# In[ ]:


#defining support vector machine model
def svm_mod(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    model = svm.SVC()
    model.fit(X_train,y_train)
    prediction=model.predict(X_val)
    print("Train Data Accuracy {0} ".format(model.score(X_train, y_train)))
    print("Test Data Accuracy {0} ".format(model.score(X_val, y_val)))


# ### Checking Accuracy Modelwise

# In[ ]:


logreg(X,y)


# In[ ]:


randfrst(X,y)


# In[ ]:


destree(X,y)


# In[ ]:


svm_mod(X,y)


# In[ ]:


knn_mod(X,y)


# # THE END!!!
