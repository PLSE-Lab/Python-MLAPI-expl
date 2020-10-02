#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# 
# **In this kernel,**
# We will apply the machine learning algorithms on our data.Then we will find which machine learning algorithm is the best for our data at the end of tutorial.
# 
# 
# **Content:**
# 1. [Exploratory Data Analysis (EDA)](#1)
# 2. [EDA Visualization](#2)
#    2. [Correlation Map](#3)
#    2. [Pair Plot](#4)
#    2. [Scatter Matrix](#5)
#    2. [Scatter Matrix-Abnormal Class](#23)
#    2. [Scatter Matrix-Normal Class](#24)
# 3. [Data PreProcessing](#6)
#    3. [Normalization Data](#7)
#    3. [Train-Test Split Data](#8)
# 4. [Machine Learning Classification Models](#9)
#       4. [Logistic Regression Classification](#10)
#       4. [K-Nearest Neighbors (KNN) Classification](#11)
#          4. [Model Complexity](#12)
#       4. [Support Vector Machine(SVM) Classification](#13)
#       4. [Naive Bayes Classification](#14)
#       4. [Decision Tree Classification](#15)
#       4. [Random Forest Classification](#16)
# 5. [Evaluation Classification Models](#17)
#    5. [Confusion Matrixes Comparison](#18)
#    5. [Bar Charts Comparison](#19)
# 6. [Conclusion](#20)   

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff # import figure factory

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# close warning
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="1"></a> <br>
# # Exploratory Data Analysis (EDA)

# ### Import Data

# In[ ]:


df = pd.read_csv("../input/column_2C_weka.csv")


# In[ ]:


# to see features and target variable
df.head()


# In[ ]:


# Display the content of data
df.info()


# As you can see:
# 
# * length: 310 (range index)
# * Features are float
# * Target variables are object that is like string

# In[ ]:


# shape gives number of rows and columns in a tuple
df.shape


# In[ ]:


df.describe()


# In[ ]:


# Display positive and negative correlation between columns
df.corr()


# In[ ]:


#sorts all correlations with ascending sort.
df.corr().unstack().sort_values().drop_duplicates()


# <a id="2"></a> <br>
# # EDA Visualization

# <a id="3"></a> <br>
# #### Correlation Map

# In[ ]:


#correlation map
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, linewidth=".5", cmap="RdPu", fmt=".2f", ax = ax)
plt.title("Correlation Map",fontsize=20)
plt.show()


# <a id="21"></a> <br>
# ### Pair Plot

# In[ ]:


sns.pairplot(data=df,hue="class",palette="Set1")
plt.suptitle("Pair Plot of Data",fontsize=20)
plt.show()   # pairplot without standard deviaton fields of data


# <a id="5"></a> <br>
# ### Scatter Matrix

# **pd.plotting.scatter_matrix:**
# * green: *normal* and red: *abnormal*
# * c:  color
# * figsize: figure size
# * diagonal: histohram of each features
# * alpha: opacity
# * s: size of marker
# * marker: marker type 

# In[ ]:


color_list = ["red" if each=="Abnormal" else "cyan" for each in df.loc[:,"class"]]
pd.plotting.scatter_matrix(df.loc[:, df.columns != "class"],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal="hist",
                                       alpha=0.5,
                                       s = 200,
                                       marker = "*",
                                       edgecolor= "black")
plt.show()


# <a id="23"></a> <br>
# ### Scatter Matrix-Abnormal Class

# In[ ]:


df_abnormal = df[df["class"]=="Abnormal"]
pd.plotting.scatter_matrix(df_abnormal.loc[:, df_abnormal.columns != "class"],
                                       c="red",
                                       figsize= [15,15],
                                       diagonal="hist",
                                       alpha=0.5,
                                       s = 200,
                                       marker = "*",
                                       edgecolor= "black")
plt.show()


# <a id="24"></a> <br>
# ### Scatter Matrix-Normal Class

# In[ ]:


df_normal = df[df['class']=='Normal']
pd.plotting.scatter_matrix(df_normal.loc[:, df_normal.columns != "class"],
                                       c="cyan",
                                       figsize= [15,15],
                                       diagonal="hist",
                                       alpha=0.5,
                                       s = 200,
                                       marker = "*",
                                       edgecolor= "black")
plt.show()


# In[ ]:


# prepare data
data1 = len(df["class"][df["class"] == "Abnormal"])
data2 = len(df["class"][df["class"] == "Normal"])

data = [go.Bar(
            x=["Abnormal","Normal"],
            y=[data1,data2],
            marker=dict(color='rgb(158,202,225)',
            line=dict(color='rgba(254, 69, 62, 1)',
            width=1.5),
        ),
    opacity=0.6
    )]

iplot(data, filename='text-hover-bar')


# <a id="6"></a> <br>
# # Data PreProcessing 

# In[ ]:


df["class"] = [0 if each == "Abnormal" else 1 for each in df["class"]]

y = df["class"].values
x_data = df.drop(["class"], axis=1)


# <a id="7"></a> <br>
# ### Normalization Data

# * Normalization Formula = (x - min(x))/(max(x)-min(x))

# In[ ]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[ ]:


x.head()


# In[ ]:


x.isnull().sum() #Indicates values not defined in our data


# In[ ]:


x.isnull().sum().sum()  #Indicates sum of values in our data


# In[ ]:


print(x.shape)
print(y.shape)


# <a id="8"></a> <br>
# ### Train-Test Split Data

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# <a id="9"></a> <br>
# # Machine Learning Classification Models

# <a id="10"></a> <br>
# ### Logistic Regression Classification

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)

#Print Train Accuracy
lr_train_accuracy = lr_model.score(x_train,y_train)
print("lr_train_accuracy = ",lr_model.score(x_train,y_train))
#Print Test Accuracy
lr_test_accuracy = lr_model.score(x_test,y_test)
print("lr_test_accuracy = ",lr_model.score(x_test,y_test))


# In[ ]:


data = [go.Bar(
            x=["lr_train_accuracy","lr_test_accuracy"],
            y=[lr_train_accuracy,lr_test_accuracy],
            marker=dict(color='rgb(158,202,225)',
            line=dict(color='rgba(254, 69, 62, 1)',
            width=1.5),
        ),
    opacity=0.6
    )]

iplot(data, filename='text-hover-bar')


# In[ ]:


y_pred = lr_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm_lr, annot=True, linewidth=0.5, fmt=".0f",  cmap='RdPu', ax = ax)
plt.xlabel = ('y_pred')
plt.ylabel = ('y_true')
plt.show()


# In[ ]:


tp ,fp ,fn ,tn= cm_lr.ravel()
print("lr_RECALL = ",tp/(tp+fn))
print("lr_PRECISION = ",(tp/(tp+fp)))


# <a id="11"></a> <br>
# ## K-Nearest Neighbors (KNN) Classification

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train,y_train)

#Print Train Accuracy
knn_train_accuracy = knn_model.score(x_train,y_train)
print("knn_train_accuracy = ",knn_model.score(x_train,y_train))
#Print Test Accuracy
knn_test_accuracy = knn_model.score(x_test,y_test)
print("knn_test_accuracy = ",knn_model.score(x_test,y_test))


# <a id="12"></a> <br>
# ### Model Complexity

# <br> Now the question is why we choose K = 3 or what value we need to choose K. The answer is in model complexity
# 
# <br> **Model complexity:**
# * K has general name. It is called a hyperparameter. For now just know K is hyperparameter and we need to choose it that gives best performace. 
# * Literature says if k is small, model is complex model can lead to overfit. It means that model memorizes the train sets and cannot predict test set with good accuracy.
# * If k is big, model that is less complex model can lead to underfit. 
# * At below, I range K value from 1 to 30(exclude) and find accuracy for each K value. As you can see in plot, when K is 1 it memorize train sets and cannot give good accuracy on test set (overfit). Also if K is 20 or 22, model is lead to underfit. Again accuracy is not enough. However look at when K is 20 or 22(best performance), accuracy has highest value almost 82%. 

# In[ ]:


# Model complexity
neighboors = np.arange(1,30)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neighboors):
    # k from 1 to 30(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # fit with knn
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))           # train accuracy
    test_accuracy.append(knn.score(x_test, y_test))              # test accuracy

# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = neighboors,
                    y = train_accuracy,
                    mode = "lines",
                    name = "train_accuracy",
                    marker = dict(color = 'rgba(160, 112, 2, 0.8)'),
                    text= "train_accuracy")
# Creating trace2
trace2 = go.Scatter(
                    x = neighboors,
                    y = test_accuracy,
                    mode = "lines+markers",
                    name = "test_accuracy",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= "test_accuracy")
data = [trace1, trace2]
layout = dict(title = 'K Value vs Accuracy',
              xaxis= dict(title= 'Number of Neighboors',ticklen= 10,zeroline= True)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

knn_train_accuracy = np.max(train_accuracy)
knn_test_accuracy = np.max(test_accuracy)
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy))))


# In[ ]:


data = [go.Bar(
            x=["knn_train_accuracy","knn_test_accuracy"],
            y=[knn_train_accuracy,knn_test_accuracy],
            marker=dict(color='rgb(158,202,225)',
            line=dict(color='rgba(254, 69, 62, 1)',
            width=1.5),
        ),
    opacity=0.6
    )]

iplot(data, filename='text-hover-bar')


# In[ ]:


y_pred = knn_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_knn = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm_knn, annot=True, linewidth=0.5, fmt=".0f",  cmap='RdPu', ax = ax)
plt.xlabel = ("y_pred")
plt.ylabel = ("y_true")
plt.show()


# In[ ]:


tp ,fp ,fn ,tn= cm_knn.ravel()
print("knn_RECALL = ",tp/(tp+fn))
print("knn_PRECISION = ",(tp/(tp+fp)))


# <a id="13"></a> <br>
# ### Support Vector Machine(SVM) Classification

# In[ ]:


from sklearn.svm import SVC

svm_model = SVC(random_state=1)
svm_model.fit(x_train,y_train)

#Print Train Accuracy
svm_train_accuracy = svm_model.score(x_train,y_train)
print("svm_train_accuracy = ",svm_model.score(x_train,y_train))
#Print Test Accuracy
svm_test_accuracy = svm_model.score(x_test,y_test)
print("svmr_test_accuracy = ",svm_model.score(x_test,y_test))


# In[ ]:


data = [go.Bar(
            x=["svm_train_accuracy","svm_test_accuracy"],
            y=[svm_train_accuracy,svm_test_accuracy],
            marker=dict(color='rgb(158,202,225)',
            line=dict(color='rgba(254, 69, 62, 1)',
            width=1.5),
        ),
    opacity=0.6
    )]

iplot(data, filename='text-hover-bar')


# In[ ]:


y_pred = svm_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_svm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm_svm, annot=True, linewidth=0.5, fmt=".0f",  cmap='RdPu', ax = ax)
plt.xlabel = ("y_pred")
plt.ylabel = ("y_true")
plt.show()


# In[ ]:


tp ,fp ,fn ,tn= cm_svm.ravel()
print("svm_RECALL = ",tp/(tp+fn))
print("svm_PRECISION = ",(tp/(tp+fp)))


# <a id="14"></a> <br>
# ### Naive Bayes Classification

# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(x_train,y_train)

#Print Train Accuracy
nb_train_accuracy = nb_model.score(x_train,y_train)
print("nb_train_accuracy = ",nb_model.score(x_train,y_train))
#Print Test Accuracy
nb_test_accuracy = nb_model.score(x_test,y_test)
print("nb_test_accuracy = ",nb_model.score(x_test,y_test))


# In[ ]:


data = [go.Bar(
            x=["nb_train_accuracy","nb_test_accuracy"],
            y=[nb_train_accuracy,nb_test_accuracy],
            marker=dict(color='rgb(158,202,225)',
            line=dict(color='rgba(254, 69, 62, 1)',
            width=1.5),
        ),
    opacity=0.6
    )]

iplot(data, filename='text-hover-bar')


# In[ ]:


y_pred = nb_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_nb = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm_nb, annot=True, linewidth=0.5, fmt=".0f",  cmap='RdPu', ax = ax)
plt.xlabel = ("y_pred")
plt.ylabel = ("y_true")
plt.show()


# In[ ]:


tp ,fp ,fn ,tn= cm_nb.ravel()
print("nb_RECALL = ",tp/(tp+fn))
print("nb_PRECISION = ",(tp/(tp+fp)))


# <a id="15"></a> <br>
# ### Decision Tree Classification

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
#if you remove random_state=1, you can see how accuracy is changing
#Accuracy changing depends on splits
dt_model = DecisionTreeClassifier(random_state=1)
dt_model.fit(x_train,y_train)

#Print Train Accuracy
dt_train_accuracy = dt_model.score(x_train,y_train)
print("dt_train_accuracy = ",dt_model.score(x_train,y_train))
#Print Test Accuracy
dt_test_accuracy = dt_model.score(x_test,y_test)
print("dt_test_accuracy = ",dt_model.score(x_test,y_test))


# In[ ]:


data = [go.Bar(
            x=["dt_train_accuracy","dt_test_accuracy"],
            y=[dt_train_accuracy,dt_test_accuracy],
            marker=dict(color='rgb(158,202,225)',
            line=dict(color='rgba(254, 69, 62, 1)',
            width=1.5),
        ),
    opacity=0.6
    )]

iplot(data, filename='text-hover-bar')


# In[ ]:


y_pred = dt_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_dt = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm_dt, annot=True, linewidth=0.5, fmt=".0f",  cmap='RdPu', ax = ax)
plt.xlabel = ("y_pred")
plt.ylabel = ("y_true")
plt.show()


# In[ ]:


tp ,fp ,fn ,tn= cm_dt.ravel()
print("dt_RECALL = ",tp/(tp+fn))
print("dt_PRECISION = ",(tp/(tp+fp)))


# <a id="16"></a> <br>
# ### Random Forest Classification

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

#n_estimators = 100 => Indicates how many trees we have
rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
rf_model.fit(x_train,y_train)

#Print Train Accuracy
rf_train_accuracy = rf_model.score(x_train,y_train)
print("rf_train_accuracy = ",rf_model.score(x_train,y_train))
#Print Test Accuracy
rf_test_accuracy = rf_model.score(x_test,y_test)
print("rf_test_accuracy = ",rf_model.score(x_test,y_test))


# In[ ]:


data = [go.Bar(
            x=["rf_train_accuracy","rf_test_accuracy"],
            y=[rf_train_accuracy,rf_test_accuracy],
            marker=dict(color='rgb(158,202,225)',
            line=dict(color='rgba(254, 69, 62, 1)',
            width=1.5),
        ),
    opacity=0.6
    )]

iplot(data, filename='text-hover-bar')


# In[ ]:


y_pred = rf_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm_rf, annot=True, linewidth=0.5, fmt=".0f",  cmap='RdPu', ax = ax)
plt.xlabel = ("y_pred")
plt.ylabel = ("y_true")
plt.show()


# In[ ]:


tp ,fp ,fn ,tn= cm_rf.ravel()
print("rf_RECALL = ",tp/(tp+fn))
print("rf_PRECISION = ",(tp/(tp+fp)))


# <a id="17"></a> <br>
# # Evaluation Classification Models

# <a id="18"></a> <br>
# ### Confusion Matrixes Comparison

# In[ ]:


plt.figure(figsize=(20,10))
plt.suptitle("Confusion Matrixes of Classification Models",fontsize=30)

plt.subplot(2,3,1)
plt.title("Logistic Regression Classification")
sns.heatmap(cm_lr,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.subplot(2,3,2)
plt.title("Decision Tree Classification")
sns.heatmap(cm_knn,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.subplot(2,3,3)
plt.title("K Nearest Neighbors(KNN) Classification")
sns.heatmap(cm_svm,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.subplot(2,3,4)
plt.title("Naive Bayes Classification")
sns.heatmap(cm_nb,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.subplot(2,3,5)
plt.title("Random Forest Classification")
sns.heatmap(cm_dt,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.subplot(2,3,6)
plt.title("Support Vector Machine(SVM) Classification")
sns.heatmap(cm_rf,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.show()


# <a id="19"></a> <br>
# ### Bar Charts Comparison

# In[ ]:


# create trace1 
trace1 = go.Bar(
         x = np.array("Logistic Regression"),
         y = np.array(lr_test_accuracy),
         name = "Logistic Regression",
         marker = dict(color ='rgba(255, 77, 77, 1)',
         line=dict(color='rgb(0,0,0)',width=1.5))
                    )
# create trace2 
trace2 = go.Bar(
         x = np.array("KNN"),
         y = np.array(knn_test_accuracy),
         name = "KNN",
         marker = dict(color ='rgba(9, 220, 125, 1)',
         line=dict(color='rgb(0,0,0)',width=1.5))
                    )
# create trace3 
trace3 = go.Bar(
         x = np.array("SVM"),
         y = np.array(svm_test_accuracy),
         name = "SVM",
         marker = dict(color ='rgba(36, 44, 188, 1)',
         line=dict(color='rgb(0,0,0)',width=1.5))
                    )
# create trace4 
trace4 = go.Bar(
         x = np.array("Naive Bayes"),
         y = np.array(nb_test_accuracy),
         name = "Naive Bayes",
         marker = dict(color ='rgba(209, 0, 224, 1)',
         line=dict(color='rgb(0,0,0)',width=1.5))
                    )
# create trace5 
trace5 = go.Bar(
         x = np.array("Decision Tree"),
         y = np.array(dt_test_accuracy),
         name = "Decision Tree",
         marker = dict(color ='rgba(0, 224, 209, 1)',
         line=dict(color='rgb(0,0,0)',width=1.5))
                    )
# create trace6 
trace6 = go.Bar(
         x = np.array("Random Forest"),
         y = np.array(rf_test_accuracy),
         name = "Random Forest",
         marker = dict(color ='rgba(255, 255, 61, 1)',
         line=dict(color='rgb(0,0,0)',width=1.5))
                    )

data = [trace1,trace2,trace3,trace4,trace5,trace6]
layout = go.Layout(barmode = "group",title="Machine Learning Classification Models Comparison")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# **As you can see,**
# the best machine learning algorithm for our data is Random Forest Classification algorithm with 86%.

# <a id="20"></a> <br>
# # Conclusion
# **If you like it, Please upvote my kernel.**<br>
# **If you have any question, I will happy to hear it**
