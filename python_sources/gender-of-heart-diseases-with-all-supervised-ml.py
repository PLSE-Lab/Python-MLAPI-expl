#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
#    Humanity has to struggle with diseasters since humanity exists. Eventhough some disaster has been popular from time to time, crucial illnesses like heart diseases became much more popular in our days. 
# In this kernel, what I will make is to analyze a data about how genders are effected from 'Heart Disease'. 
# Also, I will use Supervised Machine Learning techniques to predict datas in order to detect whether the gender is male, or female according to data.
# 
# Content:
# 1. [Logistic Regression](#1)
# 1. [K-Nearest Neighbour(KNN)](#2)
# 1. [Support Vector Machine(SVM)](#3)
# 1. [Naive Bayes Classification](#4)
# 1. [Decision Tree Classification](#5)
# 1. [Random Forest Classification](#6)
# 1. [Comparison using Plots](#7)
# 1. [Conclusion](#8)

# I will use **sklearn** for implementation. As a conclusion, I will decide which way is the best for Supervised Learning.

# In[ ]:


# import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Firstly, import data
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")


# In[ ]:


# First look to data
data.head()


# 1 for males, 0 for females according to data. 

# In[ ]:


data.info()


# Actually, if the data includes sexes as string(means as Male, Female), we had to make them int. The data gave us 1 for males, 0 for females, it can be seen on above, as well, so, there is no need to change something and we can keep doing our analysis.

# ***Normalization***
# 
# We made normalization process in order to observe all datas clearly, which means we may encounter with some problems when we see the data on charts because we have discrete values such as 233, 2.5, 130 etc.
# How normalization helps us to see datas better is based on that normalization process takes the all values between 0-1, so we can see the vales easily.

# In[ ]:


# Adjust x and y for making normalization
y = data.sex.values # make it np array
x_data = data.drop(["sex"], axis = 1) # everything, exludes sex, is on the x

# Normalization 
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values 


# ***Train Test Split***
# 
# Now, I will make a split the data for decision of how much of them will be train and how much of them will be test. 
# I will split 20% of the data for test, 80% of the data for train.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
x_train.shape


# <a id="1"></a> <br>
# # Logistic Regression
# 
# Before I countinue, I have to make a trick for using my values. As it can be seen above, shape of x_train (242, 13), but this is wrong for linear regression. Our data has to be array[number of the feature, number of the sample], so I need to make transposition them in order to be able to use.

# In[ ]:


x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


# In[ ]:


best_accuracy = [] # in order to compare all results, I will create an empty list that will be filled after all tests.

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))

best_accuracy.append(lr.score(x_test.T,y_test.T))


# It did not satisfy me, but it is early to say something. Let's continue with our another learning method: 

# <a id="2"></a> <br>
# # K-Nearest Neighbour(KNN)

# Transposition will be wrong for KNN, so make their shapes' how they were before.

# In[ ]:


x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
score_list = [] # to keep all scores for plot
best_score = 0
best_k = 0
for each in range(1,15): # I will try for 15 values of k
    knn = KNeighborsClassifier(n_neighbors = each)
    knn.fit(x_train,y_train) # train the model
    score_list.append(knn.score(x_test,y_test))
    if (knn.score(x_test,y_test) > best_score): # if you find a value that bigger than before, keep it!
       best_score = knn.score(x_test,y_test)
       best_k = each
    
plt.plot(range(1,15), score_list) # x_axis=range(1,15), y_axis=score_list
plt.xlabel("k values")
plt.ylabel("Accuracy")
plt.show()

print("The best accuracy we got is ", best_score)
print("Best accuracy's k value is ", best_k)

best_accuracy.append(best_score)


# <a id="3"></a> <br>
# # Support Vector Machine(SVM)

# In[ ]:


from sklearn.svm import SVC
svm = SVC(random_state = 42)
svm.fit(x_train,y_train)
print("Accuracy of SVM: ",svm.score(x_test,y_test))

best_accuracy.append(svm.score(x_test,y_test))


# <a id="4"></a> <br>
# # Naive Bayes Classification

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("Accuracy of NB: ", nb.score(x_test,y_test))

best_accuracy.append(nb.score(x_test,y_test))


# <a id="5"></a> <br>
# # Decision Tree Classification

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Accuracy of Decision Tree: ", dt.score(x_test,y_test))

best_accuracy.append(dt.score(x_test,y_test))


# <a id="6"></a> <br>
# # Random Forest

# Random Forest is like decision tree. You can think that we are creating a forest using trees.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state=1) 
#n_estimator = we determined of how many trees will be on our forest
rf.fit(x_train,y_train)
print("Accuracy of Random Forest: ", rf.score(x_test,y_test)) # It gives us a bit better results than decision tree.

best_accuracy.append(rf.score(x_test,y_test))


# Our learning tests have been done. As you can see, I collected all the accuracy result's on a list. Firstly, let's check the list and then, we will decide which supervised learning type gave the best result.

# In[ ]:


best_accuracy


# The best way for comparation is to visualize this value, obviously. Let's make some plots before make a decision.

# <a id="7"></a> <br>
# # Comparison Methods

# I will use Seaborn and Plotly libraries. I tried to choose the best plots in order to see the differences easily.

# In[ ]:


# Bar Plot with Seaborn
sv_ml = ["Logistic Regression", "KNN", "SVM","Naive Bayes", "Decision Tree", "Random Forest"]

plt.figure(figsize=(15,10))
sns.barplot(x = sv_ml, y = best_accuracy)
plt.xticks(rotation= 30)
plt.xlabel('Accuracy')
plt.ylabel('Supervised Learning Types')
plt.title('Supervised Learning Types v Accuracy')
plt.show()


# In[ ]:


# Pie Chart with Seaborn
colors = ['red','green','blue','cyan','purple','yellow']
labels = sv_ml
explode = [0,0,0,0,0,0]
sizes = best_accuracy

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, labels=labels, explode=explode, colors=colors, autopct='%1.1f%%')
plt.title('Comparison of Accuracies',color = 'brown',fontsize = 15)
plt.show()


# I like interactive plots, I want to 2 more plot with Plotly, as addition to Seaborn.

# In[ ]:


# I think, Bar Plot with Plotly will be much useful than Seaborn's Bar Plot, because values are too close to read with Seaborn.
import plotly.graph_objs as go
import plotly.io as pio
# Create trace
trace1 = go.Bar(
                x = sv_ml,
                y = best_accuracy,
                name = "Accuracy Plot",
                marker = dict(color = 'rgba(10, 100, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5))
)
data = [trace1]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
pio.show(fig)


# In[ ]:


# go has been defined above
trace1 = go.Scatter(
                    x = sv_ml,
                    y = best_accuracy,
                    mode = "lines",
                    name = "Accuracy",
                    marker = dict(color = 'rgba(255, 0, 0, 0.7)')
)
# trace2 for finding the top points easily with attention getting colur
trace2 =go.Scatter( 
                    x = sv_ml,
                    y = best_accuracy,
                    mode = "markers",
                    name = "Highlight Point",
                    marker = dict(color = 'rgba(0, 255, 155, 1)')
)

data = [trace1,trace2]
layout = dict(title = 'Accuracies of Supervised Learning Types',
              xaxis= dict(title= 'Accuracy',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Types',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
pio.show(fig)


# In[ ]:


# It's just a bonus, not for an analysis, just because it is one of the my favorite plots :)
# Word Cloud
from wordcloud import WordCloud

sv_ml2 = ["Logistic Regression", "KNN", "SVM","Naive Bayes", "Decision Tree", "Random Forest"]
list_label = ["Type","Accuracy"]
list_col = [sv_ml2,best_accuracy]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped) 
df = pd.DataFrame(data_dict)

cloud = df.Type

plt.subplots(figsize=(10,10))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(cloud))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()


# <a id="8"></a> <br>
# # Conclusion
# After making bunch of visualization, it can be said KNN Algorithm is the best way to prediction for our data. Eventhough the results are a bit less than what I expected, KNN has became much more better than others. 
# 
# You can see that **sklearn** provides us too much convenience, and I tried to show their implementatin as much as I can. Please vote and comment if you enjoyed my kernel, thanks for reading. 
