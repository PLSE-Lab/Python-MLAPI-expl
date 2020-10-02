#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/heart.csv")
data.info()


# In[ ]:


df = pd.DataFrame(data)
df.columns


# In[ ]:


df.index


# In[ ]:


data.describe()


# In[ ]:


import seaborn as sns # used for plot interactive graph.
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
sns.countplot(data['age'],label ="count")


# In[ ]:


#correlation graph to find the correlation
corr = data.corr()
plt.figure(figsize = (14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'coolwarm')


# In[ ]:


#conclusion, none of the features are highlt correltaed, so we could use all 


# In[ ]:


selected_features = ['age', 'cp', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']


plt.figure(figsize = (14,14))
color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B
colors = data["target"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column
pd.scatter_matrix(data[selected_features], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix


# In[ ]:


#First get an general overview
data.hist(figsize=(15,20))
plt.figure()


# In[ ]:


#cp: chest pain type
data["cp"].value_counts()


# In[ ]:


plt.xlabel('pain type')
plt.ylabel('count')
plt.title('Four type chest pain')
sns.countplot(data['cp'])


# In[ ]:


import plotly
import plotly.graph_objs as go

x_data = data['age']
y_data = data['thalach']
colors = np.random.rand(2938)
sz = np.random.rand(2000)*30

fig = go.Figure()
fig.add_scatter(x = x_data,
                y = y_data,
                mode = 'markers',
                marker = {'size': sz,
                         'color': colors,
                         'opacity': 0.6,
                         'colorscale': 'Portland'
                       })
plotly.offline.iplot(fig)


# In[ ]:


#adding filter to data
data[(data['thalach']>190)]


# In[ ]:


#adding filter to data

data[(data['age']>40) & (data['sex']==0)]
   


# In[ ]:


#using swarmplot to learn about age, gender ad orobability of having heart disearse.
#first append a new row about age measuremnet based on probability
threshold = sum(data.age)/len(data.age)
threshold_chol = sum(data.chol)/len(data.chol)
print("threshold of age:" , threshold)
print("threshold of chol: ", threshold_chol)

data['probability'] = ['high' if i> threshold else 'low' for i in data.age]
data['probability']

sns.swarmplot(x = 'sex', y = 'age', hue = "probability", data = data)


# In[ ]:


data_original = pd.read_csv("../input/heart.csv")
data['sex'] = data_original.sex
data['sex']=['female' if i == 0 else 'male' for i in data.sex]
plt.figure(figsize=(14,8))
sns.swarmplot(x = 'age', y = 'chol', hue = "sex", data = data)


# In[ ]:


data.head(5)


# In[ ]:


#aplit the dataset ready for training and testing, and apply machine leaning ALG
from sklearn.model_selection import train_test_split
x,y = data.loc[:,data.columns != 'target'], data.loc[:,'target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)


# In[ ]:


data_orginal = pd.read_csv("../input/heart.csv")
data['sex']=data_orginal.sex
data.dtypes.sample(10)
#data.select_dtypes(exclude=['object'])


# In[ ]:


#Machine learning by KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 3)
#x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
x_train_one_hot_encoded = pd.get_dummies(x_train)
x_test_one_hot_encoded = pd.get_dummies(x_test)
knn.fit(x_train_one_hot_encoded,y_train)
prediction = knn.predict(x_test_one_hot_encoded)
acc = accuracy_score(y_test, prediction)
k = knn.score(x_test_one_hot_encoded,y_test)
print("acc score with one hot encoded: ", acc)
print("knn score: ", k)
print("just only drop the non-numrical feature--->")
x_train_2 = x_train.select_dtypes(exclude=['object'])
x_test_2 = x_test.select_dtypes(exclude=['object'])
knn.fit(x_train_2,y_train)
prediction = knn.predict(x_test_2)
acc = accuracy_score(y_test, prediction)
print("acc score without one hot encoded: ", acc)


# In[ ]:


# the training score is relativly low. I will apply Grid Search to find the best parameters of KNN


# In[ ]:


x_train = x_train.select_dtypes(exclude=['object'])
x_test = x_test.select_dtypes(exclude=['object'])


# In[ ]:


# grid search cross validation with 1 hyperparameter
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,100)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(x_train,y_train)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))


# In[ ]:


#After apply GridSearch the score is not much impoved


# In[ ]:


#apply sklearn decision tree classifier on the dataset
#reference https://www.kaggle.com/drgilermo/playing-with-the-knobs-of-sklearn-decision-tree
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from subprocess import check_output
from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re

classifier = DecisionTreeClassifier(max_depth = 3)
classifier.fit(x_train, y_train)

print("Decision tree score : {}".format(classifier.score(x_test, y_test))) 

clf = DecisionTreeClassifier(max_depth = 3, criterion = "entropy")
clf.fit(x_train,y_train)
print("Decision tree score : {}".format(clf.score(x_test, y_test))) 


# In[ ]:


#apply best split and random split
t = time.time()
clf_random = DecisionTreeClassifier(max_depth = 3, splitter = 'random')
clf_random.fit(x_train,y_train)
print('random Split accuracy...',clf_random.score(x_test,y_test))
clf_best = DecisionTreeClassifier(max_depth = 3, splitter = 'best')
clf_best.fit(x_train,y_train)
print('best Split accuracy...',clf_best.score(x_test,y_test))
# conclusion: random split is not nessesary worse than best split


# In[ ]:


with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf_best,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,# true will show thw im-purity of each node~ gini value
                              feature_names = x_test.columns.values,
                             class_names = ['No', 'Yes'],
                            #  class_names = True,
                              rounded = True,
                              filled= True )#False no color indication
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")
# we have generared rhe random decision tree classifier, see below.


# Conculsion, when using the best decision tree classifier, we have reached the best score around 70%. We could make  the following conclustions from the best_classifier decision tree:
# 1. When a patient has typical anagina (cp type 1), but there no obvious result of anagina pain during excercise, then the patient have a high chance NO heart disease. (target as NO)
# 2. When a patient shows not -typical anagina, with a low ST depression induced by exercise (oldpeak <= 2.1), the patinet is probabaly HAS heart disease. (target as YES)
# 3. When a patient shows typical anagina (cp type 1), no obvious result of anagina pain during excercise, but shows a low number of major vessels (0-3) colored by flourosopy  (ca <= 0.5), then this patient has more than 50% HAVE heart disease. (target as YES)
# 
# Besides, I have tried to generate decison tree by other data mining tool, such as RapidMiner, two of the results are shown below:
# 

# 
# ![](C:/Users/s134225/Downloads/wetransfer-2abfb3/tree.JPG)

# ![](https://raw.githubusercontent.com/xuanxuanzhang/image/master/tree.JPG)
# 

#  Doc reference: https://docs.rapidminer.com/latest/studio/operators/modeling/predictive/trees/parallel_decision_tree.html
# This decison tree is generated spilting attribute selection on:gain_ratio, and the maximal depth set to 4. 
# From the 1st tree of RapidMiner, we could conclude the following:
# 1. When a patient has non-typical anagina (cp type 2-4 , cp>0.5), and when resting with blood pressure (trestbps) higher than 179, ST depression(oldoeak) smaller than 2.45, have a high chance diagnose as heart disease. (target as YES)
# 2. When a patient has typical anagina (cp type 1, cp< 0.9), alone with the maximum heart rate (thalach) reached over 181.50, then the patient as a high chance have heart disease. (target as YES)
# 
# 

# ![](https://raw.githubusercontent.com/xuanxuanzhang/image/master/tree2.JPG)

# When I have adjusted the settings with set the spliting attribute on the least entropy one (information_gain), maximum depth kept as 4, then we get the second decison tree, which age, sex ,ca, and thal involved for prediction. Then we could get the following conclusion:
# 1. When a patient has non-typical anagina (cp type 2-4 , cp>0.5), but he/she is younger than 56, then he/she has a high chance to diagnose as heart disease.(target as YES)
# 2. When a female patient has non-typical anagina (cp type 2-4 , cp>0.5) and older than 56, then she has a high chance to diagnose as heart disease.(target as YES)
# 3. When a patient has a typical anagina (cp type 1, cp<= 0.5),  and he/she has number of major vessels bigger than 0.5 (ca> 0.5, maority patient has ca smaller than 0.4), then he/she probablty has NO heart disease. (target as NO)
# 

# 
# 

# 
