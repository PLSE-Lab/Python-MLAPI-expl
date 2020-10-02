#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# Team members
#Varun M:01FB16ECS434
#Varshini:01FB16ECS433

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
file_path= "../input/Absenteeism_at_work.csv"
dataset=pd.read_csv(file_path)
dataset.head()


# In[ ]:


dataset.describe()


# In[ ]:


#Correlation matrix for the features    
corr = dataset.corr()
fig, ax = plt.subplots(figsize=(10, 10))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()


# In[ ]:


#the outliers are removed in the preprocessing stage
sns.boxplot(dataset['Absenteeism time in hours'])
median = np.median(dataset['Absenteeism time in hours'])
q75, q25 = np.percentile(dataset['Absenteeism time in hours'], [75 ,25])
interquartile = q75 - q25
print("Upper bound:",q75 + (1.5*interquartile))
print("Lower bound:",q25 - (1.5*interquartile))
#setting the lower and upper bounds for outliers
dataset= dataset[dataset['Absenteeism time in hours']<=17]
dataset= dataset[dataset['Absenteeism time in hours']>=-7]


# In[ ]:


X=dataset.drop(['Absenteeism time in hours'], axis=1)
y=dataset['Absenteeism time in hours']
from sklearn import preprocessing
x = X.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled,columns=list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


#performing scaling of the features
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 


# In[ ]:


#Model 1
#K-nearest-neighbour
error = []
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 


# In[ ]:



#Training and Predictions
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=30)  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)  


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix  
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[ ]:


#Model 2
#Support vector Machine(SVM)
from sklearn.svm import SVC  
svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix  
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))


# In[ ]:


#Model 3 
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier 

dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
y_pred = dtree_model.predict(X_test) 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))


# In[ ]:


#Conclusions
#The accuracy for Decision tree classifier is the highest amongst all other models.
#Also precision and F1 score of the Decision tree classifier is the highest.
#Hence the Best model for this dataset would be the decision tree classifier followed by support vector machine followed by k nearest neighbours.

