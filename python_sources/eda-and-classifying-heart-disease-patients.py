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


df=pd.read_csv('../input/heart.csv')


# In[ ]:


df.head()


# **The dataset contains the following features:**
# 1. age(in years)
# 2. sex: (1 = male; 0 = female)
# 3. cp: chest pain type
# 4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# 5. chol: serum cholestoral in mg/dl
# 6. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. restecg: resting electrocardiographic results
# 8. thalach: maximum heart rate achieved
# 9. exang: exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak: ST depression induced by exercise relative to rest
# 11. slope: the slope of the peak exercise ST segment
# 12. ca: number of major vessels (0-3) colored by flourosopy
# 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. target: 1 or 0 

# In[ ]:


df.info()


# **No. of Rows**

# In[ ]:


df.shape[0]


# **No. of columns**

# In[ ]:


df.shape[1]


# **Checking null value**

# In[ ]:


Null=df.isnull()
Null.sum()


# There is *no* null value in the dataset

# **What is the mean,std of the dataset**
# 
# The features described in the below data set are:
# 
# 1. Count tells us the number of NoN-empty rows in a feature.
# 
# 2. Mean tells us the mean value of that feature.
# 
# 3. Std tells us the Standard Deviation Value of that feature.
# 
# 4. Min tells us the minimum value of that feature.
# 
# 5. 25%, 50%, and 75% are the percentile/quartile of each features.
# 
# 6. Max tells us the maximum value of that feature.
# 
# 

# In[ ]:


df.describe()


# **Checking features of various attributes**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
male =len(df[df['sex'] == 1])
female = len(df[df['sex']== 0])
total=len(df.sex)
total_male_percent=(male/total)*100
total_female_percent=(female/total)*100

#plot
labels = ['male', 'female']
values = [total_male_percent, total_female_percent]

plt.figure(figsize=(8,6))
plt.title('Sex Percentage')
plt.xlabel('sex')
plt.ylabel('percentage')
plt.bar(labels, values,color=('r','b'))
plt.show()


# **Chest pain**

# In[ ]:


plt.figure(figsize=(8,6))

# Data to plot
labels = 'Chest Pain Type:0','Chest Pain Type:1','Chest Pain Type:2','Chest Pain Type:3'
values = [len(df[df['cp'] == 0]),len(df[df['cp'] == 1]),
         len(df[df['cp'] == 2]),
         len(df[df['cp'] == 3])]
colors = ['blue', 'green','orange','red']
 
# Plot
plt.title('Chest pain')
plt.xlabel('Types')
plt.ylabel('values')

plt.bar(labels,values, color=colors) 
plt.show()


# **3. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)**

# In[ ]:


plt.figure(figsize=(8,6))

# Data to plot
labels = 'fasting blood sugar < 120 mg/dl','fasting blood sugar > 120 mg/dl'
sizes = [len(df[df['fbs'] == 0]),len(df[df['cp'] == 1])]
colors = ['skyblue', 'yellowgreen','orange','gold']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
 
plt.axis('equal')
plt.show()


# **4.exang: exercise induced angina (1 = yes; 0 = no)**

# In[ ]:


plt.figure(figsize=(8,6))

# Data to plot
labels = 'No','Yes'
sizes = [len(df[df['exang'] == 0]),len(df[df['exang'] == 1])]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ** Number of people who have heart disease according to age **

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = df, hue = 'target',palette='GnBu')
plt.show()


# **Scatterplot for thalach vs. trestbps **

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df,hue='target')
plt.show()


# **Making Predictions**
# 
# 
# **Splitting the dataset into training and test set**

# In[ ]:


print(df.dtypes)


# In[ ]:


X= df.drop('target',axis=1)
y=df['target']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


# **Xgboost**

# In[ ]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
dpen = []
for i in range(5,11):
    model = XGBClassifier(max_depth = i)
    model.fit(X_train,y_train)
    target = model.predict(X_test)
    dpen.append(accuracy_score(y_test, target))
    print("accuracy : ",dpen[i-5])
print("Best accuracy: ",max(dpen))


# **KNN Algorithm **

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)  # n_neighbors means k
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)

print("{} NN Score: {:.2f}%".format(7, knn.score(X_test, y_test)*100))


# In[ ]:


# try ro find best k value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(X_train, y_train)
    scoreList.append(knn2.score(X_test, y_test))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()


print("Maximum KNN Score is {:.2f}%".format((max(scoreList))*100))


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(X_train, y_train)
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(X_test,y_test)*100))


# **SVM**

# In[ ]:


from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(X_train, y_train)
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(svm.score(X_test,y_test)*100))


# **naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
print("Accuracy of Naive Bayes: {:.2f}%".format(nb.score(X_test,y_test)*100))

