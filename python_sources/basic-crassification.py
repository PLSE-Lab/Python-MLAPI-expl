#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/heart.csv')
df.head(3)


# **Dimensions of the dataset**

# In[ ]:


print('Number of rows in the dataset: ',df.shape[0])
print('Number of columns in the dataset: ',df.shape[1])


# **Checking for null values in the dataset**

# In[ ]:


df.isnull().sum()


# There are no null values in the dataset

# In[ ]:


df.describe()


# ### **Checking features of various attributes**

# In[ ]:


male =len(df[df['sex'] == 1])
female = len(df[df['sex']== 0])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'Male','Female'
sizes = [male,female]
colors = ['skyblue', 'yellowgreen']
explode = (0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# #### **3. fbs: (fasting blood sugar > 110 mg/dl) (1 = true; 0 = false)**
# because it is used threshold value 110mg/dl

# In[ ]:


plt.figure(figsize=(8,6))

# Data to plot
labels = 'fasting blood sugar < 110 mg/dl','fasting blood sugar > 110 mg/dl'
sizes = [len(df[df['fbs'] == 0]),len(df[df['cp'] == 1])]
colors = ['skyblue', 'yellowgreen','orange','gold']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
 
plt.axis('equal')
plt.show()


# #### **4.exang: exercise induced angina (1 = yes; 0 = no)**

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


# ### **Exploratory Data Analysis**

# In[ ]:


sns.set_style('whitegrid')


# #### **1. Heatmap**

# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(df.corr(), annot = True, cmap='coolwarm',linewidths=.1)
plt.show()


# #### **Plotting the distribution of various attribures**

# #### **1. thalach: maximum heart rate achieved**

# In[ ]:


sns.distplot(df['thalach'],kde=False,bins=30)


# #### **2.chol: serum cholestoral in mg/dl **

# In[ ]:


sns.distplot(df['chol'],kde=False,bins=30)
plt.show()


# #### **3. trestbps: resting blood pressure (in mm Hg on admission to the hospital)**

# In[ ]:


sns.distplot(df['trestbps'],kde=False,bins=30,color='blue')
plt.show()


# #### **4. Number of people who have heart disease according to age **

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = df, hue = 'target',palette='GnBu')
plt.show()


# #### **5.Scatterplot for thalach vs. chol **

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='chol',y='thalach',data=df,hue='target')
plt.show()


# #### **6.Scatterplot for thalach vs. trestbps **

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df,hue='target')
plt.show()


# ### **Making Predictions**

# **Splitting the dataset into training and test set**

# In[ ]:


X= df.drop('target',axis=1)
y=df['target']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)


# **Preprocessing - Scaling the features**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


# **Implementing GridSearchCv to select best parameters and applying k-NN Algorithm**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knn =KNeighborsClassifier()
params = {'n_neighbors':[i for i in range(1,33,2)]}


# In[ ]:


model = GridSearchCV(knn,params,cv=10)


# In[ ]:


model.fit(X_train,y_train)
model.best_params_           #print's parameters best values


# **Making predictions**

# In[ ]:


predict = model.predict(X_test)


# **Checking accuracy**

# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix
print('Accuracy Score: ',accuracy_score(y_test,predict))
print('Using k-NN we get an accuracy score of: ',
      round(accuracy_score(y_test,predict),5)*100,'%')


# **Confusion Matrix**

# In[ ]:



cnf_matrix = confusion_matrix(y_test,predict)
cnf_matrix


# In[ ]:


class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

#create a heat map
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',
           fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:





# In[ ]:




