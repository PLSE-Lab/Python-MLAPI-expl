#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.columns=["pregnancies","glucose","blood_pressure","skin_thickness","insulin","bmi","diabetes_predigree_function","age","class"]


# In[ ]:


data.isnull().sum()


# In[ ]:


data = data.sort_values(by=["age","glucose"], ascending=False)
data['rank']=tuple(zip(data.age,data.glucose))
data['rank']=data.groupby('age',sort=False)['rank'].apply(lambda x : pd.Series(pd.factorize(x)[0])).values
data.head()


# In[ ]:


data.reset_index(inplace=True,drop=True)
data.head()


# In[ ]:


data['age']=data['age']
bins=[21,37,51,65,81]
labels=['Young Adult','Early Adult','Adult','Senior']
data['age_grp']=pd.cut(data['age'],bins,labels=labels)


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidths=5,fmt='.0%',ax=ax)
plt.show()


# In[ ]:


sns.pairplot(data)


# In[ ]:


fig = plt.figure(figsize = (10,10))
ax = fig.gca()
data.hist(ax = ax)
plt.show()


# In[ ]:


fig=plt.figure(figsize=(20,10))
sns.boxplot(data = data,notch = True,linewidth = 2.5, width = 0.50)
plt.show()


# In[ ]:


fig,axes=plt.subplots(nrows=2,ncols=1)

data.plot(kind='hist',y='age',bins=50,range=(0,100),density=True,ax=axes[0])
data.plot(kind='hist',y='age',bins=50,range=(0,100),density=True,ax=axes[1],cumulative=True)
plt.show()


# In[ ]:


labels = data.age_grp.value_counts().index
colors = ['green','yellow','orange','red']
explode = [0,0,0,0]
sizes = data.age_grp.value_counts().values

plt.figure(0,figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Target According to Age Group',color = 'blue',fontsize = 15)
plt.show()


# In[ ]:


plt.figure(figsize=(18,5))
sns.barplot(data.age_grp,data.glucose)
plt.show()


# In[ ]:


#age_grp - blood_pressure
result = data.groupby(["age_grp"])['blood_pressure'].aggregate(np.median).reset_index().sort_values('blood_pressure')
sns.barplot(x='age_grp', y="blood_pressure", data=data, order=result['age_grp']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[ ]:


#age_grp - skin_thickness
result = data.groupby(["age_grp"])['skin_thickness'].aggregate(np.median).reset_index().sort_values('skin_thickness')
sns.barplot(x='age_grp', y="skin_thickness", data=data, order=result['age_grp'])
plt.show()


# In[ ]:


#age_grp - insulin
result = data.groupby(["age_grp"])['insulin'].aggregate(np.median).reset_index().sort_values('insulin')
sns.barplot(x='age_grp', y="insulin", data=data, order=result['age_grp'])
plt.show()


# In[ ]:


#age_grp - bmi
result = data.groupby(["age_grp"])['bmi'].aggregate(np.median).reset_index().sort_values('bmi')
sns.barplot(x='age_grp', y="bmi", data=data, order=result['age_grp'])
plt.show()


# In[ ]:


#age_grp - diabetes_predigree_function
result = data.groupby(["age_grp"])['diabetes_predigree_function'].aggregate(np.median).reset_index().sort_values('diabetes_predigree_function')
sns.barplot(x='age_grp', y="diabetes_predigree_function", data=data, order=result['age_grp'])
plt.show()


# In[ ]:


#age_grp - diabetes_predigree_function
result = data.groupby(["age_grp"])['class'].aggregate(np.median).reset_index().sort_values('class')
sns.barplot(x='age_grp', y="class", data=data, order=result['age_grp'])
plt.show()


# In[ ]:


sns.barplot(x='pregnancies',y='age',data=data)
plt.show()


# In[ ]:


grp =data.groupby("age")
x= grp["glucose"].agg(np.mean)
y=grp["insulin"].agg(np.mean)
z=grp["blood_pressure"].agg(np.mean)


# In[ ]:


plt.figure(figsize=(16,5))
plt.plot(x,'ro',color='r')
plt.xticks(rotation=90)
plt.title('age wise glucose')
plt.xlabel('age')
plt.ylabel('glucose')
plt.show()


# In[ ]:


plt.figure(figsize=(16,5))
plt.plot(y,'r--',color='b')
plt.xticks(rotation=90)
plt.title('Age wise insulin')
plt.xlabel('age')
plt.ylabel('insulin')
plt.show()


# In[ ]:


plt.figure(figsize=(16,5))
plt.plot(z,"g^",color='g')
plt.xticks(rotation=90)
plt.xlabel('age')
plt.ylabel('blood_pressure')
plt.show()


# In[ ]:


ax=data.glucose.plot.kde()
ax=data.insulin.plot.kde()
ax=data.blood_pressure.plot.kde()
plt.legend()
plt.show()


# In[ ]:


f,ax1 = plt.subplots(figsize =(20,5))
sns.pointplot(x='age',y='glucose',data=data,color='lime',alpha=0.8)
sns.pointplot(x='age',y='blood_pressure',data=data,color='red',alpha=0.8)
sns.pointplot(x='age',y='insulin',data=data,color='blue',alpha=0.8)
plt.text(5,1,'age-glucose',color='lime',fontsize = 12,style = 'italic')
plt.text(12,1,'age-blood_pressure',color='red',fontsize = 12,style = 'italic')
plt.text(35,1,'age-insulin',color='blue',fontsize = 12,style = 'italic')
plt.xlabel('age',fontsize = 15,color='blue')
plt.ylabel('values',fontsize = 15,color='blue')
plt.title('glucose  -  blood_pressure - insulin',fontsize = 20,color='blue')

plt.grid()


# <h1>Prediction</h1>

# In[ ]:



plt.scatter(data['age'],data['glucose'])
plt.xlabel('age')
plt.ylabel('glucose')
plt.show()


# In[ ]:


# KMeans Clustering
data2 = data.loc[:,['age','glucose']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2)
labels = kmeans.predict(data2)
plt.scatter(data['age'],data['glucose'],c = labels)
plt.xlabel('age')
plt.xlabel('glucose')
plt.show()


# In[ ]:


# cross tabulation table
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# In[ ]:


# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()


# In[ ]:


data3 = data.drop('class',axis = 1)
data3 = pd.get_dummies(data3)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# In[ ]:


from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data3.iloc[200:220,:],method = 'single')
dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)
plt.show()


# In[ ]:


color_list = ['red' if i==1 else 'green' for i in data.loc[:,'class']]


# In[ ]:


from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(data2)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list )
plt.xlabel('age')
plt.xlabel('glucose')
plt.show()


# In[ ]:


# PCA
from sklearn.decomposition import PCA
model = PCA()
model.fit(data3)
transformed = model.transform(data3)
print('Principle components: ',model.components_)


# In[ ]:


# PCA variance
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler,pca)
pipeline.fit(data3)

plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()


# In[ ]:


# apply PCA
pca = PCA(n_components = 2)
pca.fit(data3)
transformed = pca.transform(data3)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list)
plt.show()


# references: https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners
