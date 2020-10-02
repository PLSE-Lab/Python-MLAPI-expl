#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'


# In[ ]:


data= pd.read_csv('../input/mtcars.csv')


# In[ ]:


data

mpg 	Miles/(US) gallon
cyl 	Number of cylinders
disp	Displacement (cu.in.)
hp	    Gross horsepower
drat	Rear axle ratio
wt	    Weight (1000 lbs)
qsec	1/4 mile time
vs	    Engine (0 = V-shaped, 1 = straight)
am	    Transmission (0 = automatic, 1 = manual)
gear	Number of forward gears
carb	Number of carburetors
# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.dtypes


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum()
#No missing values


# In[ ]:


#univariate Analysis
data.hist(grid=False, figsize=(20,10), color='pink')


# In[ ]:


#boxplot
for a in data:
    if (a=='model' or a=='vs' or a=='am'):
        continue
    else:
        plt.figure()
        data.boxplot(column=[a], grid=False)
        


# In[ ]:


data.head()


# In[ ]:


#count plot for vs
data['vs'].value_counts()
sns.countplot(data['vs'])


# In[ ]:


#count plot for vs
data['am'].value_counts()
sns.countplot(data['am'])


# In[ ]:


#count plot for vs
data['gear'].value_counts()
sns.countplot(data['gear'])


# In[ ]:


#count plot for vs
data['carb'].value_counts()
sns.countplot(data['carb'])


# In[ ]:


#count plot for cyl
data['cyl'].value_counts()
sns.countplot(data['cyl'])


# In[ ]:


#Bivariate analysis
data.corr()


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), square=True, linewidths=0.2)
plt.xticks(rotation=90)
plt.yticks(rotation=0)


# In[ ]:


plt.figure(figsize=(20,10))
sns.pairplot(data, diag_kind='kde')


# In[ ]:


#let try to use Label Encoder first
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data['Class']= le.fit_transform(data['model'])


# In[ ]:


data.head()


# In[ ]:


X1= data.iloc[:,1:12]
Y1= data.iloc[:,-1]


# In[ ]:


#lets try to plot Decision tree to find the feature importance
from sklearn.tree import DecisionTreeClassifier
tree= DecisionTreeClassifier(criterion='entropy', random_state=1)
tree.fit(X1, Y1)


# In[ ]:


imp= pd.DataFrame(index=X1.columns, data=tree.feature_importances_, columns=['Imp'] )
imp.sort_values(by='Imp', ascending=False)


# In[ ]:


sns.barplot(x=imp.index.tolist(), y=imp.values.ravel(), palette='coolwarm')

#taking only two variable #disp and #qsec as these variable has high importance


# In[ ]:


X=data[['disp','qsec']]
Y= data.iloc[:,0]


# In[ ]:


#lets try to create segments using K means clustering
from sklearn.cluster import KMeans
#using elbow method to find no of clusters
wcss=[]
for i in range(1,7):
    kmeans= KMeans(n_clusters=i, init='k-means++', random_state=1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[ ]:


print(wcss)


# In[ ]:


plt.plot(range(1,7), wcss, linestyle='--', marker='o', label='WCSS value')
plt.title('WCSS value- Elbow method')
plt.xlabel('no of clusters- K value')
plt.ylabel('Wcss value')
plt.legend()
plt.show()


# In[ ]:


#Here we got no of clusters = 2 
kmeans= KMeans(n_clusters=2, random_state=1)
kmeans.fit(X)


# In[ ]:


kmeans.predict(X)


# In[ ]:


#Cluster Center
kmeans.cluster_centers_


# In[ ]:


data['cluster']=kmeans.predict(X)
data.sort_values(by='cluster').head()


# In[ ]:


#plotting Cluster plot

plt.scatter(data.loc[data['cluster']==0]['disp'], data.loc[data['cluster']==0]['qsec'], c='green', label='cluster1-0')
plt.scatter(data.loc[data['cluster']==1]['disp'], data.loc[data['cluster']==1]['qsec'], c='red', label='cluster2-1')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='center')
plt.xlabel('disp')
plt.ylabel('qsec')
plt.legend()
plt.show()

