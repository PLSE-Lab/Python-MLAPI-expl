#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("../input/dataset1/dataset.csv")


# **Overview of the Data**

# Describes the number of attributes 

# In[ ]:


data.count()


# **Deleting the missing Values as the data is non repetitive and the missing values do not follow any trend**

# In[ ]:


data=data[data.M_weekly!='Na']
data=data[data.F_weekly!='Na']
data=data[data.All_workers!=0]


# **Description of the Data**

# In[ ]:


data.describe()


# **Changing the Data type of Srting into Integer**

# In[ ]:


occupation=data.iloc[1:,0].values
male_salary=data.iloc[1:,4].values
female_salary=data.iloc[1:,6].values
all_workers=data.iloc[1:,1].values
male_workers=data.iloc[1:,3].values
female_workers= data.iloc[1:,5].values
all_salary=data.iloc[1:,2].values
male_salary=pd.to_numeric(male_salary,downcast='integer')
female_salary=pd.to_numeric(female_salary,downcast='integer')
all_workers=pd.to_numeric(all_workers,downcast='integer')
male_workers=pd.to_numeric(male_workers,downcast='integer')
female_workers=pd.to_numeric(female_workers,downcast='integer')
all_salary=pd.to_numeric(all_salary,downcast='integer')


# **Occupation Lable Encoder**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_x=LabelEncoder()
occupation_number=label_x.fit_transform(occupation)


# **Creating a new Data Frame and adding new column into the data**

# In[ ]:


integer_data=pd.DataFrame()
integer_data['occupation_number']=occupation_number
integer_data['Occupation']=occupation
integer_data['all_workers']=all_workers
integer_data['all_salary']=all_salary
integer_data['male_workers']=male_workers
integer_data['male_salary']=male_salary
integer_data['female_workers']=female_workers
integer_data['female_salary']=female_salary
integer_data['male_share'] = integer_data.male_workers/integer_data.all_workers 
integer_data['female_share'] = integer_data.female_workers/integer_data.all_workers 
integer_data['non_weighted_all_weekly'] = (integer_data.male_salary + integer_data.female_salary)/2
integer_data['salary_gap'] = integer_data.male_salary - integer_data.female_salary
integer_data['ftom_ratio'] = integer_data.female_salary/integer_data.male_salary
integer_data['workers_ratio'] = integer_data.female_workers/integer_data.male_workers


# **Based on the workers ratio if the workers ration is >1 that indicates that the male probablility of getting the jon and vice versa. Males-1 Females-0**

# In[ ]:


l=[]
array=integer_data['workers_ratio'].values
for i in array:
    if i>1:
        l.append(0)
    else:
        l.append(1)
integer_data['male_female_prob']=l
integer_data


# In[ ]:


integer_data.describe()


# In[ ]:


integer_data.count()


# **Finding outliers and as there are very few outliers and as the values are unique for different occupations the outliers are not removed**

# In[ ]:


boxplot = integer_data.boxplot(column=['male_salary'])


# In[ ]:


boxplot = integer_data.boxplot(column=['female_salary'])


# In[ ]:


boxplot = integer_data.boxplot(column=['salary_gap'])


# **Data Visualisation**

# In[ ]:


sorted_df = integer_data.sort_values(['ftom_ratio'], ascending = [True])

from matplotlib import *

#plt.bar(sorted_df.tail(10).ftom_ratio,sorted_df.tail(10).Occupation,color='grey')
plt.bar(sorted_df.tail(10).Occupation, sorted_df.tail(10).ftom_ratio,color='olive')
#plt.xlabel(['numberofw','share','gender','non_weighted_all_weekly'], size=15)
plt.ylabel(" Female to male Salary Ratio", size=15)
#plt.title("Fields with equal salary ratio", size=18)
plt.xticks(rotation=90)
plt.yticks(rotation=90)


# In[ ]:


from matplotlib import *
plt.bar(sorted_df.head(10).Occupation, sorted_df.head(10).ftom_ratio,color='blueviolet')
#plt.xlabel(['numberofw','share','gender','non_weighted_all_weekly'], size=15)
plt.ylabel(" Female to male Salary Ratio", size=15)
#plt.title("Fields with unequal salary ratio", size=18)
plt.xticks(rotation=90)
plt.yticks(rotation=90)


# In[ ]:


sorted_df = integer_data.sort_values(['female_share'], ascending = [True])

plt.figure(figsize = (10,10))

from matplotlib import *
plt.bar(sorted_df.head(10).Occupation, sorted_df.head(10).female_share,color='peru')
#plt.xlabel(['numberofw','share','gender','non_weighted_all_weekly'], size=15)
plt.ylabel(" Female Share", size=15)
#plt.title("Fields with smallest share of women", size=18)
plt.xticks(rotation=90)
plt.yticks(rotation=90)


# In[ ]:


from matplotlib import *
plt.bar(sorted_df.tail(10).Occupation, sorted_df.tail(10).female_share,color='hotpink')
#plt.xlabel(['numberofw','share','gender','non_weighted_all_weekly'], size=15)
plt.ylabel(" Female share", size=15)
#plt.title("Fields with largest share of women", size=18)
plt.xticks(rotation=90)
plt.yticks(rotation=90)


# In[ ]:


sorted_df = integer_data.sort_values(['non_weighted_all_weekly'], ascending = [True])


from matplotlib import *
plt.bar(sorted_df.tail(10).Occupation, sorted_df.tail(10).non_weighted_all_weekly,color='orangered')
#plt.xlabel(['numberofw','share','gender','non_weighted_all_weekly'], size=15)
plt.ylabel(" salary", size=15)
#plt.title("Most paying fields", size=18)
plt.xticks(rotation=90)
plt.yticks(rotation=90)


# In[ ]:


from matplotlib import *
plt.bar(sorted_df.head(10).Occupation, sorted_df.head(10).non_weighted_all_weekly,color='grey')
#plt.xlabel(['numberofw','share','gender','non_weighted_all_weekly'], size=15)
plt.ylabel(" salary", size=15)
#plt.title("Least paying fields", size=18)
plt.xticks(rotation=90)
plt.yticks(rotation=90)


# **Male Salary, Female Salary and Salary Gap distribution to find the minimum maximum and average**

# In[ ]:


sns.distplot(integer_data.male_salary, bins = np.linspace(0.4,1.2,28))
plt.title('Male Salary Distribution')


# In[ ]:


sns.distplot(integer_data.female_salary, bins = np.linspace(0.4,1.2,28))
plt.title('Female salary Distribution')

np.mean(integer_data.female_salary)


# In[ ]:


sns.distplot(integer_data.non_weighted_all_weekly, bins = np.linspace(0.4,1.2,28))
plt.title('non_weighted_all_weekly Distribution')


# In[ ]:


sns.distplot(integer_data.salary_gap, bins = np.linspace(0.4,1.2,28))
plt.title('Salary Gap Distribution')


# **K- Means Clustering for male salary vs Occupation**

# **Elbow method to select the number of clusters and from the graph 4 clusters are selected**

# In[ ]:


from sklearn.cluster import KMeans
male_salarydf=pd.DataFrame()
male_salarydf['Male_Salary']=male_salary
male_salarydf['occupation_number']=occupation_number
male_salarydf
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(male_salarydf)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()


# **Silhoutte Score also indicates 4 clusters**

# In[ ]:


from sklearn.metrics import silhouette_score
silhouette_scores = [] 

for n_cluster in range(2,8):
    silhouette_scores.append(silhouette_score(male_salarydf, KMeans(n_clusters = n_cluster).fit_predict(male_salarydf))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show()


# In[ ]:


male_salarydf


# In[ ]:


kmeans=KMeans(n_clusters=4,random_state=0)
y_kmeans_m=kmeans.fit_predict(male_salarydf)
y_kmeans_m


# In[ ]:


male_salarydf['clusters']=y_kmeans_m
male_salarydf['Occupation']=occupation
male_salarydf
cluster1=male_salarydf[male_salarydf['clusters']==0]
cluster1=cluster1.reset_index(drop=True)

cluster2=male_salarydf[male_salarydf['clusters']==1]
cluster2=cluster2.reset_index(drop=True)

cluster3=male_salarydf[male_salarydf['clusters']==2]
cluster3=cluster3.reset_index(drop=True)

cluster4=male_salarydf[male_salarydf['clusters']==3]
cluster4=cluster4.reset_index(drop=True)


# **Each Cluster Description**

# In[ ]:


print(cluster1.describe())
print(cluster2.describe())
print(cluster3.describe())
print(cluster4.describe())


# In[ ]:


male_salarydf=male_salarydf.values
plt.scatter(male_salarydf[y_kmeans_m==0,1],male_salarydf[y_kmeans_m==0,0],s=100,c='blue',label='Low')
plt.scatter(male_salarydf[y_kmeans_m==1,1],male_salarydf[y_kmeans_m==1,0],s=100,c='green',label='High')
plt.scatter(male_salarydf[y_kmeans_m==2,1],male_salarydf[y_kmeans_m==2,0],s=100,c='purple',label='Very High')
plt.scatter(male_salarydf[y_kmeans_m==3,1],male_salarydf[y_kmeans_m==3,0],s=100,c='red',label='Medium')
plt.title('kmeans_clustering-male')
plt.xlabel('Occupation_number')
plt.ylabel('Salary')
plt.legend()
plt.show()


# **K-Means Clustering for Female Salary Vs Occupation**

# In[ ]:


from sklearn.cluster import KMeans
female_salarydf=pd.DataFrame()
female_salarydf['female_salary']=female_salary
female_salarydf['occupation_number']=occupation_number
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(female_salarydf)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()


# **Both the methods indicate 4 clusters**

# In[ ]:


from sklearn.metrics import silhouette_score
silhouette_scores = [] 

for n_cluster in range(2,8):
    silhouette_scores.append(silhouette_score(female_salarydf, KMeans(n_clusters = n_cluster).fit_predict(female_salarydf))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show()


# In[ ]:


female_salarydf


# In[ ]:


kmeans=KMeans(n_clusters=4,random_state=0)
y_kmeans=kmeans.fit_predict(female_salarydf)
y_kmeans


# In[ ]:


female_salarydf['clusters']=y_kmeans
female_salarydf['Occupation']=occupation
female_salarydf
c1=female_salarydf[female_salarydf['clusters']==0]
c1=c1.reset_index(drop=True)

c2=female_salarydf[female_salarydf['clusters']==1]
c2=c2.reset_index(drop=True)

c3=female_salarydf[female_salarydf['clusters']==2]
c3=c3.reset_index(drop=True)

c4=female_salarydf[female_salarydf['clusters']==3]
c4=c4.reset_index(drop=True)


# In[ ]:


print(c1.describe())
print(c2.describe())
print(c3.describe())
print(c4.describe())


# In[ ]:


female_salarydf=female_salarydf.values
plt.scatter(female_salarydf[y_kmeans==0,1],female_salarydf[y_kmeans==0,0],s=100,c='blue',label='Very High')
plt.scatter(female_salarydf[y_kmeans==1,1],female_salarydf[y_kmeans==1,0],s=100,c='green',label='Medium')
plt.scatter(female_salarydf[y_kmeans==2,1],female_salarydf[y_kmeans==2,0],s=100,c='purple',label='Low')
plt.scatter(female_salarydf[y_kmeans==3,1],female_salarydf[y_kmeans==3,0],s=100,c='red',label='High')
plt.title('kmeans_clustering-female')
plt.xlabel('Occupation_Number')
plt.ylabel('Salary')
plt.legend()
plt.show()


# **Knn Classification**

# Using female share of workers and female to male salary ratio to classify the probability of job going for males or females

# In[ ]:


random=pd.DataFrame()
x_new=integer_data.iloc[:,[9,12]].values
y_new=integer_data.iloc[:,14].values
from sklearn.model_selection  import train_test_split
X_train,X_test, y_train,y_test= train_test_split(x_new,y_new,test_size=0.3 ,random_state=21)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=2,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print(accuracies)

from matplotlib.colors import ListedColormap
x_set,y_set=X_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min() - 1,stop=x_set[:,0].max() + 1,step=0.01),
                   np.arange(start=x_set[:,1].min() - 1,stop=x_set[:,1].max() + 1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
              alpha=0.75,cmap=ListedColormap(("cyan","orange")))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                 c=ListedColormap(("red","green"))(i),label=j)
plt.title("K-NN(Training Set)")
plt.xlabel("Female Share of Workers")
plt.ylabel("Female to Male Salary Ratio")
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
x_set,y_set=X_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min() - 1,stop=x_set[:,0].max() + 1,step=0.01),
                   np.arange(start=x_set[:,1].min() - 1,stop=x_set[:,1].max() + 1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
              alpha=0.75,cmap=ListedColormap(("cyan","orange")))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                 c=ListedColormap(("red","green"))(i),label=j)
plt.title("K-NN(Test Set)")
plt.xlabel("Female Share of Workers")
plt.ylabel("Female to Male Salary Ratio")
plt.legend()
plt.show()


# **Heat Map**

# In[ ]:


corr=integer_data.corr()
sns.heatmap(corr)


# **Regression Model and the attributes Co-efficients plot**

# In[ ]:


columns=['ftom_ratio','all_salary','male_share','female_share']
X = integer_data[columns]
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
y = integer_data['salary_gap']

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=7)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

plt.bar(['ftom_ratio','all_salary','male_share','female_share'], model.coef_,data=integer_data,color='orange')
#plt.bar(['workers_ratio'], model.coef_,data=integer_data,color='blue')
#plt.xlabel(['numberofw','share','gender','non_weighted_all_weekly'], size=15)
plt.ylabel('Coeffecients', size=15)
plt.title("Proportionality of Attributes", size=18)


print('R^2 on training...',model.score(X_train,y_train))
print('R^2 on test...',model.score(X_test,y_test))
accuracy=model.score(X_test,y_test)
print('The Accuracy of the model built is ',accuracy*100,'%')

print('Model slope:  :', model.coef_[0])
print('Model intercept :', model.intercept_)
print('Regression Line . :',model.coef_[0],'* x + ',model.intercept_ )


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
 
# Make a fake dataset:
height = [-126.78, 102.733, 84.059, 85.415, -31.499]
bars = ('Female Male Salary Ratio', 'Male Salary', 'Median Salary', 'Average Male Female Salary', 'Workers Ratio')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height,color='magenta')
 
# Create names on the x-axis
plt.xticks(y_pos, bars)
plt.ylabel('Coeffecients')
plt.title('The Coeffecients w.r.t Salary Gap')
 
# Show graphic
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
 
# Make a fake dataset:
height = [0.728, 0.624, 0.477, 0.502,0.958]
bars = ('Female Male Salary Ratio', 'Male Salary', 'Median Salary', 'Average Male Female Salary', 'Workers Ratio')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height,color='blue')
 
# Create names on the x-axis
plt.xticks(y_pos, bars)
plt.ylabel('R^2 Value')
plt.title('R^2 Values on Test Data')
 
# Show graphic
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
 
# Make a fake dataset:
height = [0.723, 0.478, 0.325, 0.334,0.0384]
bars = ('Female Male Salary Ratio', 'Male Salary', 'Median Salary', 'Average Male Female Salary', 'Workers Ratio')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height,color='green')
 
# Create names on the x-axis
plt.xticks(y_pos, bars)
plt.ylabel('R^2 Value')
plt.title('R^2 Values on Training Data')
 
# Show graphic
plt.show()


# In[ ]:


X_linearregression=integer_data.iloc[:,10].values.reshape(-1,1)
y_linearregression=integer_data.iloc[:,5].values.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X_linearregression)

from sklearn.model_selection  import train_test_split
X_train,X_test, y_train,y_test= train_test_split(X_std,y_linearregression,test_size=0.3 ,random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predic=regressor.predict(X_test)
y_predic

print('Model slope:  :', regressor.coef_[0])
print('Model intercept :', regressor.intercept_)
print('Regression Line . :',regressor.coef_[0],'* x + ',regressor.intercept_ )
print('R^2 value onis :',regressor.score(X_train,y_train))
print('R^2 value is :',regressor.score(X_test,y_test))
print(regressor.coef_)

accuracy=regressor.score(X_test,y_test)
print('The Accuracy of the model built is ',accuracy*100,'%')


plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title(' Average Salary vs Male Salary  (Training set)')
plt.ylabel('Male Salary ')
plt.xlabel('Average Slary')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Average Salary vs Male Salary(Test set)')
plt.ylabel('Male Salary ')
plt.xlabel('Average Slary ')
plt.show()


# In[ ]:


X_linearregression=integer_data.iloc[:,10].values.reshape(-1,1)
y_linearregression=integer_data.iloc[:,7].values.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X_linearregression)

from sklearn.model_selection  import train_test_split
X_train,X_test, y_train,y_test= train_test_split(X_std,y_linearregression,test_size=0.3 ,random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predic=regressor.predict(X_test)
y_predic

print('Model slope:  :', regressor.coef_[0])
print('Model intercept :', regressor.intercept_)
print('Regression Line . :',regressor.coef_[0],'* x + ',regressor.intercept_ )
print('R^2 value onis :',regressor.score(X_train,y_train))
print('R^2 value is :',regressor.score(X_test,y_test))
print(regressor.coef_)

accuracy=regressor.score(X_test,y_test)
print('The Accuracy of the model built is ',accuracy*100,'%')


plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title(' Average Salary vs Female Salary  (Training set)')
plt.ylabel('Female Salary ')
plt.xlabel('Average Slary')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Average Salary vs Female Salary(Test set)')
plt.ylabel('Female Salary ')
plt.xlabel('Average Slary ')
plt.show()


# **H Clustering for male salary vs occupation**

# In[ ]:


import scipy.cluster.hierarchy as sch
male_salaryh=pd.DataFrame()
male_salaryh['Male_Salary']=male_salary
male_salaryh['occupation_number']=occupation_number
male_salaryh
dendrogram=sch.dendrogram(sch.linkage(male_salaryh,method='ward'))
plt.title('Dendogram')
plt.xlabel('male salary')
plt.ylabel('Distance')
plt.show()


# **Dendrogram and Silhouette Score indicates 2 Clusters**

# In[ ]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
silhouette_scores = [] 

for n_cluster in range(2,8):
    silhouette_scores.append(silhouette_score(male_salaryh, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(male_salaryh))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show()


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(male_salaryh)


# In[ ]:


male_salaryh=male_salaryh.values
plt.scatter(male_salaryh[y_hc==0,1],male_salaryh[y_hc==0,0],s=100,c='blue',label='Low')
plt.scatter(male_salaryh[y_hc==1,1],male_salaryh[y_hc==1,0],s=100,c='green',label='High')
plt.title('H_clustering-male')
plt.xlabel('Occupation_Number')
plt.ylabel('Salary')
plt.legend()
plt.show()


# **H Clustering for Female Salary and Occupation**

# In[ ]:


import scipy.cluster.hierarchy as sch
female_salaryh=pd.DataFrame()
female_salaryh['female_salary']=female_salary
female_salaryh['occupation_number']=occupation_number
dendrogram=sch.dendrogram(sch.linkage(female_salaryh,method='ward'))
plt.title('Dendogram')
plt.xlabel('female salary')
plt.ylabel('Distance')
plt.show()


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
silhouette_scores = [] 

for n_cluster in range(2,8):
    silhouette_scores.append(silhouette_score(female_salaryh, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(female_salaryh))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show()


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
y_hcf=hc.fit_predict(female_salaryh)


# In[ ]:


female_salaryh=female_salaryh.values
plt.scatter(female_salaryh[y_hcf==0,1],female_salaryh[y_hcf==0,0],s=100,c='blue',label='High')
plt.scatter(female_salaryh[y_hcf==1,1],female_salaryh[y_hcf==1,0],s=100,c='green',label='Low')
plt.title('H_clustering-female')
plt.xlabel('Occupation')
plt.ylabel('Salary')
plt.legend()
plt.show()


# In[ ]:




