#!/usr/bin/env python
# coding: utf-8

# **In this notebook we'll explore the factors contributing to employees leaving their companies and we'll also try and predict this.**

# In[161]:


#Let's import everything we will need
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
import warnings
warnings.filterwarnings("ignore")
sns.set_style('whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[162]:


#Read the csv file and create feature / label sets
X = pd.read_csv('../input/HR_comma_sep.csv')
X.head()

y = X['left']

print('Number of records: ', X.shape[0])


# In[163]:


X.head()


# # Part 1 - explore the data
# 
# Let's look at our dataset first:

# Let's check for missing data:

# In[164]:


X.isnull().values.ravel().sum()


# Good, we don't have any missing values
# 
# Now, let's check the data types:
# 

# In[165]:


X.dtypes


# We will need the salary column for our analysis, so let's map it appropriately:
# 
# 

# In[166]:


X.salary.unique()


# In[167]:


X.salary.replace({'low':1,'medium':2,'high':3},inplace=True)


# Now, let's perform some simple statistics to better understand the data:

# In[168]:


X.describe()


# # Part 2 - visualize the data
# Let's look at the correlation matrix now:

# In[172]:


fig = plt.figure(figsize=(7,4))
corr = X.corr()
sns.heatmap(corr,annot=True,cmap='seismic',
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title('Heatmap of Correlation Matrix')


# There seems to be a clear inverse relation between satisfaction levels and people leaving
# 
# 
# 

# We will now visualize the distribution of several features of interest using a histogram or a kernel density estimate plot.

# In[173]:


plt.figure(figsize = (12,8))
plt.subplot(1,2,1)
plt.plot(X.satisfaction_level[X.left == 1],X.last_evaluation[X.left == 1],'ro', alpha = 0.2)
plt.ylabel('Last Evaluation')
plt.title('Employees who left')
plt.xlabel('Satisfaction level')

plt.subplot(1,2,2)
plt.title('Employees who stayed')
plt.plot(X.satisfaction_level[X.left == 0],X.last_evaluation[X.left == 0],'bo', alpha = 0.2,)
plt.xlim([0.4,1])
plt.ylabel('Last Evaluation')
plt.xlabel('Satisfaction level')


# There are 3 clusters for those employees who left.
# 
#     The happy and appreciated - we'll call them "Winners" - those who leave because they were offered a better opportunity.
# 
#     The appreciated but unhappy - Maybe they are over-qualified for the job. Let's call them the "Frustrated"
# 
#     The unappreciated and unhappy - It is not surprising that these would leave, possibly even fired. They are simply a "Bad Match"
# 

# Let's plot the clusters using the 3 classifications above:

# In[174]:


from sklearn.cluster import KMeans
kmeans_df =  X[X.left == 1].drop([ u'number_project',
       u'average_montly_hours', u'time_spend_company', u'Work_accident',
       u'left', u'promotion_last_5years', u'sales', u'salary'],axis = 1)
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(kmeans_df)
print(kmeans.cluster_centers_)

left = X[X.left == 1]
left['label'] = kmeans.labels_
plt.figure(figsize=(10,7))
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation score')
plt.title('3 Clusters of employees who left')
plt.plot(left.satisfaction_level[left.label==0],left.last_evaluation[left.label==0],'o', alpha = 0.2, color = 'r')
plt.plot(left.satisfaction_level[left.label==1],left.last_evaluation[left.label==1],'o', alpha = 0.2, color = 'g')
plt.plot(left.satisfaction_level[left.label==2],left.last_evaluation[left.label==2],'o', alpha = 0.2, color = 'b')
plt.legend(['Winners','Frustrated','Bad Match'], loc = 'best', fontsize = 13, frameon=True)


# Let's look the the average monthly hours distribution for those that left:

# In[175]:


winners = left[left.label ==0]
frustrated = left[left.label == 1]
bad_match = left[left.label == 2]

plt.figure(figsize=(10,4))
sns.kdeplot(winners.average_montly_hours, color = 'r', shade=True)
sns.kdeplot(bad_match.average_montly_hours, color ='b', shade=True)
sns.kdeplot(frustrated.average_montly_hours, color ='g', shade=True)
plt.legend(['Winners','Bad Match','Frustrated'])
plt.title('Leavers: Hours per month distribution')


# There's definitely some useful information here.
# 
# It seems that the frustrated group works by far the longest hours (possibly, and understandably, their reason to be frustrated). The winners also work long hours, and those who are a bad match work significantly less hours.

# We will now visualize the distribution of several features of interest using a histogram or a kernel density estimate plot.

# In[176]:


#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure(figsize=(10,4))
ax=sns.kdeplot(X.loc[(X['left'] == 0),'satisfaction_level'] , color='b',shade=True, label='Stayed')
ax=sns.kdeplot(X.loc[(X['left'] == 1),'satisfaction_level'] , color='r',shade=True, label='Left')
plt.title('Satisfaction levels')


# People who have left were in general less satisfied with their jobs. However there are also people who are quite satisfied with their jobs but still left. This means that there are other factors contributing to an employee leaving their job other than them being satisfied with their job or not. satisfaction_level` seems to be a good feature in differentiating people who left the company from people who did not leave.

# In[178]:


#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure(figsize=(10,4),)
ax=sns.kdeplot(X.loc[(X['left'] == 0),'last_evaluation'] , color='b',shade=True,label='Stayed')
ax=sns.kdeplot(X.loc[(X['left'] == 1),'last_evaluation'] , color='r',shade=True, label='Left')
plt.title('Last evaluation')


# Looks like people who leave the company either did pretty bad or pretty good in their last performance evaluation. There are not many medium level performers among the people who left. If an employee is evaluated in the range of 0.6 to 0.8 (ball park), they are likely still working in the company.

# In[179]:


#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure(figsize=(10,4))
ax=sns.kdeplot(X.loc[(X['left'] == 0),'number_project'] , color='b',shade=True, label= 'Stayed')
ax=sns.kdeplot(X.loc[(X['left'] == 1),'number_project'] , color='r',shade=True, label= 'Left')
plt.title('Number of projects')


# People who left mostly work on a small number of projects (2), or on a large number of projects (5-7). I added a new feature: number of projects worked on per year. I defined this as the number of projects the employee works on during his/her employment period, divided by the total number of years the employee worked in the company. People who left mostly work on a lower number of projects per year when compare to the employees who have not left.

# In[180]:


#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure(figsize=(10,4))
ax=sns.kdeplot(X.loc[(X['left'] == 0),'average_montly_hours'] , color='b',shade=True, label='Stayed')
ax=sns.kdeplot(X.loc[(X['left'] == 1),'average_montly_hours'] , color='r',shade=True, label='Left')
plt.title('Average monthly hours worked')


# People who left either work a small amount of hours on average per month (lower than 150 hours), or they work a large number of hours (more than 250). This means that the employees who leave tend to either not work much or work a lot. The fact that employees who leave are evaluated either bad or quite good in their last performance evaluation might be related to this fact.

# In[181]:


#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure(figsize=(10,4))
ax=sns.kdeplot(X.loc[(X['left'] == 0),'salary'] , color='b',shade=True, label='Stayed')
ax=sns.kdeplot(X.loc[(X['left'] == 1),'salary'] , color='r',shade=True, label='Left')
plt.title('Salary: (1-Low; 2-Medium; 3-high)')


# We can clearly spot a trend here - the higher the salary, the lower the chances of employees leaving. 

# In[182]:


fig = plt.figure(figsize=(10,4),)
sns.barplot(x = 'time_spend_company', y = 'left', data = X, saturation=1)


# The majority of employees tend to leave around the 5th year. 

# # Part 3 - Machine learning
# *Let's break up our set into training and testing sets, as we only have one file*

# In[183]:


features = X[['satisfaction_level','average_montly_hours','promotion_last_5years','salary','number_project']]
X = features
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state=0,test_size=0.25)
print('Training set volume:', X_train.shape[0])
print('Test set volume:', X_test.shape[0])


# In[184]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[185]:


logreg.fit(X_train,y_train)


# In[186]:


accuracy_score(y_test,logreg.predict(X_test))


# In[187]:


# Create Naive Bayes classifier
clf_gb = GaussianNB()
clf_gb.fit(X_train, y_train)
predicts_gb = clf_gb.predict(X_test)
print("GB Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(y_test, predicts_gb))


# In[188]:


#Create k-nn
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print("KNN5 Accuracy Rate, which is calculated by accuracy_score() is: %f" %accuracy_score(y_test,y_pred))


# Let's try to optimize the KNN model:

# In[189]:


k_range = range(1,26)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))

plt.plot(k_range,scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Testing accuracy')


# In[190]:


#Decision Tree
clf_dt = tree.DecisionTreeClassifier(min_samples_split=25)
clf_dt.fit(X_train, y_train)
predicts_dt = clf_dt.predict(X_test)
print("Decision tree Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(y_test, predicts_dt))


# In[191]:


#SVM -> takes a few seconds to run!
clf_svm = svm.SVC(kernel='rbf',probability=False)
clf_svm.fit(X_train,y_train)
predict_svm = clf_svm.predict(X_test)
print("SVM Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(y_test, predict_svm))


# In[192]:


#Random forest classifier
clf_rf = RandomForestClassifier(n_estimators = 10,min_samples_split=2,max_depth=30)
clf_rf.fit(X_train, y_train)
accuracy_rf = clf_rf.score(X_test,y_test)
print("Random Forest Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_rf)


# In[193]:


# The Random forest classifier seems to produce the best results so far, so let's try to optimise it!

max_depth = range(1,50)
scores=[]
for r in max_depth:
    clf_rf = RandomForestClassifier(n_estimators = 10,min_samples_split=2,max_depth=r)
    clf_rf.fit(X_train,y_train)
    y_pred=clf_rf.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))
    
plt.plot(max_depth,scores)
plt.xlabel('Value of r for Random Forest')
plt.ylabel('Testing accuracy')

# we're getting different values each time, but depth of around 30 seems to give good results.


# The model which gives us the best result appears to be the Random Forest Classifier.

# # Conclusion
# 
# We now have a working model for predicting who is likely to leave the company, based on 5 input parameters.
