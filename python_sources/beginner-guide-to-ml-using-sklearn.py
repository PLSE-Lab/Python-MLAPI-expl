#!/usr/bin/env python
# coding: utf-8

# ## A beginner guide for implementing machine learning model using scikit learn and pandas.
# ## Introducing various concepts such as train_test_split,grid_searchCV,randomised_searchCV and many more .
# ## Iris dataset available in scikitlearn is used for this purpose.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.datasets import load_iris
iris=load_iris()


# In[ ]:


print(iris.feature_names)


# In[ ]:


print(iris.target_names)


# In[ ]:


X=iris.data


# In[ ]:


y=iris.target


# In[ ]:


print(y)


# In[ ]:


y_df=pd.DataFrame(y)
y_df.head()


# In[ ]:


y_df.columns=['species_index']


# In[ ]:


y_df


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


data=pd.DataFrame(X,columns=iris.feature_names)
n_data=pd.DataFrame(X,columns=['sepal_len','sepal_wid','petal_len','petal_wid'])


# In[ ]:


n_data.head()


# In[ ]:


y_df.loc[:,:]


# In[ ]:


n_data['species_index']=y_df.loc[:,:]


# In[ ]:


n_data.head()


# In[ ]:


data.head()


# In[ ]:


data.columns=data.columns.str.replace(' ','_')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


get_ipython().run_line_magic('pinfo', 'sns.pairplot')


# In[ ]:


sns.pairplot(n_data,size=3.5,vars=['sepal_len','sepal_wid','petal_len','petal_wid'],hue='species_index')


# we can clearly see that *petal_len* vs *petal_wid* proves a more promising feature to seperate the three sepecies from each other,more presisely the *blue ones* are seems to be seperated in quite a good way from each other.  

# In[ ]:


sns.pairplot(n_data,x_vars=['sepal_len','sepal_wid'],y_vars=['petal_len','petal_wid'],hue='species_index',size=4)


# ### plotting graphs gives a kind of insight to the data and thus we have a better understanding of our dataset.
# ### It is always advisable to kind of look inside the data and plotting is the best way to acheive this.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


data_train,data_test,y_train,y_test=train_test_split(data,y,test_size=0.25,random_state=5)


# In[ ]:


print(data_train.shape)
print(data_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)


# In[ ]:


knn.fit(data_train,y_train)


# In[ ]:


knn.predict([[5.1,4.8,2.4,3.5]])


# In[ ]:


knn.predict_proba([[5.1,4.8,2.4,3.5]])


# In[ ]:


#predicting on x_test 
y_pred=knn.predict(data_test)


# In[ ]:


print(y_pred)
print(y_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.accuracy_score(y_test,y_pred)


# In[ ]:


#choosing a different n_neighbors and again evaluating the accuracy
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(data_train,y_train)
y_pred=knn.predict(data_test)
metrics.accuracy_score(y_test,y_pred)


# In[ ]:


#choosing the best value of n_neighbors
a=[]
for i in range(1,30):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(data_train,y_train)
    y_pred=knn.predict(data_test)
    a.append(metrics.accuracy_score(y_test,y_pred))
print(a)    


# In[ ]:


plt.plot(a,'b--')
plt.grid()
plt.xlabel('neighbors')
plt.ylabel('accuracy_score')
plt.title('graph')


# In[ ]:


print(a[15])


# In[ ]:


#using n_neighbors=16 as it gives maximum accuracy
data_train,data_test,y_train,y_test=train_test_split(data,y,random_state=5,test_size=0.25)
knn=KNeighborsClassifier(n_neighbors=16)
knn.fit(data_train,y_train)
y_pred=knn.predict(data_test)
metrics.accuracy_score(y_test,y_pred)


# In[ ]:


#now changing the random state let's see what effect it has on accuracy_score
data_train,data_test,y_train,y_test=train_test_split(data,y,random_state=1,test_size=0.25)
knn=KNeighborsClassifier(n_neighbors=16)
knn.fit(data_train,y_train)
y_pred=knn.predict(data_test)
metrics.accuracy_score(y_test,y_pred)


# # we can see that train_test_split gives high varience estimate and changing the training and testing set greately changes the accuracy_score.
# # To solve this problem we can use k_fold_crossvalidation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,data,y,cv=10,scoring='accuracy')
print(scores)


# In[ ]:


#selecting the best n_neighbor for knn using cross_val_score
i_range=list(range(1,31))
a=[]
for i in i_range:
    knn=KNeighborsClassifier(n_neighbors=i)
    scores=cross_val_score(knn,data,y,cv=10,scoring='accuracy')
    a.append(scores.mean())
print(a)    


# In[ ]:


plt.plot(i_range,a,'b')
plt.xlabel('n_neighnors')
plt.ylabel('accuracy_score')
plt.grid()
plt.title('accuracy_graph')


# graph shows a more reliable accuracy score and thus helps in finding a better value of n_neighbors

# In[ ]:


max_acc=max(a)
print(max_acc)


# In[ ]:


a.index(max_acc)


# In[ ]:


print(a[16])


# we can see that for n_neighbors=16 the accuracy predicted by train_test_split was 100% but in reality after using n_neighbors=16 with cross_validation the score comes out to be lesser than before.

# now we can conclude that for knn the choice of best n_neighbors=13 or 21 and the accuracy is approx 98%

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=13)
scores=cross_val_score(knn,data,y,cv=10,scoring='accuracy')
print(scores.mean())


# ## Evaluating different models by cross_validation

# Evaluation by logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
scores=cross_val_score(logreg,data,y,cv=10,scoring='accuracy')
print(scores)


# In[ ]:


print(scores.mean())


# we can see that logistic regression has accuracy lower than knn so by using crossvalidation we can select best model for tha data.

# ***
# let's try **Support Vector Machine(SVM)** 

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clf=SVC(kernel='linear')
clf.fit(data_train,y_train)
y_pred=clf.predict(data_test)
print(metrics.accuracy_score(y_test,y_pred))


# By tuning the kernel to 'linear' we are getting 100% accuracy.But we are using train_test_split so, just evaluate the accuracy again by k_fold cross validation.
# But first let's tune the kernel to 'rbf'.

# In[ ]:


clf=SVC(kernel='rbf',C=1)
clf.fit(data_train,y_train)
y_pred=clf.predict(data_test)
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


clf=SVC(kernel='linear')
scores=cross_val_score(clf,data,y,cv=10,scoring='accuracy')
print(scores)


# In[ ]:


print(scores.mean())


#             By using cross validation thre accuracy is pretty much less than what we got using train_test_split

# In[ ]:


#by changing the kernel to 'rbf'
clf=SVC(kernel='rbf')
scores=cross_val_score(clf,data,y,cv=10,scoring='accuracy')
print(scores.mean())


#       rbf performed better than linear kernel.

# ## let's try Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf=RandomForestClassifier(n_estimators=10)
clf.fit(data_train,y_train)
y_pred=clf.predict(data_test)
print(metrics.accuracy_score(y_test,y_pred).mean())


# let's try a different value of n_estimator.

# In[ ]:


clf=RandomForestClassifier(n_estimators=1000)
clf.fit(data_train,y_train)
y_pred=clf.predict(data_test)
print(metrics.accuracy_score(y_test,y_pred).mean())


# In[ ]:


a=[]
clf_range=list(range(1,100))
for i in clf_range:
    clf=RandomForestClassifier(n_estimators=i)
    scores=cross_val_score(clf,data,y,cv=10,scoring='accuracy')
    a.append(scores.mean())
plt.plot(clf_range,a)


#  Random Forest is giving the accuracy of approx 97% thus fro this we conclude that the best fit model on our data is **support vector classifier** and **knn with n_neighbors=13** 

#  we can further try to improve the accuracy by tuning the parametrs using **grid searchCV**.

# ---
# ----------------------

# ## feature selection by crossvalidation

# we will use knn with n_neighbors=13 because it is the better than logistic regression.now we will se if accuracy can be further incresed by selecting good features 

# In[ ]:


data.head()


# In[ ]:


sns.pairplot(data)


# petal_length and petal_width seems to be a good choice because it is able to seperate the species in a better wey relative to other features.

# In[ ]:


sns.pairplot(data,x_vars='petal_length_(cm)',y_vars='petal_width_(cm)',size=5)


# In[ ]:


new_data=data[['sepal_length_(cm)','sepal_width_(cm)']]


# In[ ]:


new_data.head(10)


# In[ ]:


y


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,new_data,y,cv=10,scoring='accuracy')
print(scores.mean())


# In[ ]:


a=[]
new_range=list(range(1,31))
for i in new_range:
    knn=KNeighborsClassifier(n_neighbors=i)
    scores=cross_val_score(knn,new_data,y,cv=10,scoring='accuracy')
    a.append(scores.mean())
print(a)    


# In[ ]:


plt.plot(new_range,a,'r')
plt.grid()
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.title('graph')


# In[ ]:


print(max(a))


# The maximum accuracy obtained by onlu selecting two features is 80% ,which is not doing any good.we can switch back to original datafrane or try some different combinations.
# 
# To perform the task of trying different combinations efficiently we can use **grid search cv** 

# ## Trying combinations with grid search cv

# >**grid searchCV** is basically a way to automate the process of choosing the best parameters,which was initially done by k_fold cross validation by trying every feature and then computing the accuracy score by using **k_fold CrossValidation**.
# 
# >**grid searchCV** allows to pass in a dictionary of all features  mapped with a bunch of different possible values.
# And what **grid searchCV** does is ,it compute accuracy score by trying each possible combination of these features and then calculate the accuracy by using **k_fold CrossValidation**.
# 
# >The process becomes computationally infeasible if the size of dataset is large,because all combinations of featuresalong with k_fold validation kindof becomes too much. 
# 
# >**grid searchCV** takes a parameter *param_grid*.
# 
# >**param_grid**(as written in documentation):- Dictionary with parameters names (string) as keys and lists of
#     parameter settings to try as values, or a list of such
#     dictionaries, in which case the grids spanned by each dictionary
#     in the list are explored. This enables searching over any sequence
#     of parameter settings.

# In[ ]:


from sklearn.model_selection import GridSearchCV 


# In[ ]:


# finding the best_value of n_neighbors for knn model using grid_search
# creating param_grid
k_range=list(range(2,31))
print(k_range)


# In[ ]:


param_grid=dict(n_neighbors=k_range)
print(param_grid)


# In[ ]:


grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy',return_train_score=True)


# In[ ]:


grid.fit(data,y)


# In[ ]:


#getting the best estimator as found by grid_search.
grid.best_estimator_


# >we can see that the value of best n_neighbor for knn comes out to be exectly the same i.e 13 as previously found by writting loop and calculating accuracy for every neighbors.GridSearch does this job autometically.
#     
#  we can even feed multiple dictionary to param_grid and it will select the best estimator from each dictionary.   

# In[ ]:


grid.best_params_


# In[ ]:


#let's see the best score which can be acheived by training our model(knn in this case) with the best parameters.
grid.best_score_


# In[ ]:


df=pd.DataFrame(grid.cv_results_)


# In[ ]:


df.head()


# In[ ]:


df_sub=df.loc[:,['mean_train_score','std_train_score','params']]


# In[ ]:


df_sub.head()


# >The first column is basically giving the accuracy_score ,the second one is giving the varience of a particular estimator.
# If the varience is high then it might not be a good estimator.

# In[ ]:


scores=grid.cv_results_['mean_test_score']
print(scores)


# In[ ]:


plt.plot(k_range,scores,'r',linewidth=2)
plt.grid()
plt.xlabel('n_neighbors')
plt.ylabel('scores')
plt.title('graph')


# Got exactly the same graph as we got by cross_validation.

# ### we can train our models by using the best parameters which we got by grid searchCV.

# >To decrese the computational expense of **grid searchCV**  we can use a cousin of this called **randomised searchCV** .
# ## let's now see randomised searchCV

# >what **randomised searchCV** does is,it selects random combinations of given parameters and does fit, predict on the given data using **k_fold CrossValidation** and calculates the accuracy score.
# 
# >It takes much less time than **grid searchCV** as it only operates on a fixed number of times using random combinations of given parameters.
# 
# >**randomised searchCV** may produce less accurate result in comparison to **grid search** but it is very close to the best reult, it also saves a lot of computation time.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


param_dist=dict(n_neighbors=k_range)
print(param_dist)


# In[ ]:


grid_2=RandomizedSearchCV(knn,param_dist,cv=10,scoring='accuracy')


# In[ ]:



grid_2.fit(data,y)


# In[ ]:


grid_2.best_estimator_


# In[ ]:


grid_2.best_score_


# In[ ]:




