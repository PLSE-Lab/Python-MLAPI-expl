#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#A place for the imports
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data exploration
# 

# In[ ]:


abalone = pd.read_csv('../input/abalone.csv')
abalone.columns=['Sex','Length','Diameter','Height','Whole weight', 'Shucked weight','Viscera weight', 
                 'Shell weight','Rings']
abalone.sample(5)


# In[ ]:


abalone.info()


# 
# 
# 
# There are 8 numerical not-null features in the data. Feature Sex will need to be changed to dummy values in data preparation in order to use it in the model.
# 
# Let's investigate further the data as there is a possibility that some of the values that are not null are set to 0 instead.
# 
# 
# 
# 

# In[ ]:


abalone.describe()


# As mentioned it seems that there are minimum values in Height that are 0

# In[ ]:


abalone[abalone.Height == 0]


# There are two records where Height is equal to 0, it is possible that it was hard to measure it or it was simply omitted. Nevertheless, this can be treated as a NULL value and since there are only two records like that it will be simplest to ignore them.

# In[ ]:


abalone = abalone[abalone.Height > 0]
abalone.describe()


# In[ ]:


abalone.hist(figsize=(20,10), grid = False, layout=(2,4), bins = 30);


# Histograms show that the data may be skewed, so it will be reasonable to measure it. 
# 
# It also shows that there are possible outliers in Height and that there might be a strong relationship between the Diameter and Lenght and between Shell weight, Shucked weight Viscera weight and Whole weight.

# In[ ]:


nf = abalone.select_dtypes(include=[np.number]).columns
cf = abalone.select_dtypes(include=[np.object]).columns


# In[ ]:


skew_list = stats.skew(abalone[nf])
skew_list_df = pd.concat([pd.DataFrame(nf,columns=['Features']),pd.DataFrame(skew_list,columns=['Skewness'])],axis = 1)
skew_list_df.sort_values(by='Skewness', ascending = False)


# Skewness value points in which direction data is distorted in a statistical distribution, in Gaussian distribution the value for skewness is 0. In abalone data Height has highest skewness value followed by Rings.
# 
# High skewness in Height feature may be an outcome of outliers. I will investigate it further using scatter plots.

# # Scatter plots

# In[ ]:


sns.set()
cols = ['Length','Diameter','Height','Whole weight', 'Shucked weight','Viscera weight', 'Shell weight','Rings']
sns.pairplot(abalone[cols], height = 2.5)
plt.show();


# Observations:
#     
#     - Many features are highly correlated
#         - length and diameter show linear correlation
#         - the length and weight features are quadratic correlated
#         - whole weight is linearly correlated with other weight features
#     - Number of Rings is positively corelated with almost all quadratic features
#     - Possible outliers in Height features
#     
# Scatter plot analysis also shows that data mostly cover the values for Rings from 3 to little over 20, selecting only this data in the model may be taken under consideration to increase the accuracy.
# 
# First I will take a closer look at the Height outliers and then I will investigate correlations between the features.

# In[ ]:


data = pd.concat([abalone['Rings'], abalone['Height']], axis = 1)
data.plot.scatter(x='Height', y='Rings', ylim=(0,30));


# Two values seem not to follow the trend, that is why I will treat them as outliers and delete from data.

# In[ ]:


abalone = abalone[abalone.Height < 0.4]
data = pd.concat([abalone['Rings'], abalone['Height']], axis = 1)
data.plot.scatter(x='Height', y='Rings', ylim=(0,30));


# In[ ]:


abalone.hist(column = 'Height', figsize=(20,10), grid=False, layout=(2,4), bins = 30);


# Deleted data as suspected was the cause for the skewness of Height feature, now it is closer to a normal distribution.

# # Correlation matrix

# In[ ]:


corrmat = abalone.corr()
cols = corrmat.nlargest(8, 'Rings')['Rings'].index
cm = np.corrcoef(abalone[nf].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(15,15))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=nf.values, xticklabels=nf.values)
plt.show();


# The heat map shows that features are highly correlated and multicollinearity is possible.
# 
# 
#     -Whole weight is almost linearly correlated with all the features except Rings
#     -Length is linearly correlated with Diameter
#     -From all the features excluding Rings, Height is least correlated with other features
#     -Rings feature has the highest correlation with Shell Weight followed by Height, Length and Diameter
#     
# Possible solutions for a high level of collinearity in data:
# 
#     - Use principal component analysis(PCA) to generate new features
#     - Select partial features for modelling

# # Categorical Feature
# 
# 
# Finally, I will analyse the relation of Rings with the Sex feature

# In[ ]:


data = pd.concat([abalone['Rings'], abalone['Sex']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxenplot(x='Sex', y="Rings", data=abalone)
fig.axis(ymin=0, ymax=30);


#     -Distribution between Male and Female is similar
#     -Most of the Rings both for Male and Female are between 8 and 19
#     -Infants have mostly from 5 to 10 Rings
# The plot also shows that Rings majority lies between 3 to 22, as mentioned previously.

# ## Linear Regression Models

# First I will transofrm Sex feature 

# In[ ]:


abalone = pd.get_dummies(abalone)
abalone.head()


# Now I will set the X and y labels

# In[ ]:


X = abalone.drop(['Rings'], axis = 1)
y = abalone['Rings']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3)


# As this is linear problem I decided to use two models: Linear Regression and Ridge.
# 
# ### Linear Regression
# 
# Linear regression is a statistical model that examines the linear relationship between two  or more variables. Linear relationship means that when one (or more) independent variables increases (or decreases), the dependent variable increases (or decreases) also.
# 
# 

# In[ ]:


from sklearn.linear_model import LinearRegression 
paramLin = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
LinearReg = GridSearchCV(LinearRegression(),paramLin, cv = 10)
LinearReg.fit(X = X_train,y= y_train)
LinearRegmodel = LinearReg.best_estimator_
print(LinearReg.best_score_, LinearReg.best_params_)


# In[ ]:


LinearReg.score(X_train,y_train)


# In[ ]:


LinearReg.score(X_test,y_test)


# In[ ]:


predictions = LinearReg.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# ### Ridge Regression
# 
# 
# As mentioned previously there is high correlation between the features in the data. That is why i chosed to use Ridge Regression. Ridge Regression is a technique used when the data suffers from multicollinearity.By adding a degree of bias to the regression estimates, ridge regression reduces the standard errors. There is one value smaller than 0 which can be a topic of further investigatin. 
# 

# In[ ]:


from sklearn.linear_model import Ridge
paramsRidge = {'alpha':[0.01, 0.1, 1,10,100], 'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

ridgeReg = GridSearchCV(Ridge(),paramsRidge, cv = 10)
ridgeReg.fit(X = X_train,y= y_train)
Rmodel = ridgeReg.best_estimator_
print(ridgeReg.best_score_, ridgeReg.best_params_)


# In[ ]:


ridgeReg.score(X_train,y_train)


# In[ ]:


ridgeReg.score(X_test,y_test)


# In[ ]:


predictions = ridgeReg.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# ## K-means

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


# In[ ]:


kmeans = KMeans(n_clusters=3, random_state=0).fit(X_std)
y_kmeans = kmeans.predict(X_std)


# In[ ]:


plt.scatter(X_std[:, 0], X_std[:, 1], c=y_kmeans, s=50, cmap='viridis');

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# ## PCA

# As there is a high correlation between variables I will perform principal components analysis for dimensionality reduction.

# In[ ]:


corr_mat = np.corrcoef(X_std.T)


# In[ ]:


eigenvalues, eigenvectors = np.linalg.eig(corr_mat)
print('\nEigenvalues \n%s' %eigenvalues)


# In[ ]:


#eigenvalue and eigenvector pairs
pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
pairs.sort(key = lambda x: x[0], reverse = True)


# In[ ]:


sorted_eigenval = []
for i in pairs:
    sorted_eigenval.append(i[0])
print(sorted_eigenval)


# In[ ]:


total = sum(eigenvalues)
variance_explained = [(i/total)*100 for i in sorted_eigenval]


# In[ ]:


variance_explained


# In[ ]:


cum_variance_explained = np.cumsum(variance_explained)
cum_variance_explained


# In[ ]:



#Plot variance explained by the principal components
with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(8, 6))
    plt.bar(range(10), variance_explained, alpha=0.7, align='center',
            label='individual explained variance')
    plt.step(range(10), cum_variance_explained, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout();


# 
# From the above plot we see that the first three principal components can explain over 98% of the variation of the feature variables. We may project the original features from the 10-dimensional space to a 3-dimensional space.

# In[ ]:


projection_mat = np.hstack((pairs[0][1].reshape(10,1),
                           pairs[1][1].reshape(10,1),
                           pairs[2][1].reshape(10,1)))


# In[ ]:


X_new = X_std.dot(projection_mat)
X_new.shape


# ## Classification
# 

# Classification is the process of predicting the class of given data points. Classes are sometimes called as targets/ labels or categories. Classification predictive modeling is the task of approximating a mapping function (f) from input variables (X) to discrete output variables (y).
# 
# For example, spam detection in email service providers can be identified as a classification problem. This is s binary classification since there are only 2 classes as spam and not spam. A classifier utilizes some training data to understand how given input variables relate to the class. In this case, known spam and non-spam emails have to be used as the training data. When the classifier is trained accurately, it can be used to detect an unknown email.
#     
# For this task, I will use K-means clustering and Super Vector Machine.
# 
# 
# First I need to divide the Rings for that I will use the target value Age and divide it into young, medium and old.

# In[ ]:


abalone.head(5)


# In[ ]:


bins = [0,8,10,abalone['Rings'].max()]
group_names = ['young','medium','old']
abalone['Rings'] = pd.cut(abalone['Rings'],bins, labels = group_names)


# In[ ]:


dictionary = {'young':0, 'medium':1, 'old':2}
abalone['Rings'] = abalone['Rings'].map(dictionary)


# In[ ]:


abalone.head(10)


# In[ ]:


X = abalone.drop(['Rings'], axis = 1)
y = abalone['Rings']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)


# ### KNN
# 
# The k-Nearest-Neighbors  method of classification it is essentially classification by finding the most similar data points in the training data, and making an educated guess based on their classifications. This method is used in areas like recommendation systems, semantic searching, and anomaly detection.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
paramsKn = {'n_neighbors':range(1,30)}
Kneighbours = GridSearchCV(KNeighborsClassifier(),paramsKn, cv=10)

Kneighbours.fit(X=X_train,y=y_train)
Kmodel = Kneighbours.best_estimator_
print(Kneighbours.best_score_, Kneighbours.best_params_)


# ### SVM
# 
# The Support Vector Machine is a discriminative classifier formally defined by a separating hyperplane. The goal of the model is to output optimal hyperplane that will categorize the data into categories. There are many hyperplanes dividing data possible so the object is to find one that will maximize the distance from the line to the classes.

# In[ ]:


from sklearn.svm import SVC
paramsSvm = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                  'C':[0.1,1,10],'gamma':[0.01,0.1,0.5,1,2]}

Svm = GridSearchCV(SVC(),paramsSvm,cv=5)

Svm.fit(X_train,y_train)
model_svm = Svm.best_estimator_
print(Svm.best_score_,Svm.best_params_)

