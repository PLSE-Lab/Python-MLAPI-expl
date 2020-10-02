#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository. It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.The Iris dataset is a dataset for classification, machine learning, and data visualization.
# 
# The dataset contains: 3 classes (different Iris species) with 50 samples each, and then four numeric properties about those classes: Sepal Length, Sepal Width, Petal Length, and Petal Width.
# 
# The purpose of this notebook is to practice ML concepts along with data visualization. If you find it useful, please UPVOTE :)

# ## Set up

# In[ ]:


# import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid")


# In[ ]:


# import dataset
iris = pd.read_csv("../input/iris/Iris.csv")
iris.head()


# # EDA

# In[ ]:


# check the data
iris.info()


# In[ ]:


# check the statistics
iris.describe()


# In[ ]:


# Visualize distribution of indivisual parameter for different species
plt.figure(figsize=(15, 15))
# Sepal length
plt.subplot(2, 2, 1)
sns.violinplot(x='Species', y ='SepalLengthCm', data=iris, inner='quartile', palette='Paired')
plt.title('Sepal Length for Different Species', fontsize=12)
plt.xlabel('Species', fontsize=10)
plt.ylabel('Sepal Length(cm)', fontsize=10);

# Sepal width
plt.subplot(2, 2, 2)
sns.violinplot(x='Species', y ='SepalWidthCm', data=iris, inner='quartile', palette='Paired')
plt.title('Sepal width for Different Species', fontsize=12)
plt.xlabel('Species', fontsize=10)
plt.ylabel('Sepal Width(cm)', fontsize=10);

# Petal length
plt.subplot(2, 2, 3)
ax3 = sns.violinplot(x='Species', y ='PetalLengthCm', data=iris, inner='quartile', palette='Paired')
plt.title('Petal Length for Different Species', fontsize=12)
plt.xlabel('Species', fontsize=10)
plt.ylabel('Petal Width(cm)', fontsize=10);

# petal width
plt.subplot(2, 2, 4)
sns.violinplot(x='Species', y ='PetalWidthCm', data=iris, inner='quartile', palette='Paired')
plt.title('Petal width for Different Species', fontsize=12)
plt.xlabel('Species', fontsize=10)
plt.ylabel('Petal Width(cm)', fontsize=10);


# In[ ]:


# Visualize relationship between parameters
g = sns.PairGrid(data=iris, vars=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], hue='Species', size=3, palette='Set2')
g.map_diag(sns.kdeplot)
g.map_offdiag(plt.scatter)
g.add_legend();


# In[ ]:


# plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(iris.drop('Id', axis=1).corr(), cmap='YlGnBu', annot=True, fmt='.2f', vmin=0);


# # Model Selection

# ## PCA
# Since the sepal length, and petal size are highly correlated, and they are not correlated with the sepal width, maybe it's worth to use PCA to create latent feaures. 

# In[ ]:


features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = iris[features]
y = iris['Species']

# PCA is affected by scale, scale the dataset first
from sklearn import preprocessing 
# Standardizing the features
X = preprocessing.StandardScaler().fit_transform(X)


# In[ ]:


# Set up 2 pca components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X)
print(pca.explained_variance_ratio_)


# In[ ]:


# get the new df
df = pd.DataFrame(data = components, columns = ['pc1', 'pc2'])
df = pd.concat([df, iris[['Species']]], axis = 1)


# get dummy value
le = preprocessing.LabelEncoder()
le.fit(df['Species'])
df['target']=le.transform(df['Species'])

df.head()


# In[ ]:


# visualize the new df
g = sns.FacetGrid(data=df, hue='Species', palette='Set2', size=8)
g.map(plt.scatter, 'pc1', 'pc2')
g.add_legend();


# In[ ]:


# get training and testing data
from sklearn.model_selection import train_test_split
X = df[['pc1', 'pc2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


X = df[['pc1', 'pc2']]
y = df['target']
x_min, x_max = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
y_min, y_max = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf_nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
Z


# In[ ]:


# visualize function
def DecisionBoundary(clf):
    X = df[['pc1', 'pc2']]
    y = df['target']
    
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    #Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
    # Plot also the training points
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


# ## Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# fit the model
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)

pred_nb = clf_nb.predict(X_test)

# get the accuracy score
acc_nb = accuracy_score(pred_nb, y_test)
print(acc_nb)


# In[ ]:


# Visualize the model
DecisionBoundary(clf_nb)


# ## SVM

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}
# fit the model, use gridsearch to choose the best params
svr = SVC()
clf_svc = GridSearchCV(svr, parameters)
clf_svc.fit(X_train, y_train)
clf_svc.best_params_


# In[ ]:


# fit with best parameter
svr2 = SVC(kernel='rbf', C=1)
svr2.fit(X_train, y_train)

pred_svr = svr.predict(X_test)
# get the accuracy score
acc_svr = accuracy_score(pred_svr, y_test)
print(acc_svr)


# In[ ]:


# Visualize the model
DecisionBoundary(svr2)


# ## Decision Tree

# In[ ]:


# fit the decision tree model
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier(min_samples_split=8)
clf_tree.fit(X_train, y_train)

pred_tree = clf_tree.predict(X_test)

# get the accuracy score
acc_tree = accuracy_score(pred_tree, y_test)
print(acc_tree)


# In[ ]:


DecisionBoundary(clf_tree)


# ## Random Forest
# The decision tree model show some overfiting, maybe we can try random forest to minimize the overfiting.

# In[ ]:


# fit the model
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
clf_rf.fit(X_train, y_train)

pred_rf = clf_tree.predict(X_test)

# get the accuracy score
acc_rf = accuracy_score(pred_rf, y_test)
print(acc_rf)


# In[ ]:


DecisionBoundary(clf_rf)


# ## Logistic Regression

# In[ ]:


# fit logistic regression model
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, y_train)

pred_lm = lm.predict(X_test)
acc_lm = accuracy_score(pred_lm, y_test)
print(acc_lm)


# In[ ]:


DecisionBoundary(lm)


# ## K-nearest neighbours

# In[ ]:


# fit the model
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=5) 
clf_knn.fit(X_train,y_train) 

pred_knn = clf_knn.predict(X_test) 
acc_knn = accuracy_score(pred_knn, y_test)
print(acc_knn)


# In[ ]:


DecisionBoundary(clf_knn)


# # Conclusion

# In[ ]:


data_dict = {'Naive Bayes': [acc_nb], 'SVM': [acc_svr], 'DT': [acc_tree], 'Random Forest': [acc_rf], 'Logistic Regression': [acc_lm], 'K_nearest Neighbors': [acc_knn]}
df_c = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Accuracy Score'])
print(df_c)

From the accuracy score, we can see that Naive Bayes and K_nearest Neighbors predict the best. Next let's take a closer look at the recall and precision of these two methods.
# In[ ]:


# recall and precision
from sklearn.metrics import recall_score, precision_score

# params for nb 
recall_nb = recall_score(pred_nb, y_test, average = None)
precision_nb = precision_score(pred_nb, y_test, average = None)
print('precision score for naive bayes: {}\n recall score for naive bayes:{}'.format(precision_nb, recall_nb))


# In[ ]:


# params for knn
recall_knn = recall_score(pred_knn, y_test, average = None)
precision_knn = precision_score(pred_knn, y_test, average = None)
print('precision score for naive bayes: {}\n recall score for naive bayes:{}'.format(precision_knn, recall_knn))


# In general, naive bayes seems to yield a better result! But we can tweak the parameters more to see wheterh there's a better solution

# In[ ]:


DecisionBoundary(clf_nb)


# In[ ]:




