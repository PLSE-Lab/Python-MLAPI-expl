#!/usr/bin/env python
# coding: utf-8

# The purpose of this project is to predict whether a mammogram mass is benign or malignant.
# After preparing data, several algorithms will be applied for classification purposes, and their parformance accuracies will follow.
# In addition, I will show the appliaction of PCA to visualize our data, as well as ROC-AUC will be presented for a list of algorithms.
# 
# We'll be using the "mammographic masses" public dataset from the UCI repository (source: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)
# This data contains 961 instances of masses detected in mammograms, and contains the following attributes:
# 
# 1. BI-RADS assessment: 1 to 5 (ordinal)
# 2. Age: patient's age in years (integer)
# 3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
# 4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
# 5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
# 6. Severity: benign=0 or malignant=1 (binominal)
# 
# BI-RADS is an assesment of how confident the severity classification is. As this is not a "predictive" attribute we will discard it.
# The age, shape, margin, and density attributes are the features that we will build our model with, and "severity" is the classification we will attempt to predict based on those attributes.
# Note that "shape" and "margin" are nominal data types. Since they are close enough to ordinal we will not discard them. The "shape" for example is ordered increasingly from round to irregular.
# 
# We will use the following algorithms, and assess their performance on test data with and without K-Fold validation:
# 1. Decision tree
# 2. Random forest
# 3. KNN
# 4. Naive Bayes
# 5. SVM
# 6. Logistic Regression
# 

# Here is the top 5 rows of our data.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

col_names=['BI_RADS', 'age', 'shape', 'margin', 'density','severity']
file=pd.read_csv("../input/mammographic_masses.data.txt")
file.head(5)


# The top 5 rows of our data show that we need to add column names as well as to deal with "?". After doing this, it's the time to go over main statistics of the data.

# In[ ]:


file=pd.read_csv("../input/mammographic_masses.data.txt", na_values='?', names=col_names, usecols=range(1,6))
file.describe(include=("all"))


# Let's see if there are null values and drop them.

# In[ ]:


print("Number of NULL values per feature:")
file.isnull().sum()


# In[ ]:


file.dropna(inplace=True)
file.shape
file. describe(include=("all"))


# Seems there are no outliers. Let's see how each of the features is distributed.

# In[ ]:


fig, axes = plt.subplots(1,4, sharey=False, figsize=(18,4))
ax1, ax2, ax3, ax4 = axes.flatten()

ax1.hist(file['age'], bins=10, color="lightslategray")
ax2.hist(file['shape'], bins=4, color="steelblue")
ax3.hist(file['margin'], bins=5, color="mediumslateblue")
ax4.hist(file['density'], bins=4, color="darkslategray")
ax1.set_xlabel('AGE', fontsize="large")
ax2.set_xlabel('SHAPE', fontsize="large")
ax3.set_xlabel('MARGIN', fontsize="large")
ax4.set_xlabel('DENSITY', fontsize="large")
ax1.set_ylabel("AMOUNT", fontsize="large")

plt.suptitle('COMPARISON of DISTRIBUTIONS', ha='center', fontsize='x-large')
plt.show()


# In[ ]:


feature_names=['age', 'shape', 'margin', 'density']
features=file[['age', 'shape', 'margin', 'density']].values
classes=file['severity'].values


# We have different scales for the features. As some of the algorithms require a prior normalization of input data, let's normalize our data.

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
features_scaled=scaler.fit_transform(features)
print("Scaled features:")
features_scaled


# In order to train and test our data, let's split it into two parts: 80%-20%.
# Then we'll apply the above mentioned models to see which of them performs better with selected parameters.

# In[ ]:


from sklearn.model_selection import train_test_split
train_f, test_f, train_c, test_c=train_test_split(features_scaled, classes, test_size=0.2, random_state=0)


# In[ ]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dicti={SVC(kernel="rbf", C=1, gamma=1000, probability=True):"svc",
    LogisticRegression(solver="liblinear", random_state=0):"lr",
    KNeighborsClassifier(n_neighbors=10):'knn',
    RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0):'rfc',
    DecisionTreeClassifier(random_state=0):'dtc'}
for model in dicti:
    model.fit(train_f, train_c)
    pred_c=model.predict(test_f)
    accc=accuracy_score(test_c, pred_c)
    print("Accuracy score for ", dicti[model], " is ", accc.round(2))


# Here the winner was Logistic regression. However, the results may change after applying K-Fold cross validation, because, the latter  generally results in a less biased or less optimistic estimate of the model skill than other methods, such as a simple train/test split. The mean of these K scores is consider as the ultimate accuracy score for the model.

# In[ ]:


from sklearn.model_selection import cross_val_score
for model in dicti:
    score=cross_val_score(model,features_scaled,classes, cv=10)
    print("Accuracy score for ", dicti[model], "with cros. val. is ",'{:3.2f}'.format(score.mean()))


# After applying cross validation, Logistic Regression succeeds more.
# Now let's try K-Nearest-Neighbors and see if the values of K plays a significant role in the accuracy of prediction.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
for n in range(1,11):
    model=KNeighborsClassifier(n_neighbors=n)
    model.fit(train_f, train_c)
    pred_c=model.predict(test_f)
    acc=accuracy_score(test_c, pred_c)
    print(n,"neighbor(s):")
    print("Accuracy score for KNN is :", acc.round(2))
    score=cross_val_score(model,train_f,train_c, cv=10)
    print("Accuracy score for KNN with cros. val. is ",'{:3.2f}'.format(score.mean()),"\n")


# The last model to check is Naive Bayes. For this model let's apply MinMaxScaler to have all the features' values between 0 and 1. The result of this shows that Naive Bayes does not perform better than Logistic regression.

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
MNB=MultinomialNB()
features_minmax=scaler.fit_transform(features)
cv_scores=cross_val_score(MNB,features_minmax, classes, cv=10)
print("Accuracy score for Multinomial Naive Bayes with cross. val. is ",'{:3.2f}'.format(score.mean()))


# **<font size="4">ROC and AUC for different models</font>**
#     
# ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters: 
# 1. True Positive Rate 
# 2. False Positive Rate 
# 
# True Positive Rate = True Positive / (True Positive + False Negative)
# False Positive Rate = False Positive / (False Positive + True Negative)
# 
# AUC measures the entire two-dimensional area underneath the entire ROC curve from (0,0) to (1,1). One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example. So, the higher the area under the curve the better the model is.

# In[ ]:


from sklearn.metrics import roc_curve, auc
dicti={SVC(kernel="rbf", C=1, gamma=1000, probability=True):"svc",
    LogisticRegression(solver="liblinear", random_state=0):"lr",
    KNeighborsClassifier(n_neighbors=10):'knc',
    RandomForestClassifier(max_depth=3, n_estimators=100):'rfc',
    DecisionTreeClassifier():'dtc'}
for model in dicti:
    model.fit(train_f,train_c)
    prob=model.predict_proba(test_f)
    fpr, tpr, thresholds=roc_curve(test_c, prob[:,1])
    roc_auc=auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=3, label=dicti[model]+' AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.title('Receiver Operating Characteristic', fontsize=15)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.rcParams["figure.figsize"] = (10,5)
plt.show()


# A curve pulled close to the upper left corner indicates a better performing test. And one can notice that we have higest performance for Logistic Regression again.

# **<font size="4">PCA (Principal Component Analysis)</font>**
# 
# As we have 4 features, we can not visualize our data in 2D. Let's use PCA to reduce dimensionality.

# In[ ]:


from sklearn.decomposition import PCA
pca=PCA()
train_f_pca=pca.fit_transform(train_f)
test_f_pca=pca.transform(test_f)

df = pd.DataFrame({'Variance Explained':pca.explained_variance_ratio_,
             'Principal Components':['PC1','PC2', 'PC3', 'PC4']})
sns.barplot(x='Principal Components',y="Variance Explained", data=df, color="b")
plt.title("Variance Explained by Principal Components\n", fontsize=20, color="b")
plt.show()


# In[ ]:


print("Explained variance per component:")
pca.explained_variance_ratio_.tolist()


# As more than 3/4 of the data variance is explained with two components, let's consider those components in order to visualize data sets in 2D.
# 
# The following two plots show the distribution of Training and Test data.

# In[ ]:


from matplotlib.colors import ListedColormap
pca2 = PCA(2)  # project from 4 to 2 dimensions
train_f_pca2=pca2.fit_transform(train_f)
test_f_pca2=pca2.transform(test_f)
plt.scatter(train_f_pca2[:, 0], train_f_pca2[:, 1], c=train_c, edgecolor='k',s=50, alpha=0.7, cmap=ListedColormap(('g','r')))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title("Visualization of Train Data\n with two components\n", color="r", fontsize=15)
plt.colorbar(label='benign'+" "*15+'malignant')
plt.show()


# In[ ]:


plt.scatter(test_f_pca2[:, 0], test_f_pca2[:, 1], c=test_c, edgecolor='black',s=50, alpha=0.7, cmap=ListedColormap(('g','r')))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('Visualization of Test Data\n with two components')
plt.colorbar(label='beign'+" "*15+'malign')
plt.title("Visualization of Test Data\n with two components\n", color="r", fontsize=15)
plt.show()


#  I also suggest to visualize classifiers' boundaries for two best performing algorithms. Below we have it for Train and Test data separately.

# In[ ]:


from sklearn.model_selection import train_test_split
train_f, test_f, train_c, test_c=train_test_split(features_scaled, classes, test_size=0.2, random_state=0)

classifier =LogisticRegression(solver="liblinear", random_state=0)
classifier.fit(train_f_pca2, train_c)
pred_c = classifier.predict(test_f_pca2)
plt.subplot(2,1,1)

#Train set boundary
X_set, y_set = train_f_pca2, train_c
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.6, cmap = ListedColormap(('green', 'red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],s=10,
                c = ListedColormap(('green', 'red'))(i), label = j)
plt.title('Logistic Rgression\nBoundary Line with PCA (Train Set)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
       
plt.subplot(2,1,2)
#Test set boundary
X_set, y_set = test_f_pca2, test_c
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.6, cmap = ListedColormap(('green', 'red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],s=10, c = ListedColormap(('green', 'red'))(i), label = j)
plt.title('Boundary Line with PCA (Test Set)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
train_f, test_f, train_c, test_c=train_test_split(features_scaled, classes, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(max_depth=3, n_estimators=100)
classifier.fit(train_f_pca2, train_c)
pred_c = classifier.predict(test_f_pca2)
   
plt.subplot(2,1,1)
#Train set boundary
X_set, y_set = train_f_pca2, train_c
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.6, cmap = ListedColormap(('green', 'red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],s=10,
                c = ListedColormap(('green', 'red'))(i), label = j)
plt.title('Randon Forest Classifier\nBoundary Line with PCA (Train Set)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
       
plt.subplot(2,1,2)
#Test set boundary
X_set, y_set = test_f_pca2, test_c
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.6, cmap = ListedColormap(('green', 'red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],s=10, c = ListedColormap(('green', 'red'))(i), label = j)
plt.title('Boundary Line with PCA (Test Set)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.tight_layout()
plt.show()   


# The result is not bad with just two components.
# >As always, your comments are appreciated.
# 
# >I hope you found this kernel useful and informative. If so, please **<font size="4">upvote  :)</font>**  
