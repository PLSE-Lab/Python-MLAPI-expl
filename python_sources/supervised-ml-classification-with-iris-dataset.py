#!/usr/bin/env python
# coding: utf-8

# # 1. Import Libraries

# ## Check version of libraries

# In[275]:


# Python version
import sys
print('Python: {}'.format(sys.version))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# seaborn
import seaborn
print('seaborn: {}'.format(seaborn.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[276]:


# Display plots within notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Import required python libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  # For making our own cmap
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# Classification Algorithms - 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# Hide FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[277]:


# set seaborn for attractive plots
sns.set(style='darkgrid')


# # 2. Load Dataset

# In[278]:


# Load dataset into dataframe
path = '../input/Iris.csv'
feature_labels = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris_df = pd.read_csv(path)

#Drop Id column and change column names to desired names in above list
iris_df = iris_df.drop(columns='Id')
iris_df.columns = feature_labels

# keep unique classes
classes = iris_df['class'].unique()


# # 3. Summarise Data
# 
# Lets take a quick look at the data by - 
# 1. Dimension.
# 2. Peeping at the data itself.
# 3. Info of the dataframe.
# 4. Looking at the statistics of data.

# ## 3.1 Dimension of data

# In[279]:


iris_df.shape


# ## 3.2 Peeping at the data itself

# In[280]:


iris_df.head(5)


# ## 3.3 Info of data

# In[281]:


iris_df.info()


# ## 3.4 Statistics of data

# In[282]:


iris_df.describe()


# # 4. Data Visualisation
# 
# Now, lets visualise data  by -
# 1. <B>Univariate plot</B> to better understand each feature.
# 2. <B>Multivariate plot</B> to understand relationship between features.

# ## 4.1 Univariate Plot
# 
# 1. Histplot<br /> 
# 2. Boxplot<br /> 
# 3. Violinplot<br /> 

# ### 4.1.1 Histplot

# In[283]:


plt.figure(figsize=(12,8))
for i in range(1, len(feature_labels)):
    plt.subplot(2, 2, i)
    sns.distplot(iris_df[feature_labels[i-1]], kde=False)
    
plt.subplots_adjust(wspace=0.2, hspace=0.3)


# ### 4.1.2 Boxplot

# In[284]:


plt.figure(figsize=(6,5))
sns.boxplot(data=iris_df)


# In[285]:


# Boxplot variant of violinplot makes, its same plot as violinplots as below but in box format
# plt.figure(figsize=(12,10))
# for i in range(1, len(feature_labels)):
#     plt.subplot(2, 2, i)
#     sns.boxplot(x='class', y=feature_labels[i-1], data=iris_df)
    
# plt.subplots_adjust(wspace=0.2, hspace=0.2)


# ### 4.1.3 Violinplot
# 
# It is similar to boxplot, but it also gives us <B>kernel density plot</B>.<br/>
# It consists boxplot in the center of kernel density plot.

# In[286]:


# Violinplot variant of boxplot
# plt.figure(figsize=(6,5))
# sns.violinplot(data=iris_df)


# In[287]:


plt.figure(figsize=(12,8))
for i in range(1, len(feature_labels)):
    plt.subplot(2, 2, i)
    sns.violinplot(x='class', y=feature_labels[i-1], data=iris_df)
    
plt.subplots_adjust(wspace=0.2, hspace=0.3)


# ## 4.2 Multivariate Plot
# 
# 1. Pairplot

# ### 4.2.1 Pairplot

# In[288]:


sns.pairplot(iris_df, hue='class')


# # 5. Prepare Data
# 
# Now, after having a close look at the data, lets perpare it for classification.<br/>
# We will prepare data by following steps - 
# 1. Test-Train Split.
# 2. Feature Engineering.

# ## 5.1 Train-Test Split
# 
# Splitting dataset into training and testing as - 80% for training and 20% for testing our classifers.

# In[289]:


test_size = 0.2
seed = 7

features_train, features_test, labels_train, labels_test = train_test_split(np.float64(iris_df.values[:, 0:4]), iris_df['class'], test_size=0.2, random_state=seed)


# ## 5.2 Feature Engineering/Preprocessing
# 
# This step involvs - <br/>
# 1. Feature Scaling.<br/>
# 2. PCA.<br/>
# 3. Feature Selection.<br/>
# 
# 

# ### 5.2.1 Feature Scaling

# In[290]:


scaler = StandardScaler()
scaler.fit(features_train)
scaled_features_train = scaler.transform(features_train)
scaled_features_test = scaler.transform(features_test)

#### Visualise Boxplot for each scaled feature
scaled_df = pd.DataFrame(data=np.array(scaled_features_train), columns=feature_labels[0:4])
# scaled_df[feature_labels[4]] = le.inverse_transform(labels_train)
plt.figure(figsize=(6, 5))
sns.boxplot(data=scaled_df)


# ### 5.2.3 Feature Selection
# 
# Now, lets select two features (mainly because we want to visualise our classifier).<br/><br/>
# Feature selection is quite beneficial as it - 
# 1. Reduces overfitting.
# 2. Improves accuracy.
# 3. Reduces training time.
# <br/><br/>
# We might not need feature selection on this dataset as we have only 4 features, but since I want to visualize my trained classifiers, I'm gonna select two features.<br/>
# 
# There are many ways to decide which features to use, Univariate Selection, Feature Importance, Correlation matrix etc. But we're gonna visualise and use SelectKBest(Univariate Selection) and Correlation matrix(its just a bonus!).

# In[291]:


# SelectKBest using f_classif (ANOVA Test) as score function
bestfeatures = SelectKBest(score_func=f_classif, k=2)
fit = bestfeatures.fit(scaled_features_train, labels_train)

scores_df = pd.DataFrame(data=fit.scores_)
columns_df = pd.DataFrame(data=iris_df.columns)

feature_scores_df = pd.concat([columns_df,scores_df],axis=1)
feature_scores_df.columns = ['Features','Score']

# print(feature_scores_df.nlargest(4,'Score'))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Score', y='Features', order=feature_scores_df.nlargest(4,'Score')['Features'], data=feature_scores_df, palette=sns.cubehelix_palette(n_colors=4, reverse=True))
plt.subplot(1, 2, 2)
sns.heatmap(iris_df.corr(), annot=True, cmap=sns.cubehelix_palette(start=0, as_cmap=True))


# Based on our SelectKBest scores and Correlation Matrix, we'll select and 2 best features to build our classifier, i.e, 'petal-length' and 'petal-width'.

# In[292]:


transformed_features_train = scaled_features_train[:, [2, 3]]
transformed_features_test = scaled_features_test[:, [2, 3]]


# # 6. Evaluate Some Algorithms
# 
# Now, its time to build and evaluate some supervised leaning classifiers, by - 
# 1. Build classifier.
# 2. Validation.
# 3. Select best classifier.

# ## 6.1 Build Classifiers
# 
# We will build 7 classifiers, namely - 
# 1. Gaussian Naive Bayes (NB).
# 2. SVM (SVC).
# 3. KNN (KNN).
# 4. Decision Tree (CART).
# 5. Bagging (BAG).
# 6. Random Forest (RF).
# 7. AdaBoost (AB).

# In[293]:


clf_names = ['NB', 'SVC', 'KNN', 'CART', 'BAG', 'RF', 'AB']

models = [
    ('NB', GaussianNB()),
    ('SVC', SVC(C=1000, kernel='rbf', gamma=0.05)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('CART', DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=5)),
    ('BAG', BaggingClassifier(n_estimators=100)),
    ('RF', RandomForestClassifier(n_estimators=100, min_samples_split=20, min_samples_leaf=5)),
    ('AB', AdaBoostClassifier(n_estimators=100))
]


# ## 6.2 Validation
# 
# Lets use cross validation to estimate accuracy of all our predictive models using <B>10-fold cross validator</B> with <B>accuracy</B> scoring.<br/><br/>
# A 10-fold cv will divide our training set into 9 set for training our classifiers and retain 1 set for validation/testing. It will repeat this for all(10) possible sets of train and validate data set and will calculate the average accuracy over accuracy for each train-test set.

# In[294]:


scoring = 'accuracy'

cv_results = []
for name, clf in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_score = cross_val_score(clf, transformed_features_train, labels_train, cv=kfold, scoring=scoring)
    cv_results.append(cv_score)
    print('{}: {}, ({})' .format(name, cv_score.mean(), cv_score.std()))


# ## 6.3 Select Best Classifier
# 
# We have an estimate accuracy for all our classifiers. We can see that the SVC and CART outperformes every other classifier with an accuracy of about 97.5%. This makes us little suspicious to overfitting which will be confirmed when predicting.<br/>
# For now lets - 
# 1. Compare our CV results for each classifier using boxplot.
# 2. Compare classifier using contour plot (Just for fun!).

# ### 6.3.1 Boxplot

# In[295]:


plt.figure(figsize=(6, 5))
sns.boxplot(data=pd.DataFrame(data=np.array(cv_results).transpose(), columns=clf_names))

# #### Alternate choice for boxplot - univariateplot -> boxplot + violinplot
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 2, 1)
# sns.boxplot(data=pd.DataFrame(data=np.array(cv_results).transpose(), columns=clf_names))

# plt.subplot(1, 2, 2)
# sns.violinplot(data=pd.DataFrame(data=np.array(cv_results).transpose(), columns=clf_names))


# ### 6.3.2 Classifier' Contour Plots

# In[296]:


x = transformed_features_train[:, [0]].ravel()
y = transformed_features_train[:, [1]].ravel()

le = LabelEncoder()
le.fit(classes)

# Encode all classes of Iris Flower species to values [0, 1, 2] to plot contour
target_labels_encoded = le.transform(iris_df['class'].ravel())
labels_train_encoded = le.transform(labels_train)

# color sequence
c = labels_train_encoded

models_ = models.copy()
trained_models = [(name, clf.fit(np.array(transformed_features_train), c)) for name, clf in models_]

titles = ('Input Data',
          'Gaussian NB',
          'SVM with RBF kernel',
          'KNN',
          'Decision Tree',
          'Bagging',
          'Random Forest',
          'AdaBoost',
         )

def mesh_grid(x, y, h=0.02):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    return xx, yy

def plot_contour(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return ax.contourf(xx, yy, Z, **params)

xx, yy = mesh_grid(x, y)

# custom cmap
my_cmap = ListedColormap(sns.color_palette().as_hex()[0:3])

fig, sub = plt.subplots(2, 4, figsize=(20, 8))

sub[0, 0].scatter(iris_df.values[:, [2]].ravel(), iris_df.values[:, [3]].ravel(), c=target_labels_encoded, cmap=my_cmap, s=20)
sub[0, 0].set_title(titles[0])

for clf, title, ax in zip(trained_models, titles[1:], sub.flatten()[1:]):
        plot_contour(ax, clf[1], xx, yy, cmap=my_cmap, alpha=1)
        ax.scatter(x, y, c=c, cmap=my_cmap, s=25, edgecolor='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Petal length')
        ax.set_ylabel('Petal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

plt.show()


# # 7. Make Prediction
# 
# KNN is a quite simple algorithm and has also perfomed well in validation. Now lets see how it performs on out test set.<br/> <br/>
# Testing our classifier will give us an independent final check on the selected best model.<br/><br/>
# We can run the KNN directly on the test set and summarize the results as accuracy score, confusion matrix and classification report.

# In[297]:


# Selecting and predicting using KNN
clf = models[2][1]
clf.fit(transformed_features_train, labels_train)
predictions = clf.predict(transformed_features_test)

print(accuracy_score(labels_test, predictions))
print(confusion_matrix(labels_test, predictions))
print(classification_report(labels_test, predictions))


# We can see that our classifier has an accuracy of 90% on the testing set. The confusion matrix tells us that it made three errors and the classification report gives the result of classification for each class. The overall result seems good.
