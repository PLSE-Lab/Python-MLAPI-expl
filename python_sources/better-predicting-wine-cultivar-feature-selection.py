#!/usr/bin/env python
# coding: utf-8

# # Better Predicting Wine Cultivar with Feature Selection
# 
# In supervised machine learning (ML) the goal is to have an accurate model.
# This which based on previously tagged data provides predictions for new data.
# 
# The number one question when it comes to modeling is:
# **"How can I improve my results?"**
# 
# There are several basic ways to improve your prediction model:
# 1. Hyperparameters optimization
# 2. Feature extraction
# 3. Selecting another model
# 4. Adding more data
# 5. Feature selection
# 
# In this blog post, I'll walk you through how I used **Feature Selection** to improve my model.
# For the demonstration I'll use the ['Wine' dataset from UCI ML repository](https://archive.ics.uci.edu/ml/datasets/Wine), which was also available [here at kaggle](https://www.kaggle.com/brynja/wineuci)
# 
# Most of the functions are from the [sklearn (scikit-learn) module](http://scikit-learn.org/).
# 
# For the plotting functions make sure to read about [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org).
# Both are great plotting modules with great documentation.
# 
# Before we jump into the ML model and prediction we need to understand our data.
# The process of understanding the data is called EDA - exploratory data analysis.
# 
# ### EDA - Exploratory Data Analysis.
# UCI kindly gave us some basic information about the data set.
# I'll quote some of the more important info given:
# "These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines ... All attributes are continuous ... 1st attribute is class identifier (1-3)"
# 
# Based on this, it seems like a classification problem with 3 class labels and 13 numeric attributes.
# A classification problem with the goal of predicting the specific cultivar the wine was derived from.

# In[ ]:


# Loading a few important modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set() #sets a style for the seaborn plots.
np.random.seed(64)


# In[ ]:


columns_names = ['Target', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids',
                 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline']


# In[ ]:


# Loading the data from it's csv,
# and converting the 'Target' column to be a string so pandas won't infer it as a numeric value
data = pd.read_csv(os.path.join('..', 'input', 'Wine.csv'), header=None)
data.columns = columns_names
data['Target'] = data['Target'].astype(str)
data.head() # print the data's top five instances


# I named the first columns as 'Target'.
# This is the target attribute - what we are trying to predict.
# This is a classification problem, so the class label ('Target') is not a numeric but a nominal value. that's why I'm telling Pandas this columns dtype is 'str'.

# In[ ]:


data.info() # prints out a basic information about the data.


# As we can see we have 178 entries (instances).
# as we know from UCI's description of the data, we have 13 numeric attributes and one 'object' type attribute (which is the target column).
# all the columns of all the rows have data, therefore we see "178 non-null" next to every column description.

# In[ ]:


sns.countplot(data['Target']);


# It's important to check the amount of instances in each class. There is difference between the class labels but It isn't a huge difference.
# If the difference was bigger we would be in an imbalanced problem.
# That would require a lot of other things to do, but this is for another post.

# In[ ]:


# This method prints us some summary statistics for each column in our data.
data.describe()


# This is probably only informative to people who have some experience in statistics.
# Let's try to plot this information and see if it helps us understand.

# In[ ]:


# box plots are best for plotting summary statistics.
sns.boxplot(data=data);


# Unfortunately this is not a very informative plot becasue the data is not in the same value range.
# We can resolve the problem by plotting each column side by side.

# In[ ]:


data_to_plot = data.iloc[:, 1:]
fig, ax = plt.subplots(ncols=len(data_to_plot.columns))
plt.subplots_adjust(right=3, wspace=1)
for i, col in enumerate(data_to_plot.columns):
    sns.boxplot(y=data_to_plot[col], ax = ax[i]);


# This is a better way to plot the data.
# 
# We can see that we have some outliers (based on the [IQR calculation](https://en.wikipedia.org/wiki/Interquartile_range)) in almost all the feaures.
# These outliers deserve a second look, but we won't deal with them right now.

# Pair plot is a great way to see a scatter plot of all the data, of course only for two features at a time.
# Pair plot is good for small amout of features and for first glance at the columns (features), afterwords in my opinion a simple scatterplot with the relevant columns is better.

# In[ ]:


columns_to_plot = list(data.columns)
columns_to_plot.remove('Target')
sns.pairplot(data, hue='Target', vars=columns_to_plot);
# the hue parameter colors data instances baces on their value in the 'Target' column.


# The diagonal line from the top left side to the right buttom side of the pair plot are histograms of the columns.
# 
# Good feature combination for me is a feature combination that separates some of the class labels.
# 'Flavanoids' in row 7 looks like a good feature combined with the other ones. Same goes for 'Proline' in the last row.
# 
# On the other hand 'Malic_acid' (2nd row) does not look like a good feature at all.
# 
# We can have a further look at some features:

# In[ ]:


sns.lmplot(x='Proline', y='Flavanoids', hue='Target', data=data, fit_reg=False);


# In[ ]:


sns.lmplot(x='Hue', y='Flavanoids', hue='Target', data=data, fit_reg=False);


# In[ ]:


# This is a good feature comination to separate the red ones (label 3)
sns.lmplot(x='Color_intensity', y='Flavanoids', hue='Target', data=data, fit_reg=False);


# In[ ]:


sns.boxplot(x=data['Target'], y=data['OD280/OD315_of_diluted_wines']);
# this is a vey good feature to separate label 1 and 3


# Another thing that is good to check is the feature correlation. We don't like features that correlate with each other so much.
# 
# We will use the [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) to compute pairwise correlation of columns in our data.
# It's worth to emphasize that the Pearson correlation is only good for linear correlation, but as we saw from the pair plot, our data dosen't seem to correlate in any other way.

# In[ ]:


plt.figure(figsize=(18,15))
sns.heatmap(data.corr(), annot=True, fmt=".1f");


# There is a correlation between Total_phenols - Flavanoids => 0.9 which is strong.
# Typically I would delete one of the correlting features, but for now I won't do it.
# Let's plot these feautes to see the correlation.

# In[ ]:


sns.lmplot(x='Total_phenols', y ='Flavanoids', data=data, fit_reg=True);


# ## Modelling and Predicting

# As I mentioned before there are several ways in which you can improve your ML model.
# 
# Today I'll focus on **feature selection**, a very basic way to improve your model's score.

# Let's load the 'train_test_split' function, and separate our data into only the feature vectors and the target vector.
# We will split our data into 25% test and 75% train.
# 
# The 'stratify' parameter will ensure equal distribution of subgroups.
# It will keep the ration between the classes in the train and test data as they were in the actual full data.

# In[ ]:


from sklearn.model_selection import train_test_split
np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.

X = data.drop('Target', axis=1)
y = data['Target']
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)


# I'll start with a very simple classifier called [knn (k-nearest neighbors)](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
# 
# Knn classifies an object (a new data point) by the majority vote of its k nearest neighbors in the feature space.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.

model = KNeighborsClassifier()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print('score on training set:', model.score(x_train, y_train))
print('score on test set:', model.score(x_test, y_test))
print(metrics.classification_report(y_true=y_test, y_pred=pred))


# As we can see this is a pretty bad result.
# 
# The key question which will help us decide in which way we should improve our model is "whether our model is [overfitting or underfitting](https://en.wikipedia.org/wiki/Overfitting)?"
# 
# 
# 
# 
# Before I'll answer this question, there is something we didn't do and it's essential in this case:
# When we are using KNN or any other algorithm which is based on distances (like the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) in our case), normalization of the data is necessery.
# I'll be using the mean normaliztion method, which is subtracting the feature's mean and dividing by it's standard deviation. This is basically converting each data point into it's [Z-score](https://en.wikipedia.org/wiki/Standard_score).

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.

model = Pipeline(
    [
        ('scaler', StandardScaler()), # mean normalization
        ('knn', KNeighborsClassifier(n_neighbors=1))
    ]
)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print('score on training set:', model.score(x_train, y_train))
print('score on test set:', model.score(x_test, y_test))
print(metrics.classification_report(y_true=y_test, y_pred=pred))


# Great improvment!
# 
# We can use a "Pipeline" in this part. "Pipeline" is a function in sklearn that combines several other functions and enables us to use other sklearn functions with only one fit command. More on this you can read [here](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [here](http://scikit-learn.org/stable/modules/pipeline.html) and in a future blog post.
# 
# 
# So as I mentioned before, we need to understand if our mode is over or under fitting.

# In[ ]:


from sklearn.model_selection import learning_curve
np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.

def over_underfit_plot(model, X, y):
    plt.figure()

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score");
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.yticks(sorted(set(np.append(train_scores_mean, test_scores_mean))))
    
over_underfit_plot(model, x_train, y_train)


# As we can see from the plot and from the accuracy scores, we are in an overfitting situation - We have a good score on the train, but a low score on the test and by adding data we improve the model's results.
# This means the model is bad for unseen data and very good with the data it was trained on.
# 
# In an overfitting case there are a few things which need and can be done to improve our model:
# 1. Add more data - not possible in this case.
# 2. Remove unimportant features in order to make the model less complex - aka. feature selection.
# 3. Add regulization.

# ## Feature selection
# One of the ways to avoid overfitting is by selecting a subset of features from the data.
# There are a lot of ways to do feature selection. The most basic one in my opinion is removing correlating features.
# 
# As we checked before 'Total_phenols' and 'Flavanoids' are closely correlating features. Let's see what happens if we drop one of them!

# In[ ]:


np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.

X.drop('Total_phenols', axis=1, inplace =True) # delete one of the correlating features
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y) # split the data again

#fit the same model again and print the scores
model.fit(x_train, y_train)
pred = model.predict(x_test)
print('score on training set:', model.score(x_train, y_train))
print('score on test set:', model.score(x_test, y_test))
print(metrics.classification_report(y_true=y_test, y_pred=pred))


# Truly as we expected this step improved our model's score, but we are still overfitting.
# 
# Another feature selection method (and my favourite) is by using another algorithm's feature importance.
# Many algorithms rank the importance of the features in the data, based on which feature helped the most to distinguish between the target labels.
# From this ranking we can learn which features were more and less important and select just the one's which contribute the most.
# 
# Let's fit the train data in a [Random Forest classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and print the feature importance scores.
# 
# Random Forest is an ensamble that fits a number of decision tree classifiers. Ensamble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of its learning algorithms alone, in our case from a simple Decision tree.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.

model_feature_importance = RandomForestClassifier(n_estimators=1000).fit(x_train, y_train).feature_importances_
feature_scores = pd.DataFrame({'score':model_feature_importance}, index=list(x_train.columns)).sort_values('score')
sns.barplot(feature_scores['score'], feature_scores.index)


# As we can see from the plot there are features that are more important than others and we can see 5-7 features which stand out.
# I'll use the feature important scores to put a threshold for my model feature selection function.
# 
# ["SelectFromModel"](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html) is a sklearn function which takes an estimator and a threshold, extracts from the estimator the feature importance scores and returns only the features with a score above the given threshold.
# 

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.ensemble import RandomForestClassifier
np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.

model = Pipeline(
    [
        ('select', SelectFromModel(RandomForestClassifier(n_estimators=1000), threshold=0.06)),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=1))
    ]
)

model.fit(x_train, y_train)
pred = model.predict(x_test)
print('score on training set:', model.score(x_train, y_train))
print('score on test set:', model.score(x_test, y_test))
print(metrics.classification_report(y_true=y_test, y_pred=pred))


# An improvment of 8-9% in this high scores is super great and difficult.
# This is a very good score just by itself! Now we can improve our score in other ways.
# 
# 
# My point was to show how I improved my score with this very simple KNN model.
# I also wanted to show you how a few simple steps can improve your model and get a high score.
