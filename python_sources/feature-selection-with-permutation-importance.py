#!/usr/bin/env python
# coding: utf-8

# **Feature Selection with Permutation Importance**

# In machine learning, it is important to explain *why* a given model behaves the way that it does.  One way to explain the behavior of a model is to describe what features were selected and why.  There are many methods for [feature selection](http://scikit-learn.org/stable/modules/feature_selection.html), but in this tutorial we will focus only on the Permutation Importance method.  With the Permutation Importance feature selection method, the performance of a model is tested after removing each individual feature and replacing that feature with random noise.  In this way the importance of individual features can be directly compared, and a quantitative threshold can be used to determine feature inclusion.  In this tutorial we will demonstrate the use of the Permutation Importance feature selection method with the help of the Python library [ELI5](http://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html).

# We will be using the [Wisconsin Breast Cancer](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer) dataset to explore feature selection and permutation importance.  The features in this dataset are numerical measurements of nuclear size and nuclear shape, while the labels refer to the presence or absence of cancer.

# *Step 1: Import Libraries*

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score 
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance
get_ipython().run_line_magic('matplotlib', 'inline')


# *Step 2: Load Data*

# In[ ]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
data = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names, ["target"])
sizeMeasurements = pd.DataFrame(data, columns=columns)
X = sizeMeasurements[sizeMeasurements.columns[:-1]]
y = sizeMeasurements.target
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
print('\n Column Values: \n\n', sizeMeasurements.columns.values, "\n")


# Here you can see the list of features in our dataset.  We know from the biomedical literature that cancer cells often have large and mishapen nuclei.  As such, I suspect that paremeters such as "mean area" and "mean concave points" will be especially good predictors.  

# *Step 3: Plot Data*

# In[ ]:


sns.set_style("whitegrid")
plotOne = sns.FacetGrid(sizeMeasurements, hue="target",aspect=2.5)
plotOne.map(sns.kdeplot,'mean area',shade=True)
plotOne.set(xlim=(0, sizeMeasurements['mean area'].max()))
plotOne.add_legend()
plotOne.set_axis_labels('mean area', 'Proportion')
plotOne.fig.suptitle('Area vs Diagnosis (Blue = Malignant; Orange = Benign)')
plt.show()

sns.set_style("whitegrid")
plotTwo = sns.FacetGrid(sizeMeasurements, hue="target",aspect=2.5)
plotTwo.map(sns.kdeplot,'mean concave points',shade= True)
plotTwo.set(xlim=(0, sizeMeasurements['mean concave points'].max()))
plotTwo.add_legend()
plotTwo.set_axis_labels('mean concave points', 'Proportion')
plotTwo.fig.suptitle('# of Concave Points vs Diagnosis (Blue = Malignant; Orange = Benign)')
plt.show()


# The first plot confirms my prediction that healthy nuclei have a default size and that cancer cells have a wide range of sizes, typically greater than the default size.  Likewise, the second plot confirms my prediction that healthy nuclei are typically circular/elliptical and that cancer cells are mishapen and have a lot of concave points.  These first two plots are good news and suggest that we will indeed have success in this task of predicting cancer based off of measurements of nuclear shape. 

# Next we will test a Random Forest Classification strategy, but first I will need to define some helper functions that can be used to evaluate our results (learning curve + confusion matrix).

# *Step 4: Define Helper Functions*

# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

dict_characters = {0: 'Malignant', 1: 'Benign'}


# *Step 5: Evaluate Random Forest Classifier*

# In[ ]:


def runRandomForest(a, b, c, d):
    model = RandomForestClassifier()
    accuracy_scorer = make_scorer(accuracy_score)
    model.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(model, c, d, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    prediction = model.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plot_learning_curve(model, 'Learning Curve For RandomForestClassifier', a, b, (0.80,1.1), 10)
    plt.show()
    plot_confusion_matrix(cnf_matrix, classes=dict_characters,title='Confusion matrix')
    plt.show()
    print('Random Forest Classifier - Training set accuracy: %s (%s)' % (mean, stdev))
    return
runRandomForest(X_train, Y_train, X_test, Y_test)


# The performance of our Random Forest Classifier was quite good even before feature selection.  Next let's evaluate the importance of each of the features that were used.

# In[ ]:


model = RandomForestClassifier()
model.fit(X_train,Y_train)
columns = X_train.columns
coefficients = model.feature_importances_.reshape(X_train.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('RandomForestClassifier - Feature Importance:')
print('\n',fullList,'\n')


# Here you can see that the most important features were indeed the features that best describe the size and shape of the nuclei (e.g. area, radius, concavity, etc). ** Now let's try to use the Permutation Importance feature selection method in order to reduce the number of features and hopefully improve our predictive performance. ** With the Permutation Importance feature selection method, the performance of a model is tested after removing each individual feature and replacing that feature with random noise.  In this way the importance of individual features can be directly compared, and a quantitative threshold can be used to determine feature inclusion.

# In[ ]:


X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
Y_train=np.asarray(Y_train)
Y_test=np.asarray(Y_test)

# BEGIN: FEATURE SELECTION WITH PERMUTATION IMPORTANCE METHOD
#http://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
sel = SelectFromModel(PermutationImportance(RandomForestClassifier(), cv=5),threshold=0.005,).fit(X_train, Y_train)
X_train2 = sel.transform(X_train)
X_test2 = sel.transform(X_test)
# END: FEATURE SELECTION WITH PERMUTATION IMPORTANCE METHOD

runRandomForest(X_train2, Y_train, X_test2, Y_test)


# Here we can see a slightly improved performace despite dropping the majority of our features.  Next let's evaluate what features were selected and let's compare the relative importance (i.e. coefficient) that is assigned to each one.

# In[ ]:


model = RandomForestClassifier()
model.fit(X_train2,Y_train) # Needed to initialize coef_ or feature_importances_
coefficients = model.feature_importances_
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('RandomForestClassifier - Feature Importance:')
print('\n',fullList,'\n')


# We now have a model that has many fewer features and is therefore much easier to explain.  As expected, the most important features are direct measurements of nuclear size and nuclear shape.
# 
# By removing features we were able to both increase the performance of our model and improve our model's explainability.  Methods such as this can be used to dispell the notion of machine learning algorithms as irreproducible "black-boxes" and can be used to help gain new insights from machine learning models.

# For more information, please consult the relevant [publication](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) and [documentation](http://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html).

# In[ ]:




