#!/usr/bin/env python
# coding: utf-8

# **Predict Diabetes From Medical Records**

# Diabetes mellitus (DM), commonly referred to as diabetes, is a group of metabolic disorders in which there are high blood sugar levels over a prolonged period.  Type 1 diabetes results from the pancreas's failure to produce enough insulin.  Type 2 diabetes begins with insulin resistance, a condition in which cells fail to respond to insulin properly.  As of 2015, an estimated 415 million people had diabetes worldwide, with type 2 diabetes making up about 90% of the cases. This represents 8.3% of the adult population.    Source: https://en.wikipedia.org/wiki/Diabetes_mellitus
# 
# The [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) can be used to train machine learning models to predict if a given patient has diabetes.  This dataset contains measurements relating to Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.  
# 

# *Step 1: Import Python Packages*

# In[1]:


import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import graphviz 
from sklearn import model_selection
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# *Step 2: Define Helper Functions*

# In[2]:


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

def compareABunchOfDifferentModelsAccuracy(a, b, c, d):
    """
    compare performance of classifiers on X_train, X_test, Y_train, Y_test
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
    """    
    print('\nCompare Multiple Classifiers: \n')
    print('K-Fold Cross-Validation Accuracy: \n')
    names = []
    models = []
    resultsAccuracy = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('LSVM', LinearSVC()))
    models.append(('GNB', GaussianNB()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    for name, model in models:
        model.fit(a, b)
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        accuracy_results = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
        resultsAccuracy.append(accuracy_results)
        names.append(name)
        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(), accuracy_results.std())
        print(accuracyMessage) 
    # Boxplot
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison: Accuracy')
    ax = fig.add_subplot(111)
    plt.boxplot(resultsAccuracy)
    ax.set_xticklabels(names)
    ax.set_ylabel('Cross-Validation: Accuracy Score')
    plt.show()    
      
def defineModels():
    print('\nLR = LogisticRegression')
    print('RF = RandomForestClassifier')
    print('KNN = KNeighborsClassifier')
    print('SVM = Support Vector Machine SVC')
    print('LSVM = LinearSVC')
    print('GNB = GaussianNB')
    print('DTC = DecisionTreeClassifier')
    print('GBC = GradientBoostingClassifier \n\n')

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "MLPClassifier", "AdaBoost",
         "Naive Bayes", "QDA"]    

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

dict_characters = {0: 'Healthy', 1: 'Diabetes'}


# *Step 3: Inspect and Clean Data*

# When starting a new data analysis project it is important to inspect, understand, and clean the data.  When inspecting the first ten rows of data one thing that jumps out right away is that there are measurements of zero for both insulin levels and skin thickness.  

# In[3]:


dataset = read_csv('../input/diabetes.csv')
dataset.head(10)


# It would be a serious medical problem if a patient had an insulin level and skin thickness measurement of zero.  As such, we can conclude that this dataset uses the number zero to represent missing or null data.  Here we can see that as many as half of the rows contain columns with missing data.

# In[4]:


def plotHistogram(values,label,feature,title):
    sns.set_style("whitegrid")
    plotOne = sns.FacetGrid(values, hue=label,aspect=2)
    plotOne.map(sns.distplot,feature,kde=False)
    plotOne.set(xlim=(0, values[feature].max()))
    plotOne.add_legend()
    plotOne.set_axis_labels(feature, 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()
plotHistogram(dataset,"Outcome",'Insulin','Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(dataset,"Outcome",'SkinThickness','SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')


# In[5]:


dataset2 = dataset.iloc[:, :-1]
print("# of Rows, # of Columns: ",dataset2.shape)
print("\nColumn Name           # of Null Values\n")
print((dataset2[:] == 0).sum())


# In[6]:


print("# of Rows, # of Columns: ",dataset2.shape)
print("\nColumn Name              % Null Values\n")
print(((dataset2[:] == 0).sum())/768*100)


# Approximately 50% of the patients did not have their insulin levels measured.  This causes me to be concerned that maybe the doctors only measured insulin levels in unhealthy looking patients -- or maybe they only measured insulin levels after having first made a preliminary diagnosis.  If that were true then this would be a form of [data leakage](https://www.kaggle.com/dansbecker/data-leakage), and it would mean that our model would not generalize well to data collected from doctors who measure insulin levels for every patient.  In order to test this hypothesis I will check whether or not the Insulin and SkinThickness features are correlated with a diagnostic outcome (healthy/diabetic).  What we find is that these features are not highly correlated with any given outcome -- and as such we can rule out our concern of data leakage.
# 

# In[7]:


g = sns.heatmap(dataset.corr(),cmap="BrBG",annot=False)


# In[8]:


dataset.corr()


# The Insulin and SkinThickness measurements are not highly correlated with any given outcome -- and as such we can rule out our concern of data leakage.  The zero values in these categories are still erroneous, however, and therefore should not be included in our model.  It is best to replace these values with some distribution of values near to the median measurement.  Also note that it is best to impute these values *after* the [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function in order to prevent yet another form of data leakage (i.e. the testing data should not be used when calculating the median value to use during imputation).  The following histogram illustrates that the null values have indeed been replaced with median values.

# In[9]:


data = read_csv('../input/diabetes.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
imputer = Imputer(missing_values=0,strategy='median')
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)
X_train3 = pd.DataFrame(X_train2)
plotHistogram(X_train3,None,4,'Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(X_train3,None,3,'SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')


# In[10]:


labels = {0:'Pregnancies',1:'Glucose',2:'BloodPressure',3:'SkinThickness',4:'Insulin',5:'BMI',6:'DiabetesPedigreeFunction',7:'Age'}
print(labels)
print("\nColumn #, # of Zero Values\n")
print((X_train3[:] == 0).sum())
# data[:] = data[:].replace(0, np.NaN)
# print("\nColumn #, # of Null Values\n")
# print(np.isnan(X_train3).sum())


# *Step 4: Evaluate Classification Models*

# Because we have replaced all of the erroneous, missing, and null values with median values we are now ready to train and evaluate our models for predicting diabetes.

# In[11]:


compareABunchOfDifferentModelsAccuracy(X_train2, y_train, X_test2, y_test)
defineModels()
# iterate over classifiers; adapted from https://www.kaggle.com/hugues/basic-ml-best-of-10-classifiers
results = {}
for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X_train2, y_train, cv=5)
    results[name] = scores
for name, scores in results.items():
    print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100*scores.mean(), 100*scores.std() * 2))


# *Step 5: Examine Decision Tree Model in More Detail*

# Next let's explore the decision tree model in more detail.

# In[12]:


def runDecisionTree(a, b, c, d):
    model = DecisionTreeClassifier()
    accuracy_scorer = make_scorer(accuracy_score)
    model.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(model, a, b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    prediction = model.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plot_learning_curve(model, 'Learning Curve For DecisionTreeClassifier', a, b, (0.60,1.1), 10)
    #learning_curve(model, 'Learning Curve For DecisionTreeClassifier', a, b, (0.60,1.1), 10)
    plt.show()
    plot_confusion_matrix(cnf_matrix, classes=dict_characters,title='Confusion matrix')
    plt.show()
    print('DecisionTreeClassifier - Training set accuracy: %s (%s)' % (mean, stdev))
    return
runDecisionTree(X_train2, y_train, X_test2, y_test)
feature_names1 = X.columns.values
def plot_decision_tree1(a,b):
    dot_data = tree.export_graphviz(a, out_file=None, 
                             feature_names=b,  
                             class_names=['Healthy','Diabetes'],  
                             filled=False, rounded=True,  
                             special_characters=False)  
    graph = graphviz.Source(dot_data)  
    return graph 
clf1 = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=12)
clf1.fit(X_train2, y_train)
plot_decision_tree1(clf1,feature_names1)


# *Step 6: Evaluate Feature Importances*

# Many [kernel authors](https://www.kaggle.com/uciml/pima-indians-diabetes-database/kernels) have neglected to deal with the null values and missing data discussed in this notebook.  This mistake did not actually have much of an impact on the performance of most of their models, however, because, as it happens, the Insulin and SkinThickness measurements are actually very poor predictors and are assigned low feature importances as compared to features such as blood glucose levels and body mass index.

# In[13]:


feature_names = X.columns.values
clf1 = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=12)
clf1.fit(X_train2, y_train)
print('Accuracy of DecisionTreeClassifier: {:.2f}'.format(clf1.score(X_test2, y_test)))
columns = X.columns
coefficients = clf1.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('DecisionTreeClassifier - Feature Importance:')
print('\n',fullList,'\n')

feature_names = X.columns.values
clf2 = RandomForestClassifier(max_depth=3,min_samples_leaf=12)
clf2.fit(X_train2, y_train)
print('Accuracy of RandomForestClassifier: {:.2f}'.format(clf2.score(X_test2, y_test)))
columns = X.columns
coefficients = clf2.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('RandomForestClassifier - Feature Importance:')
print('\n',fullList,'\n')

clf3 = XGBClassifier()
clf3.fit(X_train2, y_train)
print('Accuracy of XGBClassifier: {:.2f}'.format(clf3.score(X_test2, y_test)))
columns = X.columns
coefficients = clf3.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('XGBClassifier - Feature Importance:')
print('\n',fullList,'\n')


# In the end we were able to predict diabetes from medical records with an accuracy of approximately 82%.  This was done by using tree-based classifiers that focus on important features such as blood glucose levels and body mass index.  In fact, we only lose 5% accuracy by dropping all of the data except for blood glucose levels and body mass index (see below).

# In[14]:


data = read_csv('../input/diabetes.csv')
data2 = data.drop(['Pregnancies','BloodPressure','DiabetesPedigreeFunction', 'Age','SkinThickness','Insulin'], axis=1)
X2 = data2.iloc[:, :-1]
y2 = data2.iloc[:, -1]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X2, y2, test_size=0.2, random_state=1)
imputer = Imputer(missing_values=0,strategy='median')
X_train3 = imputer.fit_transform(X_train3)
X_test3 = imputer.transform(X_test3)
clf4 = XGBClassifier()
clf4.fit(X_train3, y_train3)
print('Accuracy of XGBClassifier in Reduced Feature Space: {:.2f}'.format(clf4.score(X_test3, y_test3)))
columns = X2.columns
coefficients = clf4.feature_importances_.reshape(X2.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('\nXGBClassifier - Feature Importance:')
print('\n',fullList,'\n')

clf3 = XGBClassifier()
clf3.fit(X_train2, y_train)
print('\n\nAccuracy of XGBClassifier in Full Feature Space: {:.2f}'.format(clf3.score(X_test2, y_test)))
columns = X.columns
coefficients = clf3.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('XGBClassifier - Feature Importance:')
print('\n',fullList,'\n')


# *Step 7: Summarize Results*

# In this notebook we predicted diabetes from medical records with an accuracy of approximately 82%  -- and we also discussed topics such as missing data, data imputation, and feature importances.
