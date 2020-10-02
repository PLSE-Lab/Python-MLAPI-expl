#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook walks through the whole process of machine learning with its different steps and provide web services and mobile applications to predict the player positions. The main goal is to apply these differenet steps to the [Fifa 19 data set][1].
# 
# ![Fifa19 DataSet](https://storage.googleapis.com/kaggle-datasets-images/73041/162580/a17bc456289e382192bebd8bb595719d/dataset-cover.jpg?t=2018-11-04-22 9-55)
# 
# **Data Set**
# 
# [Fifa 19][2] is the most famous and played soccer game around the whole world with over 1.2 billion players. The game contains more than 70000 players in their [databases][3]. 
# 
# The data set presented in this challenge contains more than 18000 players with their different features from physical appearence, clubs, wages and their performance. Our goal is to take those feature and predict the player position correctly.
# 
# **Steps Applied**
# 
# 1. Understand the data using Google Facets
#    
# 2. Prepocess the data
# 
# 3. Apply the [EDA][4](Exploratory Data Analysis) - Visualization
# 
# 4. Divide the data to train and test datasets
# 
# 5. Apply different models and algorithms and tune the hyper parameters
# 
#     5.1. Logistic regression
#     
#     5.2. KNN
#     
#     5.3. Decision Tree
#     
#     5.4. SVM
#     
#     5.5. Neural Network
# 
# 6. Evaluate the models
# 
# 7. Apply AutoML on the data
# 
# 8. Features engineering
# 
# 9. Create Web service to apply the prediction
# 
# [1]: https://www.kaggle.com/karangadiya/fifa19
# [2]: https://www.ea.com/games/fifa/fifa-19
# [3]: https://www.fifaindex.com/
# [4]: https://en.wikipedia.org/wiki/Exploratory_data_analysis

# # 1.Understand the data using Google Facets
# 
# 
# For this part, to understand a little bit about the data, such as general information and how many data are missing we used [Google Facets][1] a very powerful tool to Understed the data which is critical to build a powerful machine learning system.
# 
# 
# **Numerical Features**
# 
# ![Numerical Feature 1](https://i.ibb.co/N6DB4Tq/01-Numerical-Feature.png)
# 
# ![Numerical Feature 2](https://i.ibb.co/fG6wnsZ/02-Numerical-Feature.png)
# 
# ![Numerical Feature 3](https://i.ibb.co/RzhY43t/03-Numerical-Feature.png)
# 
# We Observe that there is more than 18.2K players in the dataset. Some of the attributes are missing for some players such as the weak foot and the skills ratings. 
# 
# **Categorical Features**
# 
# ![Categorical Feature 1](https://i.ibb.co/MMPyzdT/04-Categoratical-Feature.png)
# 
# ![Categorical Feature 2](https://i.ibb.co/1MK9cxj/05-Categoratical-Feature.png)
# 
# We Observe that less than 1% of the data doesn't have position and some others features. Position is the Y feature therefore  we could use it as test dataset or deleting it. 
# 
# **Grouping plotting**
# 
# In This section we tried to plot the data according to different features. The legend represents the actual players positions.
# 
# * Finishing per Age per Position
# 
# ![Finishing per Age per Position](https://i.ibb.co/vvZt49L/06-Analysis.png)
# 
# We Observe that most of the high finishing skills are the strikers, which totally make sense.
# 
# * Short Pass per Age per Position
# 
# ![Short Pass per Age per Position](https://i.ibb.co/THc1Yhf/07-Analysis.png)
# 
# In the same way as the finishing, the high skilled players in the field of passing are the midlifers. 
# 
# 
# 
# [1]: https://pair-code.github.io/facets/

# # 2.Preprocess the data
# 
# **Load Libraries**
# 
# Now, we get to the coding part when we explore that data using python and anaconda libraries. First of all we load the libraires.

# In[ ]:


# Define the libraries and imports
# Panda
import pandas as pd
#mat plot
import matplotlib.pyplot as plt
#Sea born
import seaborn as sns
#Num py
import numpy as np
#Sk learn imports
from sklearn import tree,preprocessing
#ensembles
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
import sklearn.metrics as metrics
#scores
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score,auc  
#models
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_val_score,learning_curve,GridSearchCV,validation_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
#export the model
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


# We define a set of functions that would help us loading the data and preprocess some of the features that we are interested into.
# 
# Note that we are only interested in this project in 3 players positions 'Strikers', 'Midfielders' and 'Defenders'. For more deep project we could have more classes such as specific positions saying 'Left back' or 'center forward'.

# In[ ]:


# Load data from the path to the dataSet
def load_dataset(dataSet_path):
    data = pd.read_csv(dataSet_path)
    return data

#Imputation
def impute_data(df):
    df.dropna(inplace=True)

# Coversion weight to int
def weight_to_int(df):
    df['Weight'] = df['Weight'].str[:-3]
    df['Weight'] = df['Weight'].apply(lambda x: int(x))
    return df

# Coversion height to int
def height_convert(df_height):
        try:
            feet = int(df_height[0])
            dlm = df_height[-2]
            if dlm == "'":
                height = round((feet * 12 + int(df_height[-1])) * 2.54, 0)
            elif dlm != "'":
                height = round((feet * 12 + int(df_height[-2:])) * 2.54, 0)
        except ValueError:
            height = 0
        return height

def height_to_int(df):
    df['Height'] = df['Height'].apply(height_convert)
    
#One Hot Encoding of a feature
def one_hot_encoding(df,column):
    encoder = preprocessing.LabelEncoder()
    df[column] = encoder.fit_transform(df[column].values)
        

#Drop columns that we are not interested in
def drop_columns(df):
    df.drop(df.loc[:, 'Unnamed: 0':'Name' ],axis=1, inplace = True)
    df.drop(df.loc[:, 'Photo':'Special'],axis=1, inplace = True)
    df.drop(df.loc[:, 'International Reputation':'Real Face' ],axis=1, inplace = True)
    df.drop(df.loc[:, 'Jersey Number':'Contract Valid Until' ],axis=1, inplace = True)
    df.drop(df.loc[:, 'LS':'RB'],axis=1, inplace = True)
    df.drop(df.loc[:, 'GKDiving':'Release Clause'],axis=1, inplace = True)

#Transform positions to 3 categories 'Striker', 'Midfielder', 'Defender'    
def transform_positions(df):
    for i in ['ST', 'CF', 'LF', 'LS', 'LW', 'RF', 'RS', 'RW']:
      df.loc[df.Position == i , 'Position'] = 'Striker' 
    
    for i in ['CAM', 'CDM', 'LCM', 'CM', 'LAM', 'LDM', 'LM', 'RAM', 'RCM', 'RDM', 'RM']:
      df.loc[df.Position == i , 'Position'] = 'Midfielder' 
    
    for i in ['CB', 'LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB','GK']:
      df.loc[df.Position == i , 'Position'] = 'Defender' 


# Let's apply these functions to our model.

# In[ ]:


# Load dataset
df= load_dataset("../input/data.csv")
# Drop columns that we are not interested in
drop_columns(df)
# Impute the data that is null
impute_data(df)
# transform weight and height to integer values
weight_to_int(df)
height_to_int(df)
# apply the one hot encoding to the Preferred foot (L,R) => (0,1)
one_hot_encoding(df,'Preferred Foot')
# transform position to striker, midfielder, defender
transform_positions(df)
# show the 10 first rows
df.head(10)


# Let's see the features we will be working on.

# In[ ]:


df.info()


# # 3.Visualization
# 
# In this section, we will apply the EDA to visualize the data with several plottings.
# 
# **Players counts by position**

# In[ ]:


# Count number of players in each position using countplot
plt.figure(figsize=(12, 8))
plt.title("Number of Players by position")
fig = sns.countplot(x = 'Position', data =df)


# We notice that most of the players are defenders and midfielders which makes sense, since that in every team we need less strikers.
# 
# Now, we want to explore the players skills features to plot them as catagories, but these are numerical. Therefor, we will first transform them.

# In[ ]:


# Define categorical skills base on the rating
def categorize_skill(df,column):
    bins = (10,30,50,70,100)
    group_names = ['Low','Moderate','High','VeryHigh']
    categories = pd.cut(df[column],bins,labels=group_names)
    new_column = column+'_cat'
    df[new_column]=categories
categorize_skill(df,"Finishing")
categorize_skill(df,"Strength")
categorize_skill(df,"FKAccuracy")


# Let's make some plots.
# 
# **Short Passing by finishing according to the positions**
# 

# In[ ]:


# Crate Category plot from seaborn on Finishing & ShortPassing By position
sns.catplot(x="Finishing_cat", y="ShortPassing", hue="Position",
            markers=["^", "o","x"], linestyles=["-", "--","-"],
            kind="point", data=df);


# **Interceptions by Strength according to the positions**

# In[ ]:


# Crate Category plot from seaborn on  Strength & Interception By position
sns.catplot(x="Strength_cat", y="Interceptions", hue="Position",
            markers=["^", "o","x"], linestyles=["-", "--","-"],
            kind="point", data=df);


# **Interceptions by Strength according to the positions**

# In[ ]:


# Crate Category plot from seaborn on FKAccuracy & Penalties By position
sns.catplot(x="FKAccuracy_cat", y="Penalties", hue="Position",
            markers=["^", "o","x"], linestyles=["-", "--","-"],
            kind="point", data=df);


# **Physical appearences by position**
# 
# Let's plot some physical appearences such as age, height and weight and see how it will affect the players positions.

# In[ ]:


# Box plot skills by position
f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=False)
sns.despine(left=True)
sns.boxplot('Position', 'Jumping', data = df, ax=axes[0, 0])
sns.boxplot('Position', 'Age', data = df, ax=axes[0, 1])
sns.boxplot('Position', 'Height', data = df, ax=axes[1, 0])
sns.boxplot('Position', 'Weight', data = df, ax=axes[1, 1])


# **Reaction by age**

# In[ ]:


# Bar plot Reaction by Age
mean_value_per_age = df.groupby('Age')['Reactions'].mean()
p = sns.barplot(x = mean_value_per_age.index, y = mean_value_per_age.values)
p = plt.xticks(rotation=90)


# **Scatter plot skills and positions**

# In[ ]:


#Scatter plot Finishing by shortPassing classified by position
ax = sns.scatterplot(x="ShortPassing", y="Finishing", hue="Position",data=df)


# **Features correlation**
# 
# Let's correlate our features, but before that let's drop the categorical columns that we have created for plotting.

# In[ ]:


# Drop some of unuseful coloumns
drop_elements = ['Position', 'Finishing_cat', 'Strength_cat', 'FKAccuracy_cat']
train=df.drop(drop_elements, axis = 1)


# Now, let's plot all the numerical features for correlation.

# In[ ]:


# Create the heat map of features correlation
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=False)


# We see that some of the features are very correlated, this could be very helpful when we will be doing the features engineering.

# # 4.Divide the data to train and test datasets
# 
# Our test size would be 20% from the data, the most recommended into the field. We will have then 80% as training data and 20% for testing.

# In[ ]:


# Divide the data to train and test

# Drop the elements that has been created for 
drop_elements = ['Finishing_cat', 'Strength_cat', 'FKAccuracy_cat']
df=df.drop(drop_elements, axis = 1)

# Create the unique values for the positions encoded as Defender:0, Midfielder:1, Striker:2
positions = df["Position"].unique()
encoder = preprocessing.LabelEncoder()
df['Position'] = encoder.fit_transform(df['Position'])

#The Y feature is the position
y = df["Position"]

#The other features are all but the position
df.drop(columns=["Position"],inplace=True)

#Split the data
X_train_dev, X_test, y_train_dev, y_test = train_test_split(df, y, 
                                                    test_size=0.20, 
                                                    random_state=42 )


# # 5.Apply different models and algorithms and tune the hyper parameters
# 
# In this section we will train different models, find their score performance, tune the hyper parameters.
# 
# First, we will define some functions that will help us train the models, evaluate and plot the curves.
# 
# **Plot Confusion Matrix**
# 
# 

# In[ ]:


# Plot the confusion matrix
def plot_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    sns.set(font_scale=1.4)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16})
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


# **Plot Learning Curve**
# 

# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
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


# **Plot validation curve**

# In[ ]:


def plot_curve(ticks, train_scores, test_scores):
    train_scores_mean = -1 * np.mean(train_scores, axis=1)
    train_scores_std = -1 * np.std(train_scores, axis=1)
    test_scores_mean = -1 * np.mean(test_scores, axis=1)
    test_scores_std = -1 * np.std(test_scores, axis=1)

    plt.figure()
    plt.fill_between(ticks, 
                     train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, alpha=0.1, color="b")
    plt.fill_between(ticks, 
                     test_scores_mean - test_scores_std, 
                     test_scores_mean + test_scores_std, alpha=0.1, color="r")
    plt.plot(ticks, train_scores_mean, 'b-', label='Training Error')
    plt.plot(ticks, test_scores_mean, 'r-', label='Validation Error')
    plt.legend(fancybox=True, facecolor='w')

    return plt.gca()

def plot_validation_curve(clf, X, y, param_name, param_range, scoring='accuracy'):
    plt.xkcd()
    ax = plot_curve(param_range, *validation_curve(clf, X, y, cv=4, 
                                                   scoring=scoring, 
                                                   param_name=param_name, 
                                                   param_range=param_range, n_jobs=4))
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(2,12)
    ax.set_ylim(-0.97, -0.83)
    ax.set_ylabel('Error')
    ax.set_xlabel('Model complexity')
    ax.text(9, -0.94, 'Overfitting', fontsize=14)
    ax.text(3, -0.94, 'Underfitting', fontsize=14)
    ax.axvline(7, ls='--')
    plt.tight_layout()


# **Train the model and score**
# 
# This function is responisble for traing the model and scoring it. We print at the end the Accuracy and F1 Score metrics.

# In[ ]:


def train_and_score(clf,X_train,y_train,X_test,y_test):
    clf = clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    cf = confusion_matrix(y_test,preds)

    print(plot_confusion_matrix(cf, class_names=positions))

    print(" Accuracy: ",accuracy_score(y_test, preds))
    print(" F1 score: ",metrics.f1_score(y_test, preds,average='weighted'))


# # 5.1. Logistic regression
# 
# We apply our first model Logistic regression with a cross validation of 5 folds.

# In[ ]:


LR = LogisticRegressionCV(cv=5,random_state=20, solver='lbfgs',
                             multi_class='multinomial')
train_and_score(LR,X_train_dev,y_train_dev,X_test,y_test)


# In[ ]:


plot_learning_curve(LR, "Logistic Regression Curve", X_train_dev, y_train_dev)


# # 5.2. K-nearest Neighbours
# 
# Then, we apply the KNN model with gread serach that apply to the K (number of neighbors) in a range from 1 to 25 with a cross validation of 5.

# In[ ]:


#create new a knn model
knn_model = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
KNN = GridSearchCV(knn_model, param_grid, cv=5)

train_and_score(KNN,X_train_dev,y_train_dev,X_test,y_test)


# In[ ]:


plot_learning_curve(KNN, "KNN Regression Curve", X_train_dev, y_train_dev)


# We want to draw the model complexity curve and see how far the model is learning based on the n_neighbours range.

# In[ ]:


plot_validation_curve(KNeighborsClassifier(), X_train_dev, y_train_dev, param_name='n_neighbors', param_range=range(2,25))


# # 5.3. Decision Tree
# 
# First all, let's define a function that calculate the best minimum impurity a parameter to pass to the decision tree.

# In[ ]:


def min_impurity(X,y):
    tr_acc = []
    mln_set = range(75,90)                                 

    for minImp in mln_set:
        clf = tree.DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=minImp/100000)
        scores = cross_val_score(clf, X, y, cv=10)
        tr_acc.append(scores.mean())

    best_mln = mln_set[np.argmax(tr_acc)]
    return best_mln

best_min= min_impurity(X_train_dev,y_train_dev)


# Now, let's apply the Decision Tree model with the best minimum impurity.

# In[ ]:


DT = tree.DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=best_min/100000)
train_and_score(DT,X_train_dev,y_train_dev,X_test,y_test)


# In[ ]:


plot_learning_curve(DT, "Decision Tree Learning Curve", X_train_dev, y_train_dev)


# # 5.3.1. Bagging
# 
# Now, let's work with DT but using the bagging model.

# In[ ]:


DTBG = BaggingClassifier(tree.DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=best_min/100000))#
train_and_score(DTBG,X_train_dev,y_train_dev,X_test,y_test)


# In[ ]:


plot_learning_curve(DTBG, "Bagging Decision Tree Learning Curve", X_train_dev, y_train_dev)


# # 5.3.2. Boosting
# 
# Now, let's try another ensembling technique the boosting.

# In[ ]:


dtrain = xgb.DMatrix(X_train_dev, label=y_train_dev)

dtest = xgb.DMatrix(X_test,label=y_test)

param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 50  # the number of training iterations
DTBST = xgb.train(param, dtrain, num_round)
DTBST.dump_model('dump.raw.txt')
preds = DTBST.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])
cf = confusion_matrix(y_test, best_preds)

print(plot_confusion_matrix(cf, class_names=positions))
print(" Accuracy: ",accuracy_score(y_test, best_preds))
print(" F1 score: ",metrics.f1_score(y_test, best_preds,average='weighted'))


# # 5.3.3. Random Forest
# 
# Now, let's try another ensembling technique the random forest that would search to tune many parameters.

# In[ ]:


gridsearch_forest = RandomForestClassifier()

params = {
    "n_estimators": [1, 10, 100],
    "max_depth": [5,8,15], #2,3,5 85 #5,8,10 88 #5 8 15 89
    "min_samples_leaf" : [1, 2, 4]
}

RF = GridSearchCV(gridsearch_forest, param_grid=params, cv=5 )
train_and_score(RF,X_train_dev,y_train_dev,X_test,y_test)


# In[ ]:


plot_learning_curve(RF, "Random Forest Learning Curve", X_train_dev, y_train_dev)


# # 5.5. SVM
# 
# In this section we will be using the neural network with different parameters of the number of layers and the size. After tuning the parameters we find that the best option is 50-20 NN.

# In[ ]:


SVM = SVC(kernel='linear', C=1)
train_and_score(SVM,X_train_dev,y_train_dev,X_test,y_test)


# In[ ]:


plot_learning_curve(SVM, "SVM Curve", X_train_dev, y_train_dev)


# # 5.5. Neural Network
# 
# In this section we will be using the neural network with different parameters of the number of layers and the size. After tuning the parameters we find that the best option is 50-20 NN.

# In[ ]:


MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
train_and_score(MLP,X_train_dev,y_train_dev,X_test,y_test)


# In[ ]:


MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 5), random_state=1)
train_and_score(MLP,X_train_dev,y_train_dev,X_test,y_test)


# In[ ]:


MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(20, 15), random_state=1)
train_and_score(MLP,X_train_dev,y_train_dev,X_test,y_test)


# In[ ]:


MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(50, 20), random_state=1)
train_and_score(MLP,X_train_dev,y_train_dev,X_test,y_test)


# In[ ]:


plot_learning_curve(MLP, "Neural Network Curve", X_train_dev, y_train_dev)


# # 6. Evaluate the models
# 
# According to the accuracy we can see that almost all of the models get about the same score. The best model is the Neural Network with 50,20 parameters.
# 
# Now let's draw the ROC-AUC curve and plot the different models.

# In[ ]:


plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()

y1_test=y_test[(y_test ==0) | (y_test ==1)]
x1_test = X_test[X_test.index.isin(y1_test.index)]


y_predict_probabilities = LR.predict_proba(x1_test)[:,1]
fpr, tpr, _ = roc_curve(y1_test, y_predict_probabilities)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='LR (area = %0.3f)' % roc_auc)

y_predict_probabilities = KNN.predict_proba(x1_test)[:,1]
fpr, tpr, _ = roc_curve(y1_test, y_predict_probabilities)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue',
         lw=2, label='KNN (area = %0.3f)' % roc_auc)

y_predict_probabilities = RF.predict_proba(x1_test)[:,1]
fpr, tpr, _ = roc_curve(y1_test, y_predict_probabilities)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='green',
         lw=2, label='RF (area = %0.3f)' % roc_auc)

y_predict_probabilities = MLP.predict_proba(x1_test)[:,1]
fpr, tpr, _ = roc_curve(y1_test, y_predict_probabilities)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='red',
         lw=2, label='NN (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# Note that all of them are so close.

# # 7. AutoML
# We applied AutoML from SKLearn on the data trying to find the best estimator. We run AutoML on a dedicated server with some capabilities and here we present the code and the output.
# 
# ![AutoML_Code](https://i.ibb.co/PrWWTrz/AutoML-1.png)
# 
# After one hour of code running, we got finnaly the results.
# 
# ![AutoML_Result](https://i.ibb.co/vPGXLRx/AutoML-2.png)
# 
# We see that the best accuracy is : 0.895 which is so close to the Neural Network of 50,20.
# 
# Finnaly, we looked for the best estimator.
# 
# ![AutoML_Result](https://i.ibb.co/K6t45hf/AutoML-3.png)
# 
# We found out that the best estimator accoring to AutoML is the gradient boosting.

# # 8. Feature engineering
# 
# After comparing the different models we found that the Neural Network is the best model with the highest accuracy. We tried to change our features by deleting correlated features or gathering some features together, but this didn't make the model more accurate.

# # 9. Web Services
# 
# Finnaly, we create a web service that throw a post with the data to predict that could be found on the following link : http://thefourtwofour.com:1992/api/predict
# 
# Here an example of a post request and the results :
# ![Post Predict](https://i.ibb.co/RpkBYQc/Post.png)

# # Refrences
# 
# This notebook has been produced using some external kernles and ressources listed below: 
# 
# [Fifa 19 Kernel][1] 
# 
# [SK Learn][2] 
# 
# 
# [1]: https://www.kaggle.com/brunosette/classification-of-field-position-decision-tree/data?fbclid=IwAR1_gS-pS9qXcyqEtRKZvmPXlwjwbK2Fd88NV_9OsayaE8LI44LTpoKJTVQ
# [2]: https://scikit-learn.org/stable/
