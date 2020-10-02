#!/usr/bin/env python
# coding: utf-8

# ![alt text](https://www.kaggle.com/static/images/site-logo.png "Kaggle logo")
# <div align="center">
#     <a href="https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo">
#         <h1>
#             Ghouls, Goblins, and Ghosts... Boo!
#         </h1>
#     </a>
# </div>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# SETUP
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import warnings
import copy
warnings.filterwarnings("ignore")

import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()


# In[ ]:


# Define some functions
import collections

def flatten(d, parent_key='', sep='_'):
    """ Function that flatten a dict """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def showExecTime(startPoint, initialString = "", verbose=True):
    """
    Compute the execution time from an initial starting point.
    You can also pass me a string to print out at the end of computation.
    
    Parameters
    ----------
    startPoint : float, timestamp of the starting point
    initialString : string to output on the console, before the execution time
    
    Returns
    -------
    endPoint - startPoint, the difference between the two timestamps
    """
    eex = time.time()
    seconds = round(eex - startPoint, 2)
    minutes = (seconds/60)
    hours = int(minutes/60)
    minutes = int(minutes % 60)
    seconds = round(seconds % 60, 2)
    if verbose:
        print("\n- "+initialString+" Execution time: %sh %sm %ss -" % (hours, minutes, seconds))
    return eex - startPoint


# In[ ]:


# Load the test and the train tables
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# ***
# <div>
#     <h1>Data Exploration</h1>
# </div>

# In[ ]:


# Look the train
print("\nTrain structure:\n\n", train.head(5))
# Look the test
print("\nTest structure:\n\n", test.head(5))


# <p>
# Data fields:
# <ul>
#     <li><b>id</b>              - id of the creature</li>
#     <li><b>bone_length</b>     - average length of bone in the creature, normalized between 0 and 1</li>
#     <li><b>rotting_flesh</b>   - percentage of rotting flesh in the creature</li>
#     <li><b>hair_length</b>     - average hair length, normalized between 0 and 1</li>
#     <li><b>has_soul</b>        - percentage of soul in the creature</li>
#     <li><b>color</b>           - dominant color of the creature: 'white', 'black', 'clear', 'blue', 'green', 'blood'</li>
#     <li><b>type</b>            - target variable: 'Ghost', 'Goblin', and 'Ghoul'</li>
# </ul>
# </p>

# In[ ]:


print("\nTrain info:\n")
train.info()


# <p>
# In the end, we have <b>4 continous variables</b> (bone_length, rotting_flesh, hair_length, has_soul) and <b>1 categorical variable</b> (color).<br>
# We are able to see some <b>statistics</b> of the numerical variables.
# </p>

# In[ ]:


print("\nTrain description:\n\n", train.drop('id', axis=1, inplace=False).describe())
print("\nTest description:\n\n", test.drop('id', axis=1, inplace=False).describe())


# <p>
# Now we create a map that translate the type of creature into a color.<br>
# We can use this one:
# <ul>
#     <li><b>Ghost</b>  - <span style="color:#ff4141">#ff4141 (red)</span></li>
#     <li><b>Ghoul</b>  - <span style="color:#995bbe">#995bbe (violet)</span></li>
#     <li><b>Goblin</b> - <span style="color:#16dc88">#16dc88 (green)</span></li>
# </ul>

# In[ ]:


colors = {
    "Ghost" : "#ff4141",
    "Ghoul" : "#995bbe",
    "Goblin": "#16dc88"
}


# In[ ]:


sns.set(style="whitegrid", context="talk")
sns.set_color_codes("pastel")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(7, 4))

x, y = [], []
for key in colors:
    x.append(key)
    y.append(train['type'].value_counts()[key])

# Plot the different type occurrences
sns.barplot(x, y, palette=colors)

# Finalize the plot
for n, (label, _y) in enumerate(zip(x, y)):
    ax.annotate( # Attach the counts
        s='{:.0f}'.format(abs(_y)),
        xy=(n, _y),
        ha='center',va='center',
        xytext=(0,10),
        textcoords='offset points',
        weight='bold'
    )
    ax.annotate( # Attach the type label
        s=label,
        xy=(n, _y),
        ha='center',va='center',
        xytext=(0,-15),
        textcoords='offset points',
        weight='bold'
    )

# Add a legend and informative axis label
ax.set(ylabel="Number of occurrences", xlabel="Creature type")
ax.set_xticks([])
plt.setp(f.axes, yticks=[])
plt.title("Monsters occurrences")
plt.tight_layout(h_pad=3)


# In[ ]:


fig = {
    'data': [{'labels': x,
              'values': y,
              'marker': {'colors': [colors[m] for m in x]},
              'type': 'pie'}],
    'layout': {'title': 'Monsters occurences'}
}


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()

iplot(fig, filename='monsters_in_pie')


# <p>
# <h2>Numerical variables</h2><br>
# We can have a look at the different creature with respect to different features.<br>
# We have 4 different continous variable, then we can produce 6 different plot with distribution from each creature type.
# </p>

# In[ ]:


colormap = {
    "Ghost" : "Reds",
    "Ghoul" : "BuPu",
    "Goblin": "Greens"
}

# Subset the dataset by creature
ghost = train.query("type == 'Ghost'")
ghoul = train.query("type == 'Ghoul'")
goblin = train.query("type == 'Goblin'")

features = list(train.describe().columns[1:])


# In[ ]:


# Set up the matplotlib figure
f, axes = plt.subplots(2, 3, figsize=(7, 7), sharex=False, sharey=False)

ax_ind = 0
for i in range(len(features)-1):
    for j in range(i+1, len(features)):
        # Get the features
        feat1 = features[i]
        feat2 = features[j]
        
        # Set up the figure
        ax = axes.flat[ax_ind]
        ax_ind += 1
        ax.set_aspect("equal")

        # Draw the three density plots
        sns.kdeplot(ghost[feat1], ghost[feat2], ax=ax,
                         cmap=colormap['Ghost'], shade=True, shade_lowest=False, alpha=.6)
        sns.kdeplot(ghoul[feat1], ghoul[feat2], ax=ax,
                         cmap=colormap['Ghoul'], shade=True, shade_lowest=False, alpha=.6)
        sns.kdeplot(goblin[feat1], goblin[feat2], ax=ax,
                         cmap=colormap['Goblin'], shade=True, shade_lowest=False, alpha=.6)

# Conclude
plt.suptitle("Bivariate kernel densities")
f.tight_layout()


# <p>
# Maybe, also a scatterplot matrix could be useful.
# </p>

# In[ ]:


# Set up the matplotlib figure
sns.pairplot(train.drop('id', axis=1, inplace=False),
             palette=colors, hue="type",
             diag_kind="kde", diag_kws=dict(shade=True))
plt.suptitle("Pairwise relationships in the dataset")
plt.show()


# <p>
# We can notice how <i>has_soul</i> and <i>hair_length</i> seems to be the most discrimant features, in the same "direction".<br>
# In the same way, <i>bone_length</i> seems a little less strong.<br>
# Then, we can create some <i>new variables</i> that are the combination of the original ones and analyze them.
# </p>

# In[ ]:


train['new_var'] = train['hair_length'] + train['has_soul'] + train['bone_length'] - train['rotting_flesh']
train['new_var'] = train['new_var'] - min(train['new_var'])
train['new_var'] = train['new_var'] / max(train['new_var'])
train.describe()['new_var']


# In[ ]:


# Insert other features
train['bone_hair'] = train['hair_length'] + train['bone_length']
train['bone_hair'] = train['bone_hair'] - min(train['bone_hair'])
train['bone_hair'] = train['bone_hair'] / max(train['bone_hair'])

train['hair_soul'] = train['hair_length'] + train['has_soul']
train['hair_soul'] = train['hair_soul'] - min(train['hair_soul'])
train['hair_soul'] = train['hair_soul'] / max(train['hair_soul'])

train['bone_soul'] = train['bone_length'] + train['has_soul']
train['bone_soul'] = train['bone_soul'] - min(train['bone_soul'])
train['bone_soul'] = train['bone_soul'] / max(train['bone_soul'])

train['flesh_soul'] = train['rotting_flesh'] + train['has_soul']
train['flesh_soul'] = train['flesh_soul'] - min(train['flesh_soul'])
train['flesh_soul'] = train['flesh_soul'] / max(train['flesh_soul'])


# In[ ]:


# Insert other very strange features
train['new_var_2'] = (train['hair_length'] * train['has_soul'] * train['bone_length'] * (1-train['rotting_flesh'])) ** (1/4)

train['bone_hair_2'] = (train['hair_length'] * train['bone_length']) ** (1/2)

train['flesh_soul_2'] = (train['rotting_flesh'] * train['has_soul']) ** (1/2)

train['hair_soul_2'] = (train['hair_length'] * train['has_soul']) ** (1/2)

train['bone_soul_2'] = (train['bone_length'] * train['has_soul']) ** (1/2)


# In[ ]:


# Group together similar features names
old_numerical = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']
new_ones = ['new_var', 'bone_hair', 'flesh_soul', 'hair_soul', 'bone_soul']
new_ones_2 = [x+'_2' for x in new_ones]
new_ones, new_ones_2


# In[ ]:


# Subset the dataset by creature
ghost = train.query("type == 'Ghost'")
ghoul = train.query("type == 'Ghoul'")
goblin = train.query("type == 'Goblin'")

sns.set(style="darkgrid")

# Draw the three density plots
plt.figure(figsize=(8, 4))
feature_you_want = 'new_var_2' # -> you can modify it, pick what you want!
sns.kdeplot(ghost[feature_you_want], color=colors['Ghost'], shade=True, shade_lowest=False, alpha=.6)
sns.kdeplot(goblin[feature_you_want], color=colors['Goblin'], shade=True, shade_lowest=False, alpha=.6)
sns.kdeplot(ghoul[feature_you_want], color=colors['Ghoul'], shade=True, shade_lowest=False, alpha=.6)
plt.show()


# In[ ]:


# Boxplots for new variables - Grouped Horizontal Box Plot
data = [
    # One dictionary for each monster type
    {
        # Ghost
        'name': 'Ghost',
        'x': sum([list(ghost[var]) for var in new_ones + new_ones_2], []),
        'y': sum([[var]*len(ghost) for var in new_ones + new_ones_2], []),
        'marker': {'color': colors['Ghost']},
        'boxmean': False,
        'orientation': 'h',
        'type': 'box',
    },
    {
        # Goblin
        'name': 'Goblin',
        'x': sum([list(goblin[var]) for var in new_ones + new_ones_2], []),
        'y': sum([[var]*len(goblin) for var in new_ones + new_ones_2], []),
        'marker': {'color': colors['Goblin']},
        'boxmean': False,
        'orientation': 'h',
        'type': 'box',
    },
    {
        # Ghoul
        'name': 'Ghoul',
        'x': sum([list(ghoul[var]) for var in new_ones + new_ones_2], []),
        'y': sum([[var]*len(ghoul) for var in new_ones + new_ones_2], []),
        'marker': {'color': colors['Ghoul']},
        'boxmean': False,
        'orientation': 'h',
        'type': 'box',
    }
]

layout = {
    'title': 'Analysis on New Features Distributions',
    'xaxis': {
        'title': 'normalized moisture',
        'zeroline': True,
    },
    'boxmode': 'group',
    'height': 1600,
}

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig)


# <p>
# We can notice the relevant new features and the not-so-meaningful ones.<br>
# It seems that <i>flesh_soul</i> and <i>flesh_soul_2</i> are a bit confusing, while the remaining ones are more or less useful in the same way.
# </p>

# <p>
# <h2>Categorical variables</h2><br>
# 
# We should also explore the categorical variable referring to the color.
# 
# </p>

# In[ ]:


plt.figure(figsize=(9,4))
sns.countplot(x='color', hue='type', palette=colors, data=train)
plt.suptitle("Distribution of the 'color' class")
plt.show()


# <p>
# All alone, the <i>green</i>, the <i>black</i> and the <i>blue</i> monsters are not so distinguishable.
# </p>

# In[ ]:


all_colors = list(set(train['color'].values))
useful_colors = ['clear', 'white', 'blood']
color =  pd.get_dummies(train['color'])
train_data = pd.concat([train, color[useful_colors]], axis = 1)
train_data.describe()


# In[ ]:


sns.pairplot(train_data[useful_colors + ['type', 'new_var', 'hair_soul_2']], hue="type",
             diag_kind="kde", diag_kws=dict(shade=True))
plt.suptitle("Pairwise relationships in the dataset")
plt.show()


# <p>
# We can see how the <i>new variable</i> together with the <i>useful colors</i> could provide nice insights.<br>
# We should exploit this fact in order to reach better results in the classification.
# </p>

# ***
# <p>
# <h1>Data Modeling</h1><br>
# In this part, we try to <b>create some models</b> from our data, in order to characterize our creatures.<br>
# We fit the data in <u>different models</u>, feeding them with <u>different attributes</u>, with the objective to <b>compare them</b> and, in the end, pick the best one.
# </p>

# In[ ]:


from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree
from sklearn import dummy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import model_selection
from sklearn import linear_model
import time

try:
    train.drop(['flesh_soul', 'flesh_soul_2'], axis=1, inplace=True)
    new_ones.remove('flesh_soul')
    new_ones_2.remove('flesh_soul_2')
except:
    # already dropped
    pass

# Insert other features in the test set
test['new_var'] = test['hair_length'] + test['has_soul'] + test['bone_length'] - test['rotting_flesh']
test['new_var'] = test['new_var'] - min(test['new_var'])
test['new_var'] = test['new_var'] / max(test['new_var'])

test['bone_hair'] = test['hair_length'] + test['bone_length']
test['bone_hair'] = test['bone_hair'] - min(test['bone_hair'])
test['bone_hair'] = test['bone_hair'] / max(test['bone_hair'])

test['hair_soul'] = test['hair_length'] + test['has_soul']
test['hair_soul'] = test['hair_soul'] - min(test['hair_soul'])
test['hair_soul'] = test['hair_soul'] / max(test['hair_soul'])

test['bone_soul'] = test['bone_length'] + test['has_soul']
test['bone_soul'] = test['bone_soul'] - min(test['bone_soul'])
test['bone_soul'] = test['bone_soul'] / max(test['bone_soul'])

# Insert vother ery strange features
test['new_var_2'] = (test['hair_length'] * test['has_soul'] * test['bone_length'] * (1-test['rotting_flesh'])) ** (1/4)
test['bone_hair_2'] = (test['hair_length'] * test['bone_length']) ** (1/2)
test['hair_soul_2'] = (test['hair_length'] * test['has_soul']) ** (1/2)
test['bone_soul_2'] = (test['bone_length'] * test['has_soul']) ** (1/2)

color =  pd.get_dummies(test['color'])
test_data = pd.concat([test, color[all_colors]], axis = 1)

color =  pd.get_dummies(train['color'])
train_data = pd.concat([train, color[all_colors]], axis = 1)

train_data = train_data.drop('id', axis=1, inplace=False)
test_data = test_data.drop('id', axis=1, inplace=False)

# Asert that test set has only the 'type' column left
train_data.columns - test_data.columns == ['type']


# In[ ]:


# MODELS

# Naive Bayes
nb = {'name': 'Bernoulli NaiveBayes'}
nb['model'] = naive_bayes.BernoulliNB()

# Logistic Regression
lr = {'name': 'Logistic Regression'}
lr['model'] = linear_model.LogisticRegression(solver='lbfgs')

# Logistic Regression with CV
lrcv = {'name': 'Cross-Validated Logistic Regression'}
lrcv['model'] = linear_model.LogisticRegressionCV(Cs=100, solver='lbfgs', n_jobs=-1)

# SVC
svc = {'name': 'Support Vector Machine'}
svc['model'] = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True))

# Decision Tree
dtree = {'name': 'Decision Tree'}
dtree['model'] = tree.DecisionTreeClassifier(max_depth=20, min_samples_leaf=3)

# Random Forest
rf = {'name': 'Random Forest'}
rf['model'] = ensemble.RandomForestClassifier(n_estimators=1000, max_depth=20, min_samples_leaf=3, n_jobs=-2)

# K-Nearest Neighbors
k, w = 7, ['uniform', 'distance'][1]
knn = {'name': str(k)+'-Nearest Neighbors '+w}
knn['model'] = KNeighborsClassifier(n_neighbors=k, weights=w, algorithm='auto', n_jobs=-2)

# Dummy most_frequent - baseline
dummy_uni = {'name': 'Dummy MostFrequent'}
dummy_uni['model'] = dummy.DummyClassifier(strategy='most_frequent')

# Dummy Stratified - baseline
dummy_str = {'name': 'Dummy Stratified'}
dummy_str['model'] = dummy.DummyClassifier(strategy='stratified')


# ***
# <p>
# <h1>Model Validation</h1><br>
# In this part, we validate our models.
# </p>

# In[ ]:


# VALIDATION
# Keep all the models together
models = [nb, lr, lrcv, svc, dtree, rf, knn, dummy_uni, dummy_str]


# In[ ]:


# Evaluate each model
def evaluate_models(models, features, verbose=True, n_splits=10):
    for model in models:
        clf = model['model']
        if verbose: print("\n"+model['name'])
        
        # Fitting
        begin = time.time()
        clf.fit(train_data[features], train_data['type'])
        model['fit_time'] = showExecTime(begin, model['name']+" fitted.", verbose)

        # Prediction on the entire train set, where they are fitted
        begin = time.time()
        predicted_train = np.array(clf.predict_proba(train_data[features]))
        model['prediction_time'] = showExecTime(begin, model['name']+" prediction complete.", verbose)
        logloss = log_loss(train_data['type'].values, predicted_train)
        model['train_log_loss'] = logloss    
        if verbose: print ("\n\tLog_loss on the train:", logloss)

        # Accuracy performance
        predicted_train = np.array(clf.predict(train_data[features]))
        accuracy_score(train_data['type'], predicted_train)
        accuracy = accuracy_score(train_data['type'], predicted_train)
        model['train_accuracy'] = accuracy
        if verbose: print("\tTrain accuracy:", accuracy)    

        # Strified K-Fold
        skf = model_selection.StratifiedKFold(n_splits=n_splits)
        logloss_train, logloss_test = [], []
        accuracies = []
        if verbose: print("\n\tStratified K-Fold")
        for train_index, test_index in skf.split(train_data[features], train_data['type']):
            #print("\t\tTEST FOLD: [%d: %d]" % (test_index[0], test_index[-1]))
            x_train, x_test = train_data[features].iloc[train_index], train_data[features].iloc[test_index]
            y_train, y_test = train_data['type'][train_index], train_data['type'][test_index]

            # Fit the model with the X at each iteration
            clf.fit(x_train, y_train)

            # Probabilities estimation
            predict_proba_train = np.array(clf.predict_proba(x_train))
            predict_proba_test = np.array(clf.predict_proba(x_test))
            # Log loss
            logloss_train.append(log_loss(y_train, predict_proba_train))
            logloss_test.append(log_loss(y_test, predict_proba_test))

            # Classification
            predict_test = np.array(clf.predict(x_test))
            # Accuracy
            accuracy = accuracy_score(y_test, predict_test)
            accuracies.append(accuracy)

        model['skf_validation'] = {
            'logloss_train': {
                'mean': np.mean(logloss_train),
                'std': np.std(logloss_train)
            },
            'logloss_test': {
                'mean': np.mean(logloss_test),
                'std': np.std(logloss_test)
            },
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
            }
        }
        if verbose:
            print ("\tLog_loss on the train: %0.2f (+/- %0.2f)" % (np.mean(logloss_train), np.std(logloss_train) * 2))
            print ("\tLog_loss on the test: %0.2f (+/- %0.2f)" % (np.mean(logloss_test), np.std(logloss_test) * 2))
            print ("\tClassification accuracy: %0.2f (+/- %0.2f)" % (np.mean(accuracies), np.std(accuracies) * 2))
    return models


# In[ ]:


# Select the features
all_features = all_colors + new_ones + new_ones_2 + old_numerical
print(all_features)


# In[ ]:


print("Validating group 1")
models1 = {'data': evaluate_models(copy.deepcopy(models), all_colors, False),
           'attr': 'only colors'}
print("Validating group 2")
models2 = {'data': evaluate_models(copy.deepcopy(models), old_numerical, False),
           'attr': 'old numerical'}
print("Validating group 3")
models3 = {'data': evaluate_models(copy.deepcopy(models), old_numerical + new_ones + new_ones_2, False),
           'attr': 'all numerical'}
print("Validating group 4")
models4 = {'data': evaluate_models(copy.deepcopy(models), all_features, False),
           'attr': 'all features'}
print("Validating group 5")
models5 = {'data': evaluate_models(copy.deepcopy(models), new_ones, False),
           'attr': 'new ones'}
print("Validating group 6")
models6 = {'data': evaluate_models(copy.deepcopy(models), new_ones_2, False),
           'attr': 'new ones squared'}
print("Validating group 7")
models7 = {'data': evaluate_models(copy.deepcopy(models), new_ones + new_ones_2, False),
           'attr': 'all new ones'}


# In[ ]:


def bar_and_scatter(models_dict):
    df = pd.DataFrame([flatten(d) for d in models_dict['data']]).drop(['model', 'fit_time'], axis=1, inplace=False)
    names = df.name
    skf_accuracy_mean, skf_accuracy_std = df.skf_validation_accuracy_mean, df.skf_validation_accuracy_std
    skf_logloss_mean1, skf_logloss_std1 = df.skf_validation_logloss_test_mean, df.skf_validation_logloss_test_std
    bar = go.Bar(
        x=names,
        y=skf_accuracy_mean,
        name=models_dict['attr'],
        error_y=dict(
            type='data',
            array=skf_accuracy_std
        ),
        opacity=0.7
    )
    scatter = go.Scatter(
        x=names,
        y=skf_logloss_mean1,
        name=models_dict['attr'],
        error_y=dict(
            type='data',
            array=skf_logloss_std1
        ),
        opacity=0.7
    )
    return (bar, scatter)

# Get all the traces
(bar1, scatter1) = bar_and_scatter(models1)
(bar2, scatter2) = bar_and_scatter(models2)
(bar3, scatter3) = bar_and_scatter(models3)
(bar4, scatter4) = bar_and_scatter(models4)
(bar5, scatter5) = bar_and_scatter(models5)
(bar6, scatter6) = bar_and_scatter(models6)
(bar7, scatter7) = bar_and_scatter(models7)


# In[ ]:


# Bar chart for the accuracy
layout = go.Layout(
    title='Accuracy'
)
data = [bar1, bar2, bar3, bar4, bar5, bar6, bar7]
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='error-bar-bar')


# In[ ]:


layout = go.Layout(
    title='Log loss'
)
data = [scatter1, scatter2, scatter3, scatter4, scatter5, scatter6, scatter7]
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='basic-error-bar')


# ***
# <p>
# <h1>Final Classification</h1><br>
# We are now able to make the final classification.
# </p>

# In[ ]:


# Choose the model 
model = lrcv['model'] # or baseModel, choose the model you want

features = new_ones_2

evaluate_models(copy.deepcopy([lrcv]), features)

# Fitting
model.fit(train_data[features], train_data['type'])
# Predicting
predicted_test = model.predict(test_data[features])


# In[ ]:


# SUBMISSION 
submission = pd.concat([test['id'], pd.DataFrame(predicted_test)], axis=1)
submission.columns = ['id', 'type']

submission.to_csv("new_sub.csv", header=True, index=False)

