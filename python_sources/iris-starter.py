#!/usr/bin/env python
# coding: utf-8

# # 1 &nbsp; Introduction
# 
# This notebook is a simple starter analysis for the Iris dataset.
# 
# Questions and feedback are welcome!

# ### Background
# 
# Some helpful links about the *Iris* family of flowers:
# 
# - [Flower Anatomy](http://www.northernontarioflora.ca/flower_term.cfm) from the Northern Ontario Plant Database
# - [*Iris setosa*](https://www.wildflower.org/plants/result.php?id_plant=IRSE),
# [*Iris versicolor*](https://www.wildflower.org/plants/result.php?id_plant=IRVE2),
# and
# [*Iris virginica*](https://www.wildflower.org/plants/result.php?id_plant=IRVI)
# from the Lady Bird Johnson Wildflower Center
# 

# ### License
# 
# My work is licensed under CC0:
# 
# - Overview: https://creativecommons.org/publicdomain/zero/1.0/
# - Legal code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt
# 
# All other rights remain with their respective owners.

# # 2 &nbsp; Preamble
# 
# The usual `python` preamble:
# 
# - `jupyter` magic
# - `numpy`
# - `pandas`
# - `seaborn` + `matplotlib`
# - `scikit-learn`

# Click the `Code` button to take a look at hidden cells.

# In[141]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from sklearn.utils import resample
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier

plt.rcParams['figure.figsize'] = (19,5)
sns.set(style='whitegrid', color_codes=True, font_scale=1.5)
np.set_printoptions(suppress = True, linewidth = 200)
pd.set_option('display.max_rows', 100)


# In[47]:


from typing import Sequence

def plot_confmat(
        y: pd.Series,
        y_hat: Sequence,
        rotate_x: int = 0,
        rotate_y: int = 'vertical') \
        -> None:
    """
    Plot confusion matrix using `seaborn`.
    """
    classes = y.unique()
    ax = sns.heatmap(
        confusion_matrix(y, y_hat),
        xticklabels=classes,
        yticklabels=classes,
        annot=True,
        square=True,
        cmap="Blues",
        fmt='d',
        cbar=False)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=rotate_x)
    plt.yticks(rotation=rotate_y, va='center')
    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')
    
def seq(start, stop, step=None) -> np.ndarray:
    """
    Inclusive sequence.
    """
    if step is None:
        if start < stop:
            step = 1
        else:
            step = -1

    if is_int(start) and is_int(step):
        dtype = 'int'
    else:
        dtype = None

    d = max(n_dec(step), n_dec(start))
    n_step = np.floor(round(stop - start, d + 1) / step) + 1
    delta = np.arange(n_step) * step
    return np.round(start + delta, decimals=d).astype(dtype)

def is_int(x) -> bool:
    """
    Whether `x` is int.
    """
    return isinstance(x, (int, np.integer))

def n_dec(x) -> int:
    """
    Number of decimal places, using `str` conversion.
    """
    if x == 0:
        return 0
    _, _, dec = str(x).partition('.')
    return len(dec)


# Let's store the path to our data and set a global seed for reproducible results, then get started.

# In[3]:


csv = '../input/Iris.csv'
seed = 0


# # 3 &nbsp; Data
# 
# We'll load the `csv` using `pandas` and take a look at some summary statistics.

# ## Peek
# 
# Let's load the first 10 rows of our dataset and make a todo list.

# In[4]:


pd.read_csv(csv, nrows=10)


# To do:
# 
# - Drop the `Id` column.
# - Convert `Species` to `category`.
# - Split the `DataFrame` into `X` and `y` (predictors and target feature, respectively).
# - Optional: rename columns and category levels to `snake_case`. (I won't do this for this notebook.)
# 
# Note that:
# 
# - `Species` is our target.
# - All of our features are floats, which simplifies preprocessing.

# ## Load Full Dataset
# 
# Let's load our data and take a look at `X` and `y`.

# In[5]:


def load_iris(csv, y='Species'):
    df = pd.read_csv(
        csv,
        usecols=lambda x: x != 'Id',
        dtype={y: 'category'},
        engine='c',
    )
    X = df.drop(columns=y)
    y = df[y].map(lambda s: s.replace('Iris-', ''))
    return X, y


# In[6]:


X, y = load_iris(csv)


# ## Predictors

# In[7]:


X.info()


# Key takeaways:
# 
# - No missing data.
# - 4 columns.
# - Our features are 2 dimensional: sepals and petals, so polynomial features (ie,  area) might help.
# - This is a very small dataset (5 KB), which gives us the luxury of fast model training.

# ## Target

# In[8]:


y.value_counts()


# We have a balanced dataset:
# - 3 species
# - 50 rows each

# ## Summary
# 
# - Goal: classify flower species
# - 150 observations
# - 4 predictors, all floats
# - 3 balanced target categories
# - No missing data

# Next, let's take a closer look at our data.

# # 4 &nbsp; Visualization
# 
# We'll use `seaborn` for 2 plots:
# 
# - A `pairplot` to see pairwise relationships.
# - A `boxplot` for feature scaling.

# ## Pair Plot

# In[9]:


def pair(X, y):
    sns.pairplot(pd.concat((X, y), axis=1, copy=False), hue=y.name, diag_kind='kde', size=2.2)


# In[10]:


pair(X, y)


# Pairwise comparisons separate our target quite well, especially *Iris setosa* (blue).

# Let's move on to the boxplot.

# ## Boxplot

# In[11]:


def box(X):
    plt.figure(figsize=(10,5))
    sns.boxplot(data=X, orient='v');


# In[12]:


box(X)


# This view of our features is useful for feature scaling.  Notably:
# 
# - All features use the same scale: cm.
# - Therefore, all features should be strictly positive.
# - Features occupy different value ranges (eg, sepal length: 4 to 8 cm, petal width: 0 to 3 cm).
# - And, features have different variances (eg, petal length vs sepal width).
# 
# Overall, we probably don't need to worry too much about feature scaling for this dataset.  Standardization (*z*-score scaling) should be fine.

# Next, let's train a simple baseline model.

# # 5 &nbsp; Baseline: Naive Bayes
# 
# The choice of model is arbitrary, but we want something convenient for our baseline.  So, let's use Gaussian naive Bayes.  It's simple and fast, and it works out of the box with no preprocessing and no tuning.

# In[13]:


def baseline(X, y):
    acc = GaussianNB().fit(X, y).score(X, y).round(4) * 100
    print(f'{acc}% accuracy')


# In[14]:


baseline(X, y)


# Great.  We our baseline is 96% accuracy.  Let's break down our error rate using a confusion matrix.

# In[55]:


def confuse(X, y):
    plt.figure(figsize=(4.2,4.2))
    model = GaussianNB().fit(X, y)
    plot_confmat(y, model.predict(X))


# In[56]:


confuse(X, y)


# As expected *Iris setosa* is easy to classify, and *versicolor* and *virginica* overlap a bit, but not much.

# Let's explore the performance of our baseline in the next section.

# # 6 &nbsp; On Cross Validation
# 
# The holy grail of machine learning is generalization.  We want to know how well our model performs on data it hasn't seen before.  Kaggle competitions use a hidden holdout set (aka the *test set*) to uniformly rank submissions, but we don't have that here.  So, let's use cross validation to simulate a holdout set.  The method is simple:
# 
# 1. Split the data into `train` and `test`.
# 2. Fit the model on `train`.
# 3. Measure accuracy on `test`.
# 4. Repeat to taste.

# ## How should we split our data?
# 
# - First, use a fixed seed for reproducible results.
# - Second, cross validation is often performed using the *k*-fold method, but for this section, we'll be using `sklearn`'s `StratifiedShuffleSplit` instead.  This gives us better control over training size, which is important for the next question.

# ## Where should we split our data?
# 
# This is somewhat arbitrary, so let's try a few different options.
# 
# - We'll use 10%, 20%, ..., 90% of our total data as `train`, and the remainder will be `test`.
# - We'll split our data 1000 times for each percentage.

# ## To do
# 
# - Gaussian naive Bayes classifier
# - 9 percentages, 10% to 90%
# - 1000 splits for each percentage
# - 9000 models total

# ## Results

# In[17]:


def cv(X, y, model):
    fractions = seq(0.1, 0.9, step=0.1)
    n_splits = 1000
    history = np.empty((n_splits, len(fractions)))
    
    for i, fr in enumerate(fractions):
        shuffle = StratifiedShuffleSplit(n_splits, train_size=fr, test_size=None, random_state=seed)
        for j, (idx_train, idx_dev) in enumerate(shuffle.split(X, y)):
            tr_X = X.iloc[idx_train]
            tr_y = y.iloc[idx_train]
            te_X = X.iloc[idx_dev]
            te_y = y.iloc[idx_dev]
            history[j,i] = model.fit(tr_X, tr_y).score(te_X, te_y)
    
    df = pd.DataFrame(history, columns=[f'{int(fr*150)}' for fr in fractions])
    
    sns.boxplot(data=df)
    plt.xlabel('Train Size')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy vs Training Size')


# In[18]:


cv(X, y, GaussianNB())


# Key takeaways:
# 
# - Our baseline model performs well across the board, starting at just 30 observations (20% of our data), with around 95% accuracy.
# - At `train` size 15, accuracy degrades quite a bit if we get unlucky (down to 65%), but overall, model accuracy is very consistent.
# 
# ### Beware
# 
# As `train` grows, `test` *shrinks*, because the split is complementary, and thus accuracy becomes less granular:
# 
# - At `train` size 30, each misclassification reduces accuracy by less than 1%.
# - At `train` size 135, the marginal reduction is over 6%.
# 
# This is why performance variance appears to degrade if you read the plot left to right, especially at 135 `train` size, but that comparison is *not* apples to apples.  The moral of the story is that `test` size matters.

# ## The Plan
# 
# Let's use 20% of our data (30 observations, stratified) to simulate `train`.  Then, the remaining 80% (120 observations) will be our `test`.  It's a fairly arbitrary choice, but hopefully this strikes a good balance between predictive power and generalization.
# 
# ### Aside
# 
# We won't be doing any model tuning or stacking in this notebook; otherwise, we'd want more breathing room in our `train` (maybe a 50/50/50 split).  We won't be splitting `train` at all, so our workflow will differ quite a bit from the Kaggle competition setup, which often splits data into at least 4 parts:
# 
# 1. Private leaderboard (final ranking)
# 2. Public leaderboard (rate limited cross validation)
# 3. Train/dev (unlimited cross validation)
# 4. Train/train (true `train` data)

# # 7 &nbsp; Train/Test Split
# 
# Let's follow the plan.

# In[19]:


def split(csv, train_size=30):
    X, y = load_iris(csv)
    return train_test_split(X, y, train_size=train_size, test_size=None, random_state=seed, stratify=y)


# In[20]:


X, test_X, y, test_y = split(csv)


# Click through the hidden cells if you're interested in repeating our analysis thus far for our new `X`/`y` pair.

# In[21]:


X.info()


# In[22]:


y.value_counts()


# In[23]:


pair(X, y)


# In[24]:


box(X)


# In[25]:


baseline(X, y)


# In[78]:


confuse(X, y)


# # 8 &nbsp; Models

# ## Preprocessing
# 
# Let's add some polynomial features and standardize `X`.

# In[64]:


preprocess_X = make_pipeline(
    PolynomialFeatures(interaction_only=True, include_bias=False),
    StandardScaler(),
)


# We'll leave `y` as is.

# In[65]:


preprocess_y = None


# In[66]:


def preprocess(csv, X_pipeline, y_pipeline):
    X, test_X, y, test_y = split(csv)
    
    if X_pipeline:
        X = X_pipeline.fit_transform(X)
        test_X = X_pipeline.transform(test_X)

    if y_pipeline:
        y = y_pipeline.fit_transform(y)
        test_y = y_pipeline.transform(y)
    
    return X, test_X, y, test_y


# In[67]:


X, test_X, y, test_y = preprocess(csv, preprocess_X, preprocess_y)


# ## Training
# 
# Here are the models we'll be using:

# In[68]:


model_dict = {
    'rf': RandomForestClassifier(random_state=seed, n_jobs=-1),
    'nb': GaussianNB(),
    'knn': KNeighborsClassifier(n_jobs=-1),
    'xt': ExtraTreeClassifier(random_state=seed),
    'gb': GradientBoostingClassifier(random_state=seed),
    'bagg': BaggingClassifier(random_state=seed, n_jobs=-1),
    'ada': AdaBoostClassifier(random_state=seed),
    'gp': GaussianProcessClassifier(random_state=seed, n_jobs=-1),
    'logit': LogisticRegression(random_state=seed),
}


# Now, let's go ahead and `fit` each model.

# In[69]:


def fit(model_dict, X, y):
    for m in model_dict.values():
        m.fit(X, y)


# In[70]:


fit(model_dict, X, y)


# ## Metrics
# 
# Since there's no leaderboard metric, let's take a look at accuracy, ROC area under curve (AUC), and log loss.

# In[252]:


def compute_acc(fit_model_dict, test_X, test_y):
    d = dict()
    for m in fit_model_dict.values():
        name = type(m).__name__.replace('Classifier', '')
        y_hat = m.predict(test_X)
        d[name] = accuracy_score(test_y, y_hat)
    
    return pd.Series(d)

def compute_roc_auc(fit_model_dict, test_X, test_y):
    d = dict()
    for m in fit_model_dict.values():
        name = type(m).__name__.replace('Classifier', '')
        y_hat = pd.get_dummies(m.predict(test_X))
        d[name] = roc_auc_score(pd.get_dummies(test_y), y_hat)
    
    return pd.Series(d)

def compute_log_loss(fit_model_dict, test_X, test_y):
    d = dict()
    for m in fit_model_dict.values():
        name = type(m).__name__.replace('Classifier', '')
        y_hat = m.predict_proba(test_X)
        d[name] = log_loss(test_y, y_hat)
    
    return pd.Series(d)


# In[253]:


metrics_dict = {
    'acc': compute_acc,
    'auc': compute_roc_auc,
    'log_loss': compute_log_loss,
}


# In[254]:


def compute_metrics(metrics_dict, fit_model_dict, test_X, test_y):
    metrics_results = dict()
    for k, v in metrics_dict.items():
        metrics_results[k] = v(fit_model_dict, test_X, test_y)
    return metrics_results


# In[228]:


metrics_results = compute_metrics(metrics_dict, model_dict, test_X, test_y)


# ### Metrics Plots
# 
# Next, let's plot our metrics results.

# In[275]:


def plot_acc(results):
    (results * 100).plot(kind='bar')
    plt.ylim(85, 100)
    plt.xticks(rotation='vertical')
    plt.title('Accuracy')

def plot_roc_auc(results):
    (results * 100).plot(kind='bar')
    plt.ylim(90, 100)
    plt.xticks(rotation='vertical')
    plt.title('Area Under ROC Curve (higher is better)')
    
def plot_log_loss(results):
    results.plot(kind='bar')
    plt.xticks(rotation='vertical')
    plt.title('Log Loss (lower is better)')

def plot_metrics(results):
    plt.figure(figsize=(7,18))
    plt.subplots_adjust(hspace=1)
    
    plt.subplot(3,1,1)
    plot_acc(results['acc'])
    
    plt.subplot(3,1,2)
    plot_roc_auc(results['auc'])
    
    plt.subplot(3,1,3)
    plot_log_loss(results['log_loss'])


# In[276]:


plot_metrics(metrics_results)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




