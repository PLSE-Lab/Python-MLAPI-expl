#!/usr/bin/env python
# coding: utf-8

# # Predicting whether a developer uses R or Python
# Myself being an avid Python user, I thought it'd be fun to see if based on this survey I could predict whether a given developer uses R or Python - and of course if so, which features allow the classifier to determine that. I'll try to keep the analysis as simple as possible and focus on clarity of code and analysis rather than on creating anything overly complex and detailed.
# 
# ## Conclusions?
# If you do not want to go through the notebook, the quick conclusion is that among data scientists and analysts, Python and R users are pretty similar. It is however possible to create pretty decent classifiers for predicting whether a user uses R or Python, and there are a few funny conclusions and reasonings to be found within those classifiers.
# 
# ## Table of Contents:
# * [Step 1: Preprocessing the data](#preprocessing)
#     * [Step 1.1. Dealing with multiple-choice columns](#pre1)
#     * [Step 1.2. Dealing with NaNs](#nan)
#     * [Step 1.3. Dealing with Collinearity](#pre2)
#     * [Step 1.4. Creating and splitting into X and y](#pre3)
# * [Step 2: Supervised prediction of R / Python](#supervised)
#     * [Step 2.1. Random Forest classifier](#rf)
#     * [Step 2.2. Logistic Regression classifier](#lr)
#     * [Step 2.3. xgBoost Classifier](#xgb)
#     * [Step 2.4. Linear Discriminant Analysis](#lda)
# * [Step 3: Unsupervised learning with R vs. Python](#unsupervised)
#     * [Step 3.1. Clustering Analysis](#clustering)
#     * [Step 3.2. PCA Biplots](#pca)
#     * [Step 3.3. ICA Biplots](#pca)
#     * [Step 3.4. Manifold learning](#manifold)
#     * [Step 3.5. Network-based Correlation Analysis](#manifold)

# In[ ]:


from copy import deepcopy
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import hdbscan
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.base import clone

import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from matplotlib.ticker import NullFormatter


# First things first, the data has to be prepared for analysis. In this dataset we have some columns where people have been able to select multiple values; this has to be extracted and put on a format which can be interpreted by the machine learning model. Since we're on kaggle, I've chosen to only look at data analysts and data scientists, ignoring entries from all other types of developers (e.g. it would be too easy to classify all backend developers as more likely being Python users). On the same note, I've chosen to ignore certain columns such as "FrameworksWorkedWith", since e.g. any user working with Django would be easily recognizable as a Python user.

# In[ ]:


# Read in the survey results, shuffle results
df = pd.read_csv('../input/survey_results_public.csv', low_memory=False).sample(frac=1)

# Columns with multiple choice options
MULTIPLE_CHOICE = [
    'CommunicationTools','EducationTypes','SelfTaughtTypes','HackathonReasons', 
    'DatabaseWorkedWith','DatabaseDesireNextYear','PlatformWorkedWith',
    'PlatformDesireNextYear','Methodology','VersionControl',
    'AdBlockerReasons','AdsActions','ErgonomicDevices','Gender',
    'SexualOrientation','RaceEthnicity', 'LanguageWorkedWith'
]

# Dev types - let's only look at data scientists
DEV_TYPES = [
    'Data or business analyst',
    'Data scientist or machine learning specialist'
]

# Columns which we are not interested in (predicting Python/R would be too easy with them)
DROP_COLUMNS = [
    'IDE', 'FrameworkWorkedWith', 'FrameworkDesireNextYear',
    'LanguageDesireNextYear', 'DevType', 'CurrencySymbol',
    'Salary', 'SalaryType', 'Respondent', 'Currency'
]


# # Step 1: Preprocessing data<a class="anchor" id="preprocessing"></a>
# 
# ## 1.1. Dealing with multiple-choice columns<a class="anchor" id="pre1"></a>
# Some of the columns allow multiple options; e.g. which methodologies people have worked with etc. For all these,  I'll create new one-hot-encoding columns for each option. For all the rest of the categorical variables, I simply create dummy columns using the convenience function included in pandas. Nice and easy.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Pick off data science types\ndf = df.loc[df.DevType.str.contains(\'|\'.join(DEV_TYPES)).fillna(False)]\n\n# Drop too easy columns\nprint(f">> Deleting columns with simple Python/R relations: {DROP_COLUMNS}")\ndf.drop(DROP_COLUMNS, axis=1, inplace=True)\n\n# Go through all object columns\nfor c in MULTIPLE_CHOICE:\n    \n    # Check if there are multiple entries in this column\n    temp = df[c].str.split(\';\', expand=True)\n\n    # Get all the possible values in this column\n    new_columns = pd.unique(temp.values.ravel())\n    for new_c in new_columns:\n        if new_c and new_c is not np.nan:\n            \n            # Create new column for each unique column\n            idx = df[c].str.contains(new_c, regex=False).fillna(False)\n            df.loc[idx, f"{c}_{new_c}"] = 1\n\n    # Info to the user\n    print(f">> Multiple entries in {c}. Added {len(new_columns)} one-hot-encoding columns")\n\n    # Drop the original column\n    df.drop(c, axis=1, inplace=True)\n        \n# For all the remaining categorical columns, create dummy columns\ndf = pd.get_dummies(df)')


# ## 1.2. Dealing with missing values
# We ahve quite a few missing values. For all dummy features I'll replace NaN with 0, and otherwide I'll replace with median.

# In[ ]:


# Fill in missing values
df.dropna(axis=1, how='all', inplace=True)
dummy_columns = [c for c in df.columns if len(df[c].unique()) == 2]
non_dummy = [c for c in df.columns if c not in dummy_columns]
df[dummy_columns] = df[dummy_columns].fillna(0)
df[non_dummy] = df[non_dummy].fillna(df[non_dummy].median())

print(f">> Filled NaNs in {len(dummy_columns)} OHE columns with 0")
print(f">> Filled NaNs in {len(non_dummy)} non-OHE columns with median values")


# ## 1.3. Dealing with Collinearity<a class="anchor" id="pre2"></a>
# Given the nature of our dataset, we have several features / columns which are highly collinear, which may throw off subsequent analysis. I'll just go with a very simply option of dropping columns that have >0.75 correlations with other columns

# In[ ]:


# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]

# Drop those columns
print(f">> Dropping the following columns due to high correlations: {to_drop}")
df = df.drop(to_drop, axis=1)


# ## 1.4. Creating and splitting into X and y<a class="anchor" id="pre3"></a>
# Having performed the encoding of all the categorical data, the next step is to get our labels (users using only R and users using only Python), and then split our dataset into X and y, which can then be fed to the classifier.

# In[ ]:


# Remove outliers for converted salary
print(">> Removing salary outliers")
df = df[df.ConvertedSalary < df.ConvertedSalary.mean() + df.ConvertedSalary.std()*3]

# Scale dataframe
print(">> Scaling non-dummy columns")
nondummy_columns = [c for c in df.columns if df[c].max() > 1]
scaled_df = deepcopy(df)
scaled_df.loc[:, nondummy_columns] = scale(df[nondummy_columns])

# Create target - ignore all users who do not use either
print(">> Getting R, Python and R&Python user indexes")
R_only_idx = (df.LanguageWorkedWith_R == 1) & (df.LanguageWorkedWith_Python == 0)
Python_only_idx = (df.LanguageWorkedWith_R == 0) & (df.LanguageWorkedWith_Python == 1)
R_and_Python_idx = (df.LanguageWorkedWith_R == 1) & (df.LanguageWorkedWith_Python == 1)

# Set the classes
scaled_df.loc[R_only_idx, 'RorPython'] = 0
scaled_df.loc[Python_only_idx, 'RorPython'] = 1
scaled_df.loc[R_and_Python_idx, 'RorPython'] = 2
scaled_df.dropna(subset=['RorPython'], axis=0, inplace=True)

# Split into X and y (with all Python / R users)
print(">> Storing subset with all Python and R users")
y_all = scaled_df['RorPython']
X_all = scaled_df.drop(['LanguageWorkedWith_Python', 'LanguageWorkedWith_R', 'RorPython'], axis=1)

# Split into X and y (with all Python-only and R-only users)
print(">> Storing subset with all Python-only and R-only users")
df_only = scaled_df[scaled_df.RorPython != 2]
y_only = df_only['RorPython']
X_only = df_only.drop(['LanguageWorkedWith_Python', 'LanguageWorkedWith_R', 'RorPython'], axis=1)


# # Step 2: Supervised prediction of R / Python<a class="anchor" id="supervised"></a>
# ## 2.1. Random Forest Classification<a class="anchor" id="rf"></a>
# Now that the data is ready, we can start trying to predict if a user prefers Python or R. I'm implementing the following small convenience function for performing 10-fold stratified cross-validation, and visualizing the results in terms of ROC, feature importance, and for the most important features, I also visualize the difference in fractions of R and Python users for each of those features.

# In[ ]:


def evaluate(clf, X, y, ignore_columns=[], plot_features=15):

    # Columns to use in classification
    use_cols = np.array([c for c in X.columns if c not in ignore_columns])

    # Create 10-fold cross validated predictions
    predicted_probas = cross_val_predict(
        clone(clf),        
        X[use_cols], y,
        cv=10,
        n_jobs=1, verbose=0,
        method='predict_proba'
    )

    # Fit classifier on all data
    full_clf = clone(clf)
    full_clf.fit(X[use_cols], y)

    # Create feature importance and explanations next to each other
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Show ROC plot
    skplt.metrics.plot_roc(y, predicted_probas, ax=axes[0])   
    
    # Extract feature importances from the random forest
    if hasattr(full_clf, 'feature_importances_'):
        axes[1].set_title("Feature importances")
        importances = full_clf.feature_importances_
        
        indices = np.argsort(importances)[::-1][:plot_features]        
    else:
        axes[1].set_title("Feature Coefficients")
        importances = full_clf.coef_[0]
        indices = np.argsort(np.abs(full_clf.coef_[0]))[::-1][:plot_features]        
    
    # Can we put on stds?
    if hasattr(full_clf, 'estimators_'):
        std = np.std([tree.feature_importances_ for tree in full_clf.estimators_], axis=0)        
    else:
        std = np.zeros(len(importances))    
    
    # Create coefficient or importances plot
    axes[1].bar(range(plot_features), importances[indices], color="r", yerr=std[indices], align="center")
    axes[1].set_xticks(range(plot_features))
    axes[1].set_xticklabels(use_cols[indices], rotation=90)
    axes[1].set_xlim([-1, plot_features])
    
    # Plot mean values
    X[use_cols[indices]].         groupby(y).mean().T.         rename(columns={0.0: "R", 1.0: "Python"}).         plot(kind='bar', ax=axes[2], title='Mean Value for Users')
    plt.show()
        
# Evaluate a RF model on all the data columns
clf_rf = ExtraTreesClassifier(n_jobs=-1, n_estimators=100, class_weight='balanced')
evaluate(clf_rf, X_only, y_only)


# Wow, at first glance that seems to be quite the amazing classifier with an ROC of of around 0.85. Looking at the feature importances though, it's clear that it has a pretty easy job, in that almost all developers who work with Ruby or Rust are R users. Looking closer, it seems that more users are using different languages (Java, C++, Linux, Bash, etc.), the more likely it is to be a Python user (not that surprising I reckon), and if you're on a Windows machine, you're more likely to be using R. Let's try to remove some of all these columns from the analysis, and see how good the model does then.

# In[ ]:


# Ignore columns with these prefixes
ignore_prefixes = ['LanguageWorkedWith_', 'PlatformWorkedWith_', 'DatabaseWorkedWith_', 'OperatingSystem_']
ignore_columns = [c for c in X_only.columns if any(check in c for check in ignore_prefixes)]

# Run the model and evaluate classifier
evaluate(clf_rf, X_only, y_only, ignore_columns=ignore_columns)


# Then we get an AUC_ROC of about 0.7, whic still isn't too bad. Without digging too much deeper, the model qualitatively suggests that:
# 
# * If you're looking to move towards Linux next year, you're more likely a Python user
# * If you studied statistics you're more likely R, and if computer science then Python
# * If you're young (18-24 years old), you're more likely Python user
# * If you do code competitions, you're more likely a Python user
# * If you want an android next year, you're more likely a Python user
# * If you want to learn SQL next year, more likely R user
# * If you user MS office, you're more likely an R user
# * If you want an Rasperry Pi next year, you're more likely an Python user
# * If you're a full time student, you're more likely to be a Python user
# * If you're using Agile methodology, you're more likely to be a Python user
# * If you're more worried than excited about AI, then you're more likely to be an R user

# ## Step 2.2. Logistic Regression Classification<a class="anchor" id="lr"></a>
# Let's see if a logistic regression classifier agrees with the outcome of the random forest classifier.

# In[ ]:


# Evaluate a RF model on all the data columns
clf_lr = LogisticRegression(class_weight='balanced', C=0.05)
evaluate(clf_lr, X_only, y_only, ignore_columns=ignore_columns)


# Seems to agree pretty well with the random forest features, and the classifier performs at about the same ROC

# ## Step 2.3. xgBoost Classifier<a class="anchor" id="xgboost"></a>
# Let us also try a slighly more advanced classifier, to see if we can create a better classifier

# In[ ]:


# Evaluate a RF model on all the data columns
clf_xgb = XGBClassifier(n_jobs=-1, n_estimators=100)
evaluate(clf_xgb, X_only, y_only, ignore_columns=ignore_columns)


# I'd like to see if we can improve a little bit on the xgBoost classifier, so inspired by [previous notebook](https://www.kaggle.com/nanomathias/bayesian-optimization-of-xgboost-lb-0-9769), I'll perform bayesian optimization to see if we can find some better parameters.

# In[ ]:


from skopt import BayesSearchCV

# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(
        n_jobs = 1,
        objective = 'binary:logistic',
        eval_metric = 'auc',
        silent=1,
        tree_method='approx'
    ),
    search_spaces = {
        'learning_rate': (0.01, 10.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 30),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 10000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (0.1, 10, 'log-uniform')
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 4,
    n_iter = 3,   
    verbose = 0,
    refit = True,
    random_state = 42
)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
# Fit the model
result = bayes_cv_tuner.fit(
    X_only[[c for c in X_only.columns if c not in ignore_columns]].values,
    y_only.values,
    callback=status_print
)


# Running this for about 200 iterations, I found that it wasn't finding any better models. The results from the best model is then as follows:

# In[ ]:


best_params = {
    'colsample_bylevel': 0.24573122383897958, 
    'colsample_bytree': 0.6265238053481696,
    'gamma': 1.3485673446209135e-06,
    'learning_rate': 0.035385067445099304,
    'max_delta_step': 18,
    'max_depth': 12, 
    'min_child_weight': 4,
    'n_estimators': 93,
    'reg_alpha': 1.9910199304506005e-05,
    'reg_lambda': 0.0021525020638473143,
    'scale_pos_weight': 0.1,
    'subsample': 0.8262150586465882
}
# Evaluate a RF model on all the data columns
clf_xgb = XGBClassifier(n_jobs=-1, **best_params)
evaluate(clf_xgb, X_only, y_only, ignore_columns=ignore_columns)


# Overall it doesn't seem like there's much to be gained from tuning the model; if one were to go further, one would have to start looking at feature engineering, but for me that's overkill for this notebook.

# ## Step 2.4. Linear Discriminant Analysis<a class="anchor" id="lda"></a>
# Given that we are trying to find differences between Python, R and Python&R users, we should  try to perform Linear Discriminant Analysis, which is very similar to PCA, but in addition to trying to maximize the variance in our component axes, we are also trying to maximize the separation between multiple classes. The technique is well-described in the scikit-documentation [here](http://scikit-learn.org/stable/modules/lda_qda.html).

# In[ ]:


# Colors
pyColor = '#ff7f0e'
rColor = '#1f77b4'
rpyColor = '#2ca02c'
classes = ['R', 'Python', 'R & Python']

def get_angle(v1, v2):
    """Calculate angle between two vectors"""
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.degrees(np.arctan2(sinang, cosang))

def annotate_embedding(loadings, pc_x, pc_y, ax, scaling=10, n_features=10, angle_thr=20):
    """Function for adding feature loadings to PCA or LDA embedding"""

    # Find the [n_features] longest vectors in the feature loading dataframe
    loadings['VectorLength'] = np.sqrt(loadings[pc_x]**2 + loadings[pc_y]**2)
    loadings = loadings.sort_values(by='VectorLength', ascending=False)
    
    # Plot each of the longest vectors 
    for feature, row in loadings.iloc[0:n_features].iterrows():
        vector = np.array([row[pc_x]*scaling, row[pc_y]*scaling])
        ax.arrow(0, 0, vector[0], vector[1], head_width=0.2, head_length=0.3)
        ax.annotate(feature, xy=(0, 0), xytext=(vector[0], vector[1]), fontsize=8)
    
    # Return sorted list of top features
    top_features = loadings.index.tolist()
    return top_features

def descriminant_analysis(clf):
    """Perform discriminant analysis and show results in plot"""
    
    # Get X without countries
    X_subset = X_all[[c for c in X_all.columns if 'Country' not in c]]

    # Run PCA on scaled numeric dataframe, and retrieve the projected data
    trafo = clf.fit_transform(X_subset, y_all)

    # The transformed data is in a numpy matrix. This may be inconvenient if we want to further
    # process the data, and have a more visual impression of what each column is etc. We therefore
    # put transformed/projected data into new dataframe, where we specify column names and index
    clf_df = pd.DataFrame(
        trafo,
        index=X_all.index,
        columns=["PC" + str(i + 1) for i in range(trafo.shape[1])]
    )

    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(224)    

    # How many features to check
    n_check = 20

    # Show biplots
    clf_df.loc[y_all==2].plot(
        kind="scatter", x="PC1", y="PC2", ax=ax1, label="R & Python", c=rpyColor, alpha=0.2
    )
    clf_df.loc[y_all==1].plot(
        kind="scatter", x="PC1", y="PC2", ax=ax1, label="Python", c=pyColor, alpha=0.2
    )
    clf_df.loc[y_all==0].plot(
        kind="scatter", x="PC1", y="PC2", ax=ax1, label="R", c=rColor, alpha=0.2
    )

    # Plot feature loadings on the biplot
    scalings = pd.DataFrame(clf.scalings_, columns=['PC1', 'PC2'], index=X_subset.columns)
    top_features = annotate_embedding(scalings, 'PC1', 'PC2', ax1, scaling=1.5, n_features=n_check)
    top_features = top_features[0:n_check]
    ax1.set_title(clf.__class__.__name__ + " embedding with scalings")    

    # Show ratio of users
    scaled_df[top_features+['RorPython']].         groupby('RorPython').mean().T.         rename(columns={0.0: "R", 1.0: "Python", 2.0: "Python & R"}).         plot(kind='bar', ax=ax2, title='Mean Value for Users - for most important feature')
    ax2.set_xticklabels([])

    # Show coefficients
    coef = pd.DataFrame(clf.coef_, columns=X_subset.columns, index=classes).T
    coef.loc[top_features].plot(kind='bar', ax=ax3)
    ax3.set_title(clf.__class__.__name__ + " coefficients - for most important feature")
    plt.show()


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Run analysis
descriminant_analysis(LinearDiscriminantAnalysis())


# This looks quite promising. In the left-hand plot I've plotted all the respondees on the LDA subspace; i.e. used LDA for dimensionality reduction. On top of that plot, I've also put the "loadings" or "scalings", which indicates to some degree how we should intepret this subspace in terms of our original features, using the same function as previously for PCA. On the right-hand side I've picked the most important features, and plotted the mean value of those for the different groups (top) and the LDA coefficients for different groups (bottom). A few of the conclusions from previous re-occur, and a few fun new ones pop up, i.e.
# 
# * More python users have had their adblocker disabled during the last month. Also more have adblocker installed though.
# * People who've been coding for 3-11 years more likely Python users, 12+ years more likely R.
# * People using Python&R are more often moderately happy than people using only one of the tools (intersting?)

# # Step 3: Unsupervised learning with R vs. Python<a class="anchor" id="unsupervised"></a>
# Now that we have the data processed, we may as well pass it through a few unsupervised learning algorithms to see if we can find any patterns in it, and compare it for R users vs. Python users. 
# 
# ## Step 3.1. Cluster Analyses<a class="anchor" id="clustering"></a>
# The first thing that springs to mind when we are talking about unsupervised learning is clustering. I'll go for two clustering algorithms; HDBSCAN and K-Means. The reasons for chosing HDBSCAN of other typical clustering algorithms (K-Means, etc.) are nicely summarized [here](http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html) I'm primarily including K-means as well simply just as a reference, since I'm not too fond of that algorithm.

# In[ ]:


def get_cluster_colors(clusterer, palette='Paired'):
    """Create cluster colors based on labels and probability assignments"""
    n_clusters = len(np.unique(clusterer.labels_))
    color_palette = sns.color_palette(palette, n_clusters)
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
    if hasattr(clusterer, 'probabilities_'):
        cluster_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
    return cluster_colors

# Prepare figure
_, ax = plt.subplots(1, 3, figsize=(20, 5))
settings = {'s':50, 'linewidth':0, 'alpha':0.2}

print(">> Clustering using HDBSCAN")
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
clusterer.fit(X_all)

print(">> Calculating elbow plot for KMeans")
kmeans = KMeans(random_state=42)
skplt.cluster.plot_elbow_curve(kmeans, X_all, cluster_ranges=[1, 5, 10, 20], ax=ax[0])

print(">> Dimensionality reduction using TSNE")
projection = manifold.TSNE(init='pca', random_state=42).fit_transform(X_all)

print(">> Clustering using K-Means")
kmeans = KMeans(n_clusters=6).fit(X_all)

# PLot on figure
ax[1].scatter(*projection.T, c=get_cluster_colors(clusterer), **settings)
ax[2].scatter(*projection.T, c=get_cluster_colors(kmeans), **settings)
ax[1].set_title('HDBSCAN Clusters')
ax[2].set_title('K-Means Clusters')
plt.show()


# In the left-most figure above we can see from the elbow plot that the optimal number of K-Means clusters is probably around 6. In the middle and right-most figures, we see the identified clusters as visualized on a default TSNE embedding (we'll play more with dimensionality reduction in following sections). 
# 
# It is observed that HDBSCAN filters out a lot of the datapoints as noise, and only idenfies a few clusters, which actually overall correspond nicely to groups of points in the TSNE embedding.  It is important to remember that TSNE **does not** preserve distances or density from the original dataset, rather it tries to preserve nearest neighbors, and as a result can create "fake" patterns, which do not neccesarily have any easily intepretable meaning. Still, the K-Means still manages to pick out the groups of respondees suggested by TSNE, so it is definitely interesting to analyze these groups a bit further. Note, all clustering is performed in the original >700 dimensional space, and are then shown in the embedded 2D space.
# 
# Let us try to inspect these identified clusters, and see if we can find what is different within these clusters as compared to the average of all the users. I'll only look at the two clusters identified by the HDBSCAN algorithm for now.

# In[ ]:


# Get number of clusters identified by HDBSCAN
unique_clusters = [c for c in np.unique(clusterer.labels_) if c > -1]

# Placeholder for our plotting
_, axes = plt.subplots(1, len(unique_clusters), figsize=(20, 5))

# Go through clusters identified by HDBSCAN
for i, label in enumerate(unique_clusters):
    
    # Get index of this cluster
    idx = clusterer.labels_ == label
    
    # Identify feature where the median differs significantly
    median_diff = (X_all.median() - X_all[idx].median()).abs().sort_values(ascending=False)
    
    # Create boxplot of these features for all vs cluster
    top = median_diff.index[0:20]
    temp_concat = pd.concat([X_all.loc[:, top], X_all.loc[idx, top]], axis=0).reset_index(drop=True)
    temp_concat['Cluster'] = 'Cluster {}'.format(i+1)
    temp_concat.loc[0:len(X_all),'Cluster'] = 'All respondees'
    temp_long = pd.melt(temp_concat, id_vars='Cluster')
    
    sns.boxplot(x='variable', y='value', hue='Cluster', data=temp_long, ax=axes[i])
    for tick in axes[i].get_xticklabels():
        tick.set_rotation(90)
    axes[i].set_title(f'Cluster #{i+1} - {idx.sum()} respondees')


# Going into the schema for the questions, the following things can be said about the two clusters who stick out:
#     
# On my run, this gave two interesting clusters, both of which consists primarily of female developers, that overall have differences from the average. It's quite fun to go over the figures to see these differences compared to the average user.

# ## Step 3.2. Principal Component Analysis<a class="anchor" id="pca"></a>
# Let us start out by performing a PCA analysis on our data; hopefully the first few components will contain a lot of information about the dataset.

# In[ ]:


# Create a PCA object, specifying how many components we wish to keep
pca = PCA(n_components=50)

# Run PCA on scaled numeric dataframe, and retrieve the projected data
pca_trafo = pca.fit_transform(X_all)

# The transformed data is in a numpy matrix. This may be inconvenient if we want to further
# process the data, and have a more visual impression of what each column is etc. We therefore
# put transformed/projected data into new dataframe, where we specify column names and index
pca_df = pd.DataFrame(
    pca_trafo,
    index=X_all.index,
    columns=["PC" + str(i + 1) for i in range(pca_trafo.shape[1])]
)

# Create two plots next to each other
_, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = list(itertools.chain.from_iterable(axes))

# Plot the explained variance# Plot t 
axes[0].plot(
    pca.explained_variance_ratio_, "--o", linewidth=2,
    label="Explained variance ratio"
)

# Plot the cumulative explained variance
axes[0].plot(
    pca.explained_variance_ratio_.cumsum(), "--o", linewidth=2,
    label="Cumulative explained variance ratio"
)

# Show legend
axes[0].legend(loc="best", frameon=True)
    
# Feature loadings on each component
loadings = pd.DataFrame(
    pca.components_,
    index=['PC'+str(i+1) for i in range(len(pca.components_))],
    columns=X_all.columns
).T

# Show biplots
for i in range(1, 4):
    
    # Components to be plottet
    x, y = "PC"+str(i), "PC"+str(i+1)
    
    # Plot biplots
    settings = {'kind': 'scatter', 'ax': axes[i], 'alpha': 0.2, 'x': x, 'y': y}
    pca_df.loc[y_all==2].plot(label='Python & R', c=rpyColor, **settings)
    pca_df.loc[y_all==1].plot(label='Python', c=pyColor, **settings)
    pca_df.loc[y_all==0].plot(label='R', c=rColor, **settings)
    
    # Show annotations on the plot
    annotate_embedding(loadings, x, y, axes[i], scaling=15, n_features=20, angle_thr=20)
    
# Show the plot
plt.show()


# So with the first few components we still have around 10% of the variation in the dataset in total, which is OK. We do not see any clear patterns in the PCA, and no clear cluster of R and Python users - this is to be expected given that if we had clear clusters, then it'd be much easier to have done the classification in the previous section. Based on the PCA of the first few components, R and Python users qualitatively seem more or less the same, although there are some areas in the PCA where there might be more non-R users. This is something that could be investigated further.
# 
# ## Step 3.3. Independent Component Analysis<a class="anchor" id="pca"></a>
# Let us also try to perform an independent component analysis (ICA) of the data to see how that fares. This is basically a computational method for separating multivariate signal into additive components (independent components). It's used in pretty much the same manner in sklearn, so let's just see what it does:

# In[ ]:


from sklearn.decomposition import FastICA

# Create a ICA object, specifying how many components we wish to keep
ica = FastICA(n_components=50, algorithm='deflation')

# Run ICA on scaled numeric dataframe, and retrieve the projected data
ica_trafo = ica.fit_transform(X_all)

# The transformed data is in a numpy matrix. This may be inconvenient if we want to further
# process the data, and have a more visual impression of what each column is etc. We therefore
# put transformed/projected data into new dataframe, where we specify column names and index
ica_df = pd.DataFrame(
    ica_trafo,
    index=X_all.index,
    columns=["IC" + str(i + 1) for i in range(ica_trafo.shape[1])]
)

# Create two plots next to each other
_, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = list(itertools.chain.from_iterable(axes))


# Show legend
axes[0].legend(loc="best", frameon=True)
    
# Feature loadings on each component
loadings = pd.DataFrame(
    ica.components_,
    index=['IC'+str(i+1) for i in range(len(ica.components_))],
    columns=X_all.columns
).T

# Show biplots
for i in range(0, 4):
    
    # Components to be plottet
    x, y = "IC"+str(i+1), "IC"+str(i+2)
    
    # Plot biplots
    settings = {'kind': 'scatter', 'ax': axes[i], 'alpha': 0.2, 'x': x, 'y': y}
    ica_df.loc[y_all==2].plot(label='Python & R', c=rpyColor, **settings)
    ica_df.loc[y_all==1].plot(label='Python', c=pyColor, **settings)
    ica_df.loc[y_all==0].plot(label='R', c=rColor, **settings)
    
# Show the plot
plt.show()


# Interestingly, where the PCA didn't find much of any patterns in the first few components, here we actually do seem to find some patterns, although these patterns to not seem to pertain to R or Python users specifically. Hence, let's not go forward with this techniuqe either.

# ## Step 3.4. Manifold Learning<a class="anchor" id="manifold"></a>
# Manifold learning are methods of non-linear dimensionality reduction; i.e. essentially reducing the dimensionality of very high-dimensional datasets while trying to retain certain characteristics between samples, such as their similarity etc. Plenty of these methods are implemented in scikit-learn, so let's try a bunch of them to see their results, as well as see if we can see a split between R and Python users within these embedded dimensions.

# In[ ]:


# Settings
n_neighbors = 10

# Figure with all embeddings
_, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = list(itertools.chain.from_iterable(axes))

# Other manifold methods
models = {
    'LLE': manifold.LocallyLinearEmbedding(n_neighbors, method='standard', eigen_solver='dense', n_jobs=4),
    'LTSA': manifold.LocallyLinearEmbedding(n_neighbors, method='ltsa', eigen_solver='dense', n_jobs=4),
    'Hessian LLE': manifold.LocallyLinearEmbedding(n_neighbors, method='hessian', eigen_solver='dense', n_jobs=4),
    'Modified LLE': manifold.LocallyLinearEmbedding(n_neighbors, method='modified', eigen_solver='dense', n_jobs=4),
    'Isomap': manifold.Isomap(n_neighbors, n_jobs=4),
    'MDS': manifold.MDS(max_iter=100),
    'SpectralEmbedding': manifold.SpectralEmbedding(n_neighbors=n_neighbors, n_jobs=4),
    'TSNE': manifold.TSNE(init='pca')
}
for i, (label, model) in tqdm(enumerate(models.items())):
    
    # Create embedding
    Y = model.fit_transform(X_all)
    
    # Add plot
    axes[i].scatter(Y[y_all==2, 0], Y[y_all==2, 1], label='Python & R', c=rpyColor, alpha=0.5)
    axes[i].scatter(Y[y_all==1, 0], Y[y_all==1, 1], label='Python', c=pyColor, alpha=0.5)
    axes[i].scatter(Y[y_all==0, 0], Y[y_all==0, 1], label='R', c=rColor, alpha=0.5)    
    axes[i].legend(loc='best')
    axes[i].xaxis.set_major_formatter(NullFormatter())
    axes[i].yaxis.set_major_formatter(NullFormatter())
    axes[i].set_title(label)
    
# Show figure
plt.axis('tight')
plt.show()


# It's pretty fun to  see how the different algorithms work differently; the code and idea is basically just adapted from [scikit learn example](http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html). For some of the `LocallyLinearEmbedding` methods it seems like certain Python users are put into areas where there are no R users, but that might as well be attributed to the fact that we simply have more Python users in the dataset, and therefore a higher propensity of "weird" people that fall outside the normal. MDS, IsoMap and TSNE all show that both Python and R users are very similarly distributed, so again the conclusion is that Python and R users are more or less the same.
# 
# ## Step 3.5. Correlation-based Network Analysis
# It could also be fun to do some network analysis on the data - in the following I draw networks where edges are drawn between users that have a correlation coefficient above a given threshold with each other.

# In[ ]:


# Sample subset of users
samples = 500
data = pd.concat([
    X_all[y_all == 0].sample(samples),
    X_all[y_all == 1].sample(samples),
    X_all[y_all == 2].sample(samples)
], axis=1)

# Calculate correlation matrix for users
corr = data.T.corr()

# Set colors in dataframe
data.loc[y_all == 0, 'color'] = rColor
data.loc[y_all == 1, 'color'] = pyColor
data.loc[y_all == 2, 'color'] = rpyColor

# Transform it in a links data frame (3 columns only):
links = corr.stack().reset_index()
links.columns = ['from', 'to','value']

# Prepare plot
_, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, thr in enumerate([0.25, 0.3, 0.35, 0.4, 0.45]):

    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    links_filtered=links.loc[ (links['value'] > thr) & (links['from'] != links['to']) ]
    links_filtered

    # Build your graph
    G=nx.from_pandas_dataframe(links_filtered, 'from', 'to', create_using=nx.Graph())

    # Get colors of users (in proper order)
    colors_ordered = data.reindex(G.nodes())['color']

    # Plot the network:
    nx.draw(
        G, 
        pos=nx.spring_layout(G),
        with_labels=False, 
        node_color=colors_ordered, 
        node_size=10,
        edge_color='black',
        linewidths=5,
        ax=axes[i]
    )
    axes[i].axis('on')
    axes[i].set_title(f"Correlation threshold: {thr}")

# Save figure
plt.savefig('network_analysis.png')
plt.show()


# This looks extremely interesting - for low thresholds it seems like three groups are clearly separated by each other; python, R, and Python-R users. Either I did something wrong with the coloring, or... will investigate more when I have time.

# In[ ]:




