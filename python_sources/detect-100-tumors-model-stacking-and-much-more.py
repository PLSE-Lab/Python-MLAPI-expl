#!/usr/bin/env python
# coding: utf-8

# # Detect 100% Malignant Tumors! - A Machine Learning Project 
# ## *Applying and discussing a wide variety of Machine Learning techniques to detect Malignant Tumors* 
# ---
# ## Introduction
# <p>
#    On this kernel we will go through the whole process of developing a Machine Learning model - from EDA to parameter tuning and model stacking. We will use the Breast Cancer dataset to try to predict if a tumor is benign or malignant. This dataset was obtained on <a href="https://www.kaggle.com/uciml/breast-cancer-wisconsin-data">kaggle</a>. The link contains the complete description but all you need to know to understand this analysis will be defined here. 
# </p>
# 
# <p>
#     I put a few comments on throughout the analysis to either clarify some points or give my opinion on a subject. Those are presented in blockquotes colored in dark blue. <br>
# <blockquote>
#     <font color="darkblue">This is a comment!</font>
# </blockquote>
# I hope you enjoy this work and can get some useful insight or piece of code from it.</p>
# 
# **Keywords**:<br>
# Python, Machine Learning, Model Stacking, Feature Engineering, Health
# 
# ### Contents
# 
# ![](https://i.imgur.com/TZCFAfs.png)
# 
# 
# ### TLDR Version
# 
# This dataset contains information on 569 breast tumors and the mean, standard error and worst measures for 10 different properties. I start with an EDA analysing each properties' distribution, followed by the pair interactions and then the correlations with our target: the tumor diagnosis.
# 
# After the EDA I set up 10 out-of-the-box models for a first evaluation and use cross-validation to measure them. I use Recall instead of Accuracy or F1-Score since I want to detect all malignant tumors. After the first results I analyse features importances, do a single round of feature selection and evaluate the models again. By the end of the chapter I analyse model errors and from the 10 first models I choose 5 for model tuning: Logistic Regression, SVC, Random Forest, Gradient Boosting and KNN.
# 
# I then proceed to tune the five models using GridSearchCV and prepare the data for model stacking by predicting probabilities for both train and test sets. Using Logistic Regression as a second-level model, I tune its parameters and finish the construction phase.
# 
# Finally, I test all first level models and the stacked Logistic Regression on our untouched test-set. For the first level models, using regular 0.5 threshold Logistic Regression performed best with 95,8% Recall. By lowering the threshold SVC and Logistic Regression tied with over 98% recall with SVC having a higher Accuracy. By using the model-stacking technique, Logistic Regression was able to obtain 100% Recall on the test set. On the last chapter I summarize the findings and conclusions.
# 
# On Annex - A I repeat a few Machine Learning steps using SMOTE to generate new data points making the data balanced. 
# 
# On Annex - B I use three different dimensionality reduction techniques to see if I can reduce the dataset and still get a good test score.

# ---

# ---

# # 1 - The Dataset
# ---
# ## 1.1 - Introducing the Data

# ### General Information
# - Original format: csv
# - Dataset shape: 569 x 33 (rows x columns)
# - Granularity: Each row derives from an unique sample of breast mass
# - There are no null values in this data.
# - The values are in different scales
# 
# ### Features in the dataset
# For each sample ten properties were measured:
# 
# <ol>
#     <li><b>Radius</b> - Mean distances from center to points on the perimeter</li>
#     <li><b>Texture</b> - Standard deviation of gray scale values</li>
#     <li><b>Perimeter</b></li>
#     <li><b>Area</b></li>
#     <li><b>Smoothness</b> - Local variation in radius lengths</li>
#     <li><b>Compactness</b> - Perimeter^2/Area - 1</li>
#     <li><b>Concavity</b> - Severity of concave portions of the contour</li>
#     <li><b>Concave points</b> - Number of concave portions of the contour</li>
#     <li><b>Simmetry</b></li>
#     <li><b>Fractal Dimension</b> - Coastline approximation - 1 </li>
# </ol>
# 
# <blockquote>
#     <font color='darkblue'>
#         <b>From <a href="https://en.wikipedia.org/wiki/Fractal_dimension">wikipedia</a>:</b>
#         <br><i>[...] a <b>fractal dimension</b> is a ratio providing a statistical index of complexity comparing how detail in a pattern (strictly speaking, a fractal pattern) changes with the scale at which it is measured.</i>
#     </font>
#     </blockquote>
#     
# And for each of these properties we have three calculated values:
# - **Mean**
# - **Standard Error**
# - **Worst** (Average of the 3 largest values)
# 
# All the measures are float types.
# 
# ### Target
# Our target is the categorical column *diagnosis* with either B (benign) or M (malignant).<br>
# There are 357 benign classes and 212 malignant classes - roughly **37% malignant tumors**.

# <hr/>

# ---

# ## 1.2 -  Importing Libraries

# We need only the basic tools for an EDA for now.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='whitegrid', rc={'axes.grid': False})


# In[ ]:


dataset = pd.read_csv('../input/data.csv')
dataset.sample(5)


# In[ ]:


# Our last column is just an error in the data reading. Dropping it
dataset = dataset.drop(['id', 'Unnamed: 32'], axis=1)

# Creating a binary target column to allow some data manipulations later on
dataset['Target'] = dataset['diagnosis'].map({'B':0, 'M':1})

# Getting lists with features. This will be useful on visualization
mean_feats = np.concatenate([['diagnosis'], dataset.iloc[:,1:11].columns.tolist()])
error_feats = np.concatenate([['diagnosis'], dataset.iloc[:,11:21].columns.tolist()])
worst_feats = np.concatenate([['diagnosis'], dataset.iloc[:,21:31].columns.tolist()])


# <hr>

# ## 1.3 - Defining Train/Test sets

# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.3,
                               stratify=dataset['Target'], random_state=42) 


# Using the stratify parameter we can guarantee that both our train and test sets have the same proportion of both classes. We must make sure our train data is as close as possible to the data it is going to be evaluated on.

# In[ ]:


train_diag = train.diagnosis.value_counts() / train.shape[0]
train_diag.rename('Train', inplace=True)
test_diag = test.diagnosis.value_counts() / test.shape[0]
test_diag.rename('Test', inplace=True)

pd.concat([train_diag, test_diag], axis=1)


# <hr/>

# <hr>

# # 2 - Exploratory Data Analysis
# Now we will look at individual features and combinations of features

# <hr>

# ## 2.1 - Data Distributions
# 
# We are dealing with three measures: mean, std. error and the 'worst'. We should look at each separatedly.
# 
# We can plot the distribution and look for skewness on the mean values but there is not much we can obtain from plotting distributions of the other two: 
# - Std. error is already a parameter obtained from a distribution and only have positive values so it's likely that we find it to be right-skewed.
# - Worst is a biased subsample of the measure's samples
# 
# The <a href='https://en.wikipedia.org/wiki/Central_limit_theorem'>Central Limit Theroem</a> states that the distribution of the mean values should look like a normal distribution. Let's explore that.

# In[ ]:


train.iloc[:,1:11].hist(figsize=(10,12), bins=20, layout=(5,2), grid=False)
plt.tight_layout();


# We can see that some features are pretty skewed. We can measure its skewness using pandas *skew* method and we can try comparing it to a log transformation of the same values to see if we can reduce the skewness.

# In[ ]:


log_means = np.log1p(train.iloc[:,1:11])

skewness = pd.DataFrame({'Original Skewness':train.iloc[:,1:11].skew(),
                         'Log Transformed':log_means.skew()})
skewness['Skewness Reduction'] = skewness['Original Skewness'] - skewness['Log Transformed']
skewness


# We managed to greatly reduce skewness on **Radius, Texture, Perimeter and Area**. The other features were barely influenced by our log transformation. 
# 
# There are four features with skewness higher than one after the log transformation: **Compactness, Concavity, Concave Points and Fractal Dimension**. Perhaps the measure error on them is higher or maybe it is somehow biased. Let's explore how are the standard errors for each measure!

# In[ ]:


measure_index = ['radius', 'texture', 'perimeter', 'area',
                'smoothness', 'compactness', 'concavity',
                'concave points', 'symmetry', 'fractal_dimension']

measure_data = np.c_[train.iloc[:,1:11].mean().values, train.iloc[:,11:21].mean().values]

measure_df = pd.DataFrame(data=measure_data, columns=['Mean', 'Error'], index=measure_index)

measure_df['Error pct'] = 100 * measure_df['Error'] / measure_df['Mean']

measure_df


# This might explain part of our high skewness: from our four highly skewed features, three of them have standard errors of more than 20%! 
# 
# Many things can cause that (e.g. uncallibrated measuring instruments). I don't have any other ideas to explore on fractal dimensions distribution for now.
# 
# For the log transformations: we will come back to them later on Chapter 5.

# <hr>

# ## 2.2 - Features Overlook 
# We can use seaborn's amazing pairplot to give a first overview on all features and some pair interactions. 
# 
# ### Mean Features Plot
# 
# There are a few things to point out on this plot.
# - From all the histograms in the grid's diagonal plots, only fractal dimension has no visual impact on the tumor's class. That is also observed in all plots on the last row/column. This is convenient because Fractal Dimension is the unexplained skewed feature we've just talked about. This is a strong candidate for a feature selection later on.
# - The second lowest visual impact (I'm saying visual because we will see some numbers later) is on symmetry.  
# - All the other features appear to have a significant impact on the classification of tumors and the scatterplots look quite 'separable'. 
# - We also can observe some 'pretty plots' on the related geometrical features Radius, Area and Perimeter, which is to be expected. This high correlation between features might be a problem for some ML algorithms

# In[ ]:


sns.set(style='whitegrid', font_scale=1.35, rc={'axes.grid': False})


# In[ ]:


p = sns.pairplot(train[mean_feats], hue='diagnosis',
             plot_kws={'alpha':0.6}, palette='magma')

plt.subplots_adjust(hspace=0.05, wspace=0.05)
handles = p._legend_data.values()
labels = p._legend_data.keys()
p.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=2)
p.fig.set_dpi(80);


# ### Error Features Plot
# This one surprised me at first. 
# 
# I didn't expect to find anything here - and most features' errors don't appear to have an impact - but look at Area, Perimeter, Radius and Compactness. The data suggests the higher the error on these features, the higher the chance of having a malignant tumor. How can we interpret that?
# 
# Let's remember how Standard error is calculated: by dividing the standard deviation by the squareroot of the sample size.
# 
# $$SE = {\sigma\over \sqrt{n}}.$$
# 
# I will assume that the sample sizes do not change for each tumor sample, so the Standard error's variation is due to the Standard Deviation only. 
# 
# Assuming that is the case, we can interpret that the malignant tumors have higher irregularity on their geometry, which causes the higher standard deviation!

# In[ ]:


p = sns.pairplot(train[error_feats], hue='diagnosis',
             plot_kws={'alpha':0.6, }, palette='magma')

plt.subplots_adjust(hspace=0.05, wspace=0.05)
handles = p._legend_data.values()
labels = p._legend_data.keys()
p.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=2)
p.fig.set_dpi(80);


# ### Worst Features Plot
# These plots look very similar to the previous one. This is to be expected since the worst features are subsamples of the mean data. 
# 
# It is hard to tell which one is more important for a predicting model only by looking at those visuals. We need to get some numbers to see if there is a significant difference.

# In[ ]:


p = sns.pairplot(train[worst_feats], hue='diagnosis',
                 plot_kws={'alpha':0.6}, palette='magma')

plt.subplots_adjust(hspace=0.05, wspace=0.05)
handles = p._legend_data.values()
labels = p._legend_data.keys()
p.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=2)
p.fig.set_dpi(80);


# ---

# ## 2.3 - Correlations
# To calculate the correlations we can use the pandas *corr* method. 
# 
# To visualize it better we can use the classic seaborn's heatmap - which is perfectly fine - but I will plot it using horizontal bar charts. 
# <blockquote>
#     <font color='darkblue'>
#             <b>The downside of not plotting a heatmap</b> is that we do not see how features are correlated to each other: there might be redundant features we don't need to feed a machine learning model. We can already see highly correlated features from our previous plots (e.g. Perimeter and Area), but I've chosen to keep them all and let the algorithms decide for them selves which ones are important and which ones aren't (feature selection and regularization).
#     </font>
#     </blockquote>

# In[ ]:


sns.set(style='whitegrid')


# In[ ]:


def feat_class(feat):
    if 'worst' in feat:
        return 'Worst'
    elif 'mean' in feat:
        return 'Mean'
    elif 'se' in feat:
        return 'Standard Error'


# In[ ]:


corrs = train.corr()[['Target']].sort_values('Target',
                                             ascending=False)[1:].reset_index()
corrs.rename(columns={'index':'Features'}, inplace=True)
corrs['Class'] = corrs['Features'].apply(feat_class)
corrs['Main'] = corrs['Features'].apply(lambda x: x.split('_')[0])


# ### Correlation by Feature Type
# First, lets see if we can find a predominant type of feature (*worst, mean or se*). Did we visualize it correctly in the previous plots?

# In[ ]:


fig, ax = plt.subplots(figsize=(8,7), dpi=80)
sns.barplot(data=corrs, x='Target', y='Features', ax=ax,
            hue='Class', dodge=False, palette='tab10')
ax.legend(bbox_to_anchor=(1.0, 1.0), loc=2)
ax.xaxis.tick_top()
ax.xaxis.label.set_visible(False)
ax.set_xlim(-0.1, 1.0)
ax.yaxis.label.set_visible(False)
ax.set_title('Pearson Correlation Between Target and Features by Feature Type');


# **Insights from the plot:**
# - At a first look, Standard Error seems to be the least important kind of measure we are dealing with (of the 6 lowest, 5 are standard error). We correctly pointed out that it did have an impact but only on a few features (radius, area and perimeter).
# - Aside from that, Worst has the top 3, follower by Mean;

# ### Correlation by Main Features
# Next we will plot the same graph but grouping the correlations by their main features (area, radius, etc... ).

# In[ ]:


plot_ord = corrs.sort_values('Features')['Features']
hue_ord = corrs.sort_values('Main')['Main'].unique()

fig, ax = plt.subplots(figsize=(8,7), dpi=80)
sns.barplot(data=corrs, x='Target', y='Features', ax=ax, order=plot_ord,
            hue='Main', hue_order=hue_ord, dodge=False, palette='Paired')
ax.legend(bbox_to_anchor=(1.0, 1.0), loc=2)
ax.xaxis.tick_top()
ax.xaxis.label.set_visible(False)
ax.yaxis.label.set_visible(False)
ax.set_xlim(-0.1, 1.0)
ax.set_title('Pearson Correlation Between Target and Features by Main Feature');


# <b>Insights from the plot</b>:
# <ul>
# <li>We can observe that <u>for all features except for Fractal Dimension have a similar pattern</u>: The two highest correlated feature types are WORST and MEAN and the lowest is the STANDARD ERROR.</li>
# <li>Fractal dimension has been the exception since 3.1. Apparently all that matters in terms of this feature are the worst measures.</li>
# </ul>
# 
# That said, we must remember that Pearson's correlation can only measure two individual features and we can't see how the combination of features influence in our target. As I've mentioned: I will keep them and let my model decide.

# ### Chapter Recap:
# - We've analysed how our features impact on our target.
# - We've pointed out that there are many features correlated to each other
# 
# Time to do some machine learning.

# <hr>

# <hr>

# # 3 - First Models

# On this section we will:
# - Pick different out-of-the-box models and evaluate them in our training data;
# - See if the first results give us any tips on how to improve our data somehow and test some ideas (feature engineering);
# - Choose the top five most promising and distinct models
# 
# The models we will be using are:
# - Logistic Regression
# - LDA
# - Support Vector Classifier (SVC)
# - Linear SVC
# - Decision Tree
# - Random Forests
# - Gradient Boos Classifier
# - AdaBoost Classifier
# - XGB
# - K-Nearest Neighbors

# In[ ]:


# Importing Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

# Importing other tools
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


# <hr>

# ## 3.1 - Setting Up the Training

# We have a small dataset. In order to make the most out of it we will be using cross-validation to evaluate our models . 
# 
# First lets create the models with standard parameters.

# In[ ]:


# Defining random seed
seed=42

# Creating Models
logreg = LogisticRegression(solver='lbfgs', random_state=seed)
lda = LinearDiscriminantAnalysis()
svc = SVC(random_state=seed, probability=True)
lin_svc = LinearSVC(random_state=seed)
l_svc = CalibratedClassifierCV(lin_svc, cv=5)
dtree = DecisionTreeClassifier(random_state=seed)
rf = RandomForestClassifier(10, random_state=seed)
gdb = GradientBoostingClassifier(random_state=seed)
adb = AdaBoostClassifier(random_state=seed)
xgb = XGBClassifier(random_state=seed)
knn = KNeighborsClassifier()

first_models = [logreg, lda, svc, l_svc,
                dtree, rf, gdb, adb, xgb, knn]
first_model_names = ['Logistic Regression', 'LDA', 'SVC', 'Linear SVC',
                    'Decision Tree', 'Random Forest', 'GradientBoosting',
                    'AdaBoost', 'XGB', 'K-Neighbors'] 

# Defining other steps
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, random_state=seed)
std_sca = StandardScaler()


# Splitting X and Y for our training and test sets

# In[ ]:


X_train = train.drop(['diagnosis', 'Target'] ,axis=1)
y_train = train['Target']


# ---

# ## 3.2 - Evaluating the first models

# ### Choosing the Proper Measure to Evaluate the Model Performance
# There are **a lot** of ways to measure the quality of your model and we must choose it carefully. This is one of the most important parts of a Machine Learning Project.
# 
# Our objective isn't classifying correctly the tumors. If that was the case simply using Accuracy - which is the ratio of correctly predicted classes - would do the job.
# 
# However, the objective of this analysis is **detecting malignant tumors**. And how do we measure that? Not with Accuracy, but with **RECALL**. 
# 
# Recall answers the following question: *from all the malignant tumors in our data, how many did we catch?*. Recall is calculated by dividing the True positives by the total number of positives (positive = malignant). It is important to realize that a high Recall doesn't mean a high Accuracy and there is often a trade-off between different performance measures. 
# 
# That said, we will be making our decisions based on Recall but we will also measure Accuracy to see the difference between them. Moving on!

# <blockquote>
#     <font color='darkblue'>
#         <b>Coding Explanation:</b><br>
#         The code on the cell below does the following steps:
#         <ol>
#             <li><b>Setting up:</b></li>
#             <ol>
#                 <li>Creates an array to store the out-of-fold predictions that we will use later on. Its shape is the training size by the number of models we have;</li>
#                 <li>Creates a list to store the Accuracy and Recall scores</li>
#             </ol>
#             <li><b>Outer Loop</b>: Iterating through Models</li>
#             <ol>
#                 <li>Creates a data pipeline with the scaler and the model</li>
#                 <li>Creates two arrays to store each fold's accuracy and recall</li>
#                 <li>Executes the inner loop</li>
#                 <li>By the end of the cross-validation, stores the mean and the standard deviation for those two measures in the scores list</li>
#             </ol>
#             <li><b>Inner Loop</b>: Cross-Validation</li>
#             <ol>
#                 <li>Splits the training data into train/validation data</li>
#                 <li>Fits the model with the CV training data and predicts the validation data</li>
#                 <li>Stores the out-of-fold predictions (which is the validation predictions) in oof_preds</li>
#                 <li>Measures the Accuracy and Recall for the fold and stores in an array</li>
#             </ol>
#         </ol>
#     </font>
#     </blockquote>

# In[ ]:


train_size = X_train.shape[0]
n_models = len(first_models)
oof_pred = np.zeros((train_size, n_models))
scores = []

for n, model in enumerate(first_models):
    model_pipeline = Pipeline(steps=[('Scaler', std_sca),
                                     ('Estimator', model)])
    accuracy = np.zeros(n_folds)
    recall = np.zeros(n_folds)
    
    for i, (train_ix, val_ix) in enumerate(skf.split(X_train, y_train)):
        x_tr,  y_tr  = X_train.iloc[train_ix], y_train.iloc[train_ix]
        x_val, y_val = X_train.iloc[val_ix],   y_train.iloc[val_ix]
        
        model_pipeline.fit(x_tr, y_tr)
        val_pred = model_pipeline.predict(x_val)
        
        oof_pred[val_ix, n] = model_pipeline.predict_proba(x_val)[:,1]
        
        fold_acc = accuracy_score(y_val, val_pred)
        fold_rec = recall_score(y_val, val_pred)
        
        accuracy[i] = fold_acc
        recall[i] = fold_rec
    
    scores.append({'Accuracy'          : accuracy.mean(),
                   'Recall'            : recall.mean()})


# <blockquote>
#     <font color='darkblue'>
#         <b>Why not scale the data before? Why the pipeline?</b><br>
#         This is a common and easy to avoid data-leakage mistake when using Scalers or other feature processing algorithms. In the case of standard scaling, it follows a simple process: it reads the data and calculates its mean and standard deviation then it centers the dataset mean to 0 and scales its standard deviation to 1<br>
#         <br>Let's say we weren't using cross-validation and instead we just had a train and a validation set. If we were to fit our StandardScaler with all the data, our scaled dataset would have information on the test set distribution - aka data leakage. We don't want our model to 'see' any data other than the training set so this is to be avoided.<br>
#         <br>This applies to cross-validation as well. When cross-validating, we create N train-validation splits and for every fold we must scale our data based on the training data only. So, how do we do that?<br>
#         <br>We use a <b>Pipeline</b> which <i>glues</i> our Model to other data preprocessing APIs. In our case, when er fit our pipeline we are fitting and transforming the data with StandardScaler and then fitting the ML Model. This could also be done separatedly, by having more code explicitly fitting and transforming the data for each fold. However, using Pipelines (especially when there are more processing steps) makes it cleaner and more reusable/scalable.<br>
#     </font>
#     </blockquote>
# 

# ### First Models' Results

# In[ ]:


measure_cols = ['Accuracy', 'Recall']#, 'Accuracy Std.Dev.', 'Recall Std.Dev.']

first_scores = pd.DataFrame(columns=measure_cols)

for name, score in zip(first_model_names, scores):
    
    new_row = pd.Series(data=score, name=name)
    first_scores = first_scores.append(new_row)
    
first_scores = first_scores.sort_values('Recall', ascending=False)
first_scores


# This table shows us each model ordered by its Recall, descending.
# 
# **Insights**:
# - SVC and Logistic Regression got the highest scores, while Decision Tree and LDA got the lowest.
# - LDA does not provide optimal results if the features are highly correlated - which some are. This poor result isn't a big surprise. 
# - All the other models got above 95% accuracy and 90% recall on a first try.

# <hr>

# ## 3.3 - Feature Selection

# Most models provide a method that returns feature importances or coefficients so we can have an idea of what is being considered the most important features of our dataset. SVC, Linear SVC and KNN are the ones that don't have it.
# 
# Let's see if we can find anything from the other models preferences. 

# In[ ]:


feature_names = X_train.columns
feat_imp_df = pd.DataFrame(columns=first_model_names, index=feature_names)

# Dropping the Models that don't have feature importances for this analysis
feat_imp_df.drop(['SVC', 'Linear SVC', 'K-Neighbors'], axis=1, inplace=True)

# I'm using absolute values for logistic Regression and LDA because we only care about the magnitude of the coefficient, not its direction 
feat_imp_df['Logistic Regression'] = np.abs(logreg.coef_.ravel())
feat_imp_df['LDA'] = np.abs(lda.coef_.ravel())
feat_imp_df['Decision Tree'] = dtree.feature_importances_
feat_imp_df['Random Forest'] = rf.feature_importances_
feat_imp_df['GradientBoosting'] = gdb.feature_importances_
feat_imp_df['AdaBoost'] = adb.feature_importances_
feat_imp_df['XGB'] = xgb.feature_importances_


# So this is how our table looks like right now. Each model has its own measure for each feature's importances. You will notice that some measures are in different scales. 
# 
# In order to compare the importances between the models we need to scale them. I will use sklearn MinMaxScaler to shrink them to a [0, 1] interval and then sum the features importances for each model.

# In[ ]:


feat_imp_df.head(3)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

scaled_fi = pd.DataFrame(data=mms.fit_transform(feat_imp_df),
                         columns=feat_imp_df.columns,
                         index=feat_imp_df.index)
scaled_fi['Overall'] = scaled_fi.sum(axis=1)


# In[ ]:


ordered_ranking = scaled_fi.sort_values('Overall', ascending=False)
fig, ax = plt.subplots(figsize=(10,7), dpi=80)
sns.barplot(data=ordered_ranking, y=ordered_ranking.index, x='Overall', palette='magma')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.set_visible(False)
ax.grid(False)
ax.set_title('Feature Importances for all Models');


# **Insights**:
# - Worst Perimeter is the most important features between models;
# - There is a clear preference for Worst features on models. The top 6 features are 'Worst';
# - All Fractal Dimension features are in the bottom 5. Symmetry's Mean and S.Error is pretty low as well, but Symmetry Worst, curiously, is pretty high. 
# 
# This is what our models have to tell us. If we decided on dropping features based on the correlations plotted in Chapter Three we would've gotten some of them wrong. 
# 
# Let's try now removing the Bottom 5 and repeat the training to see if we get any better results. Just copying the code already used before.

# In[ ]:


train_v2 = train.drop(ordered_ranking.index[:-6:-1], axis=1)
test_v2 = test.drop(ordered_ranking.index[:-6:-1], axis=1)

X_train_v2 = train_v2.drop(['diagnosis', 'Target'] ,axis=1)
X_test_v2 = test_v2.drop(['diagnosis', 'Target'] ,axis=1)


# In[ ]:


train_size = X_train_v2.shape[0]
test_size = test.shape[0]
n_models = len(first_models)
oof_pred = np.zeros((train_size, n_models))
scores = []

for n, model in enumerate(first_models):
    model_pipeline = Pipeline(steps=[('Scaler', std_sca),
                                     ('Estimator', model)])
    accuracy = np.zeros(n_folds)
    recall = np.zeros(n_folds)
    
    for i, (train_ix, val_ix) in enumerate(skf.split(X_train_v2, y_train)):
        x_tr,  y_tr  = X_train_v2.iloc[train_ix], y_train.iloc[train_ix]
        x_val, y_val = X_train_v2.iloc[val_ix],   y_train.iloc[val_ix]
        
        model_pipeline.fit(x_tr, y_tr)
        val_pred = model_pipeline.predict(x_val)
        
        oof_pred[val_ix, n] = model_pipeline.predict_proba(x_val)[:,1]
        
        fold_acc = accuracy_score(y_val, val_pred)
        fold_rec = recall_score(y_val, val_pred)
        
        accuracy[i] = fold_acc
        recall[i] = fold_rec
    
    scores.append({'Accuracy'          : accuracy.mean(),
                   'Recall'            : recall.mean()})


# In[ ]:


measure_cols = ['Accuracy', 'Recall']

fs_scores = pd.DataFrame(columns=measure_cols)

for name, score in zip(first_model_names, scores):
    
    new_row = pd.Series(data=score, name=name)
    fs_scores = fs_scores.append(new_row)
    
fs_scores = fs_scores.sort_values('Recall', ascending=False)


# In[ ]:


d={'First Scores':first_scores, 'Less Features':fs_scores}
pd.concat(d, axis=1, sort=False)


# **Insights from Feature Selection**:
# - What changed?
#     - Logistic Regression and LDA didn't change at all;
#     - SVC and Gradient Boosting slightly improved (probably just one extra sample);
#     - KNN, Decision Tree, Linear SVC and XGB got worst;
#     - AdaBoost and Random Forest greatly improved
# - Our bottom models are the same as before (LDA, Decision Tree and Linear SVC). 
# 
# 
# <b>It is not clear if removing the features was a good decision or not. When in doubt, opt for the simpler choice: We are removing them.</b>
# 
# We will start our model selection by dropping Decision Tree and LDA. We will also drop Linear SVC because we can tweak the regular SVC parameters to obtain a Linear SVC.
# 
# <blockquote>
#     <font color='darkblue'>
#         <b>Linear SVCs</b> train and predict really faster than a SVC with linear parameters, but once more, our dataset is small so it's not a problem
#     </font>
# </blockquote>

# <hr>

# ## 3.4 - Analysing Model Errors

# We will start selecting the next models by creating a dataframe with all the models' out-of-fold predictions to compare their results.

# In[ ]:


oof_dataframe = pd.DataFrame(data=oof_pred, columns=first_model_names, index=train.index)
oof_dataframe['Target'] = train['Target']
oof_dataframe = oof_dataframe.drop(['LDA', 'Decision Tree', 'Linear SVC'], axis=1)


# ### Can't get them right
# Lets see if we can find examples that all models got the classification wrong. The function defined below does just that.

# In[ ]:


def all_wrong(x):
    predictions = sum(x[:7])
    target = x[7]
    if (target == 1 and predictions == 0) or        (target == 0 and predictions == 7):
        return True
    
    else: return False


# In[ ]:


oof_dataframe['All_wrong'] = round(oof_dataframe).apply(all_wrong, axis=1)
oof_dataframe.query("All_wrong == True")


# We have those five tumors that no model got right. By the looks of it, AdaBoost was the one that was closest to classifying it right. (The standard threshold is 0.5 probability). I'm out of ideas to further explore these for now.

# ### Getting Different Opinions
# Simply plotting correlations will be hard to distinguish which models are least correlated with the rest. This is due to the fact that all remaining models have over 95% accuracy so their overall correlation will be high.
# 
# A better way to approach this is by looking at the tumors that our models classified wrong and/or that they didn't agree on the classification. We can map the models' predictions for 'Easy' ones (that most of them got right) and filter them out. This way we can focus only on how different their 'opinions' are.

# <blockquote>
#     <h3><font color='darkblue'>On Model Stacking</font></h3>
#         <p><font color='darkblue'>
#             <b>'Why pick models that don't agree with each other?'</b><br>
#                 We are looking for uncorrelated models for model stacking and this question is a really common one. Why is it better?<br>
#                 <a href='http://blog.kaggle.com/author/bengorman/'>Ben Gorman</a> has a nice post on this topic explaining it and I suggest you read it if you want to get the intuition on stacking. (<a href='http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/'>blog post</a>)
#         </font></p>
#         <p><font color='darkblue'>
#             Nonetheless I will try to put it <u>in simple words</u> here:<br>
#             Model Stacking is just like building any kind of team. You don't want everyone in your team good in the same things, you want diversity so you can perform well on different cases/scenarios. If you want to build a diagnosis medical team for all kinds of scenarios you probably don't want only infectologists. Putting it this way, it is intuitively wiser to get different kinds of specializations on your team. 
#         </font></p>
#         <p><font color='darkblue'>
#             <u>It works exactly the same in Machine Learning.</u>
#         </font></p>
#         <p><font color='darkblue'>
#             <b>Imagine we only had three models and had to choose two: A Linear SVC, a Logistic Regression and a KNN</b>. Both Linear SVC and Logistic Regression are linear models so they have similar results. Let's also say that:
#         </font></p>
#             <ul><font color='darkblue'>
#                 <li>We only have two features: Radius and Perimeter</li>
#                 <li><b>Linear SVC and Logistic Regression</b> perform best on cases where all features are high (high radius and perimeter)</li>
#                 <li><b>KNN</b> performs best on the opposite scenario - low radius and perimeter</li>
#             </font></ul>
#         <p><font color='darkblue'>
#             In other words, LinearSVC and LogReg are highly correlated with eachother but not with KNN. The obvious choice is to pick KNN and one of the other ones, otherwise we will never get all cases right. Let's pick LSVC for this example.
#         </font></p>
#         <p><font color='darkblue'>
#             The fantastic thing about Model Stacking is that our second level model is able to learn when to use each models' opinion for every data point. If we - ontop of Radius and Perimeter features - added the predictions from KNN and and LSVC as features, our second-level model is able to, for instance, associate high radius and perimeter cases with LSVC predictions and don't listen to KNN on such cases.
#         </font></p>

# In[ ]:


# We have 7 models + our target so the perfect scores would be 0 and 1
# I am also adding to the Easy Ones group cases that only one model disagrees with the rest
oof_dataframe['Easy_one'] = round(oof_dataframe).sum(axis=1).isin([0, 1, 7, 8])


# In[ ]:


# We define our Hard_ones dataset by filtering easy_ones out
hard_ones = oof_dataframe.query("Easy_one == False and     All_wrong == False").drop(['Easy_one', 'All_wrong'], axis=1)

plt.figure(figsize=(10,8), dpi=80)
sns.heatmap(hard_ones.corr(),
            vmin=-0.4, vmax=0.7, annot=True)
plt.title("Correlation between models for the 'Hard Ones'");


# This heatmap shows correlation between prediction probabilities for all models on the hard tumor samples defined above. We are looking at the 5% our models can't get right.
# 
# **Insights:**
# - The most correlated models are XGB and GradientBoosting, which is to be expected. Between those two, however, GradientBoosting is less correlated with the other models (-0,44 with Logistic Regression and -0,52 with SVC). **We should keep GradientBoostin and drop XGB**;
# - **SVC and Logistic Regression** are the ones with the highest correlation with the Target on these hard data points and they are fundamentally different. **Keeping them**.
# - **KNN** also has a high correlation with the target and an overall low correlation with the other models - **stays**.
# - Finally we will also **keep Random Forest** to have another one besides Gradient Boosting to disagree with Logistic Regression and SVC.
# 
# This leaves us with five models out of our ten initial ones:
# - Logistic Regression
# - SVC
# - GradientBoosting
# - Random Forest
# - KNN

# ### Chapter Recap:
# 
# In this chapter we:
# - Started with a 10 model list
# - Trained them using cross-validation and measured Accuracy and Recall
# - Analysed each models' feature importances and dropped five features out of our dataset
# - Analysed the models' predictions for the classifications they disagreed on
# - Chose 5 models for the next analysis

# <hr>

# <hr>

# # 4 - Fine Tuning the System

# ## 4.1 - Hyperparameters
# Listed below are the five models and the parameters we are going to tune (not all will be listed).
# 
# ![](https://i.imgur.com/PMfff5N.png)

# <hr>

# ## 4.2 - Tuning Tools

# Sklearn's GridSearchCV is our best friend for parameter tuning. We will optimize our models for Recall. Lets start importing it. 

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Defining this function to make our lives easier on tuning
def train_gridsearch(model, x=X_train_v2, y=y_train, name=None):
    t_model = model
    t_model.fit(x, y)
    print(30*'-')
    if name != None: print(name)
    print('\nBest Parameters:')
    for item in t_model.best_params_.items():
        print(item[0], ': ', item[1])
    print('\nScore: ', t_model.best_score_, '\n')
    print(30*'-')


# <blockquote>
#     <font color='darkblue'>
#         A common way to start searching for parameters is to pick multiples of 10 and then refine it as you go. <br>
#         Another way to do this is to use <b>RandomizedSearchCV</b> and create a large parameter space. This API will randomly pick parameters to test (this technique is more recommended if you have too many parameters to try and don't have the time)
#     </font>
#     </blockquote>

# One last thing before starting our tuning is to create another Pipeline step for the log transformation we applied in Chapter 3.1. This transformation might be helpful and we want to try it with GridSearchCV
# 
# We can create a Logger class using *BaseEstimator* and *TransformerMixin* so we can put it inside a pipeline. Inside this Logger class we can define our log transformation function and set a parameter to trigger it.

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class Logger(BaseEstimator, TransformerMixin):
    def __init__(self, apply_log = True):
        self.apply_log = apply_log
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        logX = X.copy()
        
        if self.apply_log:
            logX = np.log1p(X)
            return logX
    
        else: return X

logger = Logger()


# <hr>

# ## 4.3 - Logistic Regression

# The first model we are tuning is Logistic Regression. Let's start listing the parameters in a dictionary so we can feed our GridSearchCV      

# In[ ]:


# Logistic Regression Initial Parameters
log_pams = [{'M__solver':['liblinear'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.001, 0.01, 0.1, 1, 10, ],
             'M__penalty':['l1'], 
             'L__apply_log':[True, False]},
            {'M__solver':['lbfgs'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.001, 0.01, 0.1, 1, 10, ],
             'M__penalty':['l2'], 
             'L__apply_log':[True, False]}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
log_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', logreg)])

log_gs = GridSearchCV(log_pipe, log_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False)

train_gridsearch(log_gs)


# Our best C is at 1 so we might refine our parameters near that value. A second run on parameter tuning could look like:

# In[ ]:


# Logistic Regression Initial Parameters
log_pams = [{'M__solver':['liblinear'],
             'M__class_weight':['balanced'],
             'M__C': [0.5, 0.75, 1, 1.25, 1.5],
             'M__penalty':['l1'], 
             'L__apply_log':[True]}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
log_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', logreg)])

log_gs = GridSearchCV(log_pipe, log_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False)

train_gridsearch(log_gs)


# Let's settle for that. Already a great improvement from the first model results.
# 
# At least for Logistic Regression, our log transformation worked well.

# In[ ]:


logreg_tuned = log_gs.best_estimator_


# <hr />

# ## 4.4 - SVC

# In[ ]:


# SVC Initial Parameters
svc_pams = [{'M__kernel':['rbf'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
             'M__gamma':['auto', 'scale', 0.001, 0.01, 0.1],
             'L__apply_log':[True, False]}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
svc_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', svc)])

svc_gs = GridSearchCV(svc_pipe, svc_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False, refit=True)

train_gridsearch(svc_gs)


# We got an amazing result for the first round of tuning for SVC.

# In[ ]:


# SVC Second round Parameters
svc_pams = [{'M__kernel':['rbf'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.05, 0.07, 0.1, 0.12, 0.15, 0.2],
             'M__gamma':[0.05, 0.1, 0.15, 0.5, 1.0],
             'L__apply_log':[True, False]}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
svc_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', svc)])

svc_gs = GridSearchCV(svc_pipe, svc_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False)

train_gridsearch(svc_gs)


# **100%!** In our training data, through cross-validation, our model was able to predict all the malignant tumors. 
# 
# **This, however, comes at a cost!** SVC is probably overfitting and/or classifying a lot of benign tumors as malign to get 100% recall. As discussed in the beginning of chapter 4, there is often a trade-off between performance measures.
# 
# Let's take a quick detour and see how is this tuned SVC classifying our training data. A confusion matrix will expose this impostor!

# In[ ]:


print(confusion_matrix(y_train, svc_gs.predict(X_train_v2)))


# As expected, SVC is 'lowering the bar' to classify malignant tumors and in that process it is wrongly classying many (128) benign tumors as malignant. Let's try tuning it again and using F1-Score instead (F1 is an average of Recall and Precision).

# In[ ]:


# SVC Initial Parameters
svc_pams = [{'M__kernel':['rbf'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
             'M__gamma':['auto', 'scale', 0.001, 0.01, 0.1],
             'L__apply_log':[True, False]}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
svc_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', svc)])

svc_gs = GridSearchCV(svc_pipe, svc_pams, 'f1',
                      cv=skf, n_jobs=-1, iid=False,
                      refit=True)

train_gridsearch(svc_gs)


# In[ ]:


# SVC Second Parameters
svc_pams = [{'M__kernel':['rbf'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [5, 7.5, 10, 12.5, 15],
             'M__gamma':[0.005, 0.01, 0.015],
             'L__apply_log':[True, False]}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
svc_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', svc)])

svc_gs = GridSearchCV(svc_pipe, svc_pams, 'f1',
                      cv=skf, n_jobs=-1, iid=False,
                      refit=True)

train_gridsearch(svc_gs)


# If we check the Confusion Matrix again, SVC is way more balanced now. We can also measure the Recall using cross-validation.

# In[ ]:


print(30*'-')
print('Confusion Matrix:')
print(confusion_matrix(y_train, svc_gs.predict(X_train_v2)))
print('\nCV Recall Score:')
print(cross_validate(svc_gs, X_train_v2, y_train,
                     scoring='recall', cv=skf)['test_score'].mean())
print(30*'-')


# Ok, this is way better than optimizing GridSearch with Recall. We weren't able to improve the recall from the first score in Chapter 4 and the log transformation didn't help here. Moving on.  

# In[ ]:


svc_tuned = svc_gs.best_estimator_


# <hr>

# ## 4.5 - GradientBoosting

# <blockquote>
#     <font color='darkblue'>
#         <b>Training GDB takes too long</b>. I took the starting parameters I used out of the code and put it here so I don't have to wait this long everytime.<br>
#         <br>'max_depth':[3, 4, 6, 8],
#         <br>'min_samples_leaf':[1, 2],
#         <br>'max_features': [None, 0.6, 0.75, 0.9],
#         <br>'learning_rate':[0.001, 0.01, 0.1, 1.0],
#         <br>'n_estimators':[30, 60, 100, 200],
#         <br>'subsample':[0.1, 0.5, 0.8, 1.0],
#         <br>'apply_log':[False, True]

# In[ ]:


# GradientBoosting Second round Parameters
gdb_pams = {'M__max_depth':[3],
            'M__min_samples_leaf':[2],
            'M__max_features': [0.9, 0.95],
            'M__learning_rate':[0.05, 0.1, 0.15],
            'M__n_estimators':[60, 80],
            'M__subsample':[0.8, 0.9, 1.0],
            'L__apply_log':[False]}

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
gdb_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', gdb)])

gdb_gs = GridSearchCV(gdb_pipe, gdb_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False, refit=True)

train_gridsearch(gdb_gs)


# We weren't able to improve the score on the second round of tuning. Moving on.

# In[ ]:


gdb_tuned = gdb_gs.best_estimator_


# <hr>

# ## 4.6 - Random Forest

# <blockquote>
#     <font color='darkblue'>
#         <b>Initial Parameters:</b>
#         <br>'max_depth':[None, 4, 8, 16],
#         <br>'min_samples_leaf':[1, 2],
#         <br>'max_features': [None, 0.6, 0.75, 0.9, 'auto'],
#         <br>'bootstrap':[True, False],
#         <br>'n_estimators':[10, 30, 60, 100, 200],
#         <br>'class_weight':[None, 'balanced'],
#         <br>'apply_log':[False, True]

# In[ ]:


# Random Forest Second round Parameters
rf_pams = {'M__max_depth':[None],
           'M__min_samples_leaf':[1, 2],
           'M__max_features': [0.8, 0.9, 0.95],
           'M__n_estimators':[8, 10, 12],
           'M__class_weight':['balanced'],
           'L__apply_log':[False, True]}

# It is important to apply the log transformer before the scaling
rf_pipe = Pipeline(steps=[('L', logger),
                          ('S', std_sca),
                          ('M', rf)])

rf_gs = GridSearchCV(rf_pipe, rf_pams, scoring='recall',
                     cv=skf, n_jobs=-1, iid=False)

train_gridsearch(rf_gs)


# Quite few trees for our random forest. Moving on.

# In[ ]:


rf_tuned = rf_gs.best_estimator_


# <hr>

# ## 4.7 - K-Nearest Neighbors

# There is not much to tune in KNN so we will just go for a single round of tuning.

# In[ ]:


knn_pams = {'M__n_neighbors':np.arange(2, 16),
            'M__weights':['uniform', 'distance'],
            'M__p':[1, 2, 3],
            'L__apply_log':[False, True]}

# It is important to apply the log transformer before the scaling
knn_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', knn)])

knn_gs = GridSearchCV(knn_pipe, knn_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False)

train_gridsearch(knn_gs)


# Quite an improvement from our first attempt in Chapter 4 (0,9117). 

# In[ ]:


knn_tuned = knn_gs.best_estimator_


# <hr>

# <hr>

# # 5 - Model Stacking

# So we have our five tuned models ready to be tested. Before we go into the final part of this study I will prepare a stacked model like mentioned before.
# 
# We will be using **Logistic Regression** as a second-level model because it was the one which had the best scores so far. First we need the tuned models predictions. 
# <hr>
# 
# ## 5.1 - Defining the Stacking Data
# 
# We can use sklearn's *cross_val_predict* to make our lives easier.

# In[ ]:


from sklearn.model_selection import cross_val_predict

tuned_models = [logreg_tuned, svc_tuned,
                gdb_tuned, rf_tuned, knn_tuned]
tuned_names = ['Logistic Regression', 'SVC', 'GradientBoosting', 'RandomForest', 'KNNeighbors']


# In[ ]:


tuned_oof_pred = np.zeros(shape=(train_size, 5)) # 5 models

for i, model in enumerate(tuned_models):
    tuned_oof_pred[:,i] = cross_val_predict(model, X_train_v2,
                                            method='predict_proba',
                                            y=y_train,
                                            cv=skf)[:,1]

tuned_train_pred = pd.DataFrame(data=tuned_oof_pred, 
                            index=X_train_v2.index, 
                            columns=tuned_names)


# Doing the same for the test data. Here we don't need to use cross-validation: We simply fit on the training data and predict the test data.

# In[ ]:


tuned_test_pred = np.zeros(shape=(test_size, 5)) # 5 models

for i, model in enumerate(tuned_models):
    
    model.fit(X_train_v2, y_train)
    tuned_test_pred[:,i] = model.predict_proba(X_test_v2)[:,1]

tuned_test_pred = pd.DataFrame(data=tuned_test_pred, 
                            index=X_test_v2.index, 
                            columns=tuned_names)


# We will also need to scale the training and test data and then concatenate with our first-level predictions.

# In[ ]:


X_train_scaled = std_sca.fit_transform(X_train_v2)
X_test_scaled = std_sca.transform(X_test_v2)

X_train_final = np.concatenate([X_train_scaled, tuned_train_pred], axis=1)
X_test_final = np.concatenate([X_test_scaled, tuned_test_pred], axis=1)


# <hr>

# ## 5.2 - Tuning Second Level Model

# Our data is ready. Only thing missing now is to tune our LogReg. We will do two quick rounds of tuning.

# In[ ]:


# Logistic Regression Initial Parameters
log_pams = [{'solver':['liblinear'],
             'class_weight':[None, 'balanced'],
             'C': [0.001, 0.01, 0.1, 1, 10],
             'penalty':['l1']},
            {'solver':['lbfgs'],
             'class_weight':[None, 'balanced'],
             'C': [0.001, 0.01, 0.1, 1, 10],
             'penalty':['l2']}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
log_lvl2 = LogisticRegression(random_state=seed)

log_lvl2_gs = GridSearchCV(log_lvl2, log_pams, scoring='recall',
                           cv=skf, n_jobs=-1, iid=False)

train_gridsearch(log_lvl2_gs, x=X_train_final, y=y_train)

log_snd_lvl = log_lvl2_gs.best_estimator_


# In[ ]:


# Logistic Regression Second Round Parameters
log_pams = {'solver':['liblinear'],
            'class_weight':[None],
            'C': [0.005, 0.01, 0.015, 0.02],
            'penalty':['l1']}

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
log_lvl2 = LogisticRegression(random_state=seed)

log_lvl2_gs = GridSearchCV(log_lvl2, log_pams, scoring='recall',
                           cv=skf, n_jobs=-1, iid=False)

train_gridsearch(log_lvl2_gs, x=X_train_final, y=y_train)

log_snd_lvl = log_lvl2_gs.best_estimator_


# That's the best we can get from it. 
# 
# **Everything ready - let's dive into the test set!**

# <hr>

# <hr>

# # 6 - Test Evaluation
# We will be evaluating all the first level models individually and the second level Logistic Regression as well.
# 
# ---
# 
# ## 6.1 - First Level Models
# We already have our test probability predictions defined for our second level model - we just need to round it to get the predictions.

# In[ ]:


test_predictions = round(tuned_test_pred)
y_test = test['Target']


# The function defined below plots the results in a prettier way

# In[ ]:


def confusion_plot(y_true, pred, ax, name):
    ax.xaxis.set_ticks_position('top')
    sns.heatmap(confusion_matrix(y_test, pred),
                ax=ax, annot=True, square=True, cbar=False,
                fmt='.0f', cmap='BuGn_r', vmax=10)
    ax.set_title(f'{name}\n\nPredicted')
    ax.set_xlabel(f'Accuracy:  {100*accuracy_score(y_test, pred):.4}%     \nRecall: {100*recall_score(y_test, pred):.4}%')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.label.set_fontsize(11)

    for tik in ax.get_xticklines():
        tik.set_visible(False)
    for tk in ax.get_yticklabels():
        tk.set_visible(False)


# In[ ]:


fig, axes = plt.subplots(1,5, figsize=(12,5), dpi=80)
fig.subplots_adjust(wspace=0.3)

for name, col, ax in zip(tuned_names, tuned_test_pred.columns, axes):
    pred = np.round(tuned_test_pred[col])
    confusion_plot(y_test, pred, ax, name)

for tick in axes[0].get_yticklabels():
    tick.set_visible(True)

axes[0].set_ylabel('True');


# **Insights from the results:**
# - Even though we optimized most models for Recall, recall is still always lower than accuracy. 
# - All our models got at least 90% recall - for this data this means 6 malignant tumors not detected
# - Our good old Logistic Regression performed the best, **with 95,3% malgiinant tumors detected**. 
# 
# We have each model's probabilities. One thing we can do is to **lower the probability threshhold** (50% is the standard). This will make us potentially lose in Accuracy but get a higher Recall. The code below lowers it to 25% (arbitrarly chosen).

# In[ ]:


fig, axes = plt.subplots(1,5, figsize=(13,6), dpi=80)
fig.subplots_adjust(wspace=0.3)

for name, col, ax in zip(tuned_names, tuned_test_pred.columns, axes):
    pred = tuned_test_pred[col].apply(lambda x: 1 if x>=0.25 else 0)
    confusion_plot(y_test, pred, ax, name)

for tick in axes[0].get_yticklabels():
    tick.set_visible(True)

axes[0].set_ylabel('True');


# All models improved with the lower threshold, now the lowest being 93.75% Recall for KNN.
# 
# SVC has an incredible **98.4% Recall and 98.2% Accuracy**! Logistic Regression is right behind with the same Recall and a lower accuracy. Those two were our best models since the beginning (Chapter 4).

# <hr>

# ## 6.2 - Second-Level Model

# After an amazing score on SVC, let's see if our Second-Level Logistic Regression can beat it.

# In[ ]:


log_snd_lvl.fit(X_train_final, y_train)
second_lvl_pred = pd.Series(log_snd_lvl.predict_proba(X_test_final)[:,1])


# In[ ]:


fig, ax = plt.subplots(figsize=(5,3), dpi=80)
confusion_plot(y_test, np.round(second_lvl_pred), ax, 'Second-Level Logistic Regression')
for tick in ax.get_yticklabels():
    tick.set_visible(True)
ax.set_ylabel('True');


# Right on target! I did not expect it to perform so well. Don't even need to change the threshold - we got a **100% Recall on test set!!**
# 
# That's it for our Test Evaluations.

# ---

# ---

# # 7. Conclusion

# In **Chapter One** we started introducing this study and reading the data
# 
# In **Chapter Two**, exploring our data, we've found some interesting information:
# - We have a few right-skewed features and some of it is explained by the above 20% error
# - Applying a log transformation reduced overwall skewness
# - There are quite a few correlated features, which is expected (e.g. radius and area)
# - The plots hinted us of fractal dimension being a good candidate for removal - which was confirmed on chapter three
# - We've also studied correlations (Chapter 2.3) and the 'Worst' features appeared to be the most important.
# 
# In **Chapter Three** we've worked on a first list of models using their standard parameters. We've evaluated them on Recall and Accuracy by using cross-validation. Based on the first results we did a single round of feature selection by analysing the feature importances and then we analysed the models' errors to help us decide on which ones to keep and which ones to drop. By the end of Chapter Three we had five remaining models: Logistic Regression, SVC, Random Forest, Gradient Boosting and KNN.
# 
# **Chapter Four** was all about tuning the models selected on the previous chapter to optimize the hyperparameters.
# 
# **Chapter Five** sets up the data for a model stacking by predicting the probabilities for each tuned model for the training and test sets and then tunes the Second-Level model - a Logistic Regression - using the new training data.
# 
# Finally, **Chapter Six** evaluates all models on the untouched test-set. For the first-level models, SVC and Logistic Regression performed best (which was already hinted on chapter three without any tuning). Lowering the threshold got a 98,5% Recall for them. With the second-level Logistic Regression we were able to reach 100% Recall!   
# 
# Please leave your thoughts or questions in the comments. Any feedback is welcome.
# 
# If you've enjoyed it, let me know by UPVOTING! This way I will get motivated to make more Kernels for you.

# <hr>

# <hr>

# # <font color='darkgreen'>Annex - A: Unbalanced Data with SMOTE </font>
# As you might recall, our dataset is unbalanced: we have more benign tumors. 
# 
# Having unbalanced datasets might be more difficult for your model to learn how to classifiy properly. Ideally, we would need our data with classes split evenly. There are different ways to approach this problem and I suggest you reading this <a href='https://medium.com/james-blogs/handling-imbalanced-data-in-classification-problems-7de598c1059f'>blog post</a> by Hoang Minh for an introduction to the problem and other links for further reading.
# 
# The most intuitive way to deal with this is to drop some of the data from the class with most ocurrences until we have a 50-50 balance. However, we have a pretty small dataset already so that is not a good option here. If we can't lower the number of benigns, we need to increase the number of malignants and gathering more real data is not feasible - so we create synthetic data. 
# 
# **SMOTE** (Synthetic Minority Over-Sampling Technique) is one of the methods to do that. Roughly SMOTE looks for the location of the minority class in our feature space and creates synthetic data *between* the real data. **Put in simple words**: Say we have a 1D (only one feature) data. SMOTE finds out that there are two malignant tumors with values of 10 and 12 (e.g. radius) so it creates another malignant data point between those two, with a value of 11.
# 
# Enough explaining. This is how our unbalanced dataset is at the moment:

# In[ ]:


pd.concat([train_diag, test_diag], axis=1)


# <hr>

# ## <font color='darkgreen'>A.1 - Balancing the Dataset
# We can with a few lines of code import SMOTE and generate the new data points:

# In[ ]:


from imblearn.over_sampling import SMOTE

# I'm using 6 neighbors (the default is 5) because our KNN model optimized for this number in chapter four
sm = SMOTE(k_neighbors=6, random_state=seed)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

X_train_bal = pd.DataFrame(data=X_train_res, columns=X_train.columns)
X_train_bal = X_train_bal.drop(ordered_ranking.index[:-6:-1], axis=1)

y_train_bal = pd.Series(y_train_res, name='Target')
y_train_bal.value_counts()


# <hr>

# ## <font color='darkgreen'>A.2 - Comparing with First Scores
# Let's compare our first results for all 10 models with the same process used in Chapter 3 but using the balanced dataset.

# In[ ]:


train_size = X_train_bal.shape[0]
n_models = len(first_models)
oof_pred = np.zeros((train_size, n_models))
scores = []

for n, model in enumerate(first_models):
    model_pipeline = Pipeline(steps=[('Scaler', std_sca),
                                     ('Estimator', model)])
    accuracy = np.zeros(n_folds)
    recall = np.zeros(n_folds)
    
    for i, (train_ix, val_ix) in enumerate(skf.split(X_train_bal, y_train_bal)):
        x_tr,  y_tr  = X_train_bal.iloc[train_ix], y_train_bal.iloc[train_ix]
        x_val, y_val = X_train_bal.iloc[val_ix],   y_train_bal.iloc[val_ix]
        
        model_pipeline.fit(x_tr, y_tr)
        val_pred = model_pipeline.predict(x_val)
        
        oof_pred[val_ix, n] = model_pipeline.predict_proba(x_val)[:,1]
        
        fold_acc = accuracy_score(y_val, val_pred)
        fold_rec = recall_score(y_val, val_pred)
        
        accuracy[i] = fold_acc
        recall[i] = fold_rec
    
    scores.append({'Accuracy'          : accuracy.mean(),
                   'Recall'            : recall.mean()})


# In[ ]:


measure_cols = ['Accuracy', 'Recall']#, 'Accuracy Std.Dev.', 'Recall Std.Dev.']

balanced_scores = pd.DataFrame(columns=measure_cols)

for name, score in zip(first_model_names, scores):
    
    new_row = pd.Series(data=score, name=name)
    balanced_scores = balanced_scores.append(new_row)
    
balanced_scores = balanced_scores.sort_values('Recall', ascending=False)

d={'First Scores':first_scores, 'Rebalanced Classes':balanced_scores}
pd.concat(d, axis=1, sort=False)


# We can see and overall improvement in Recall for all models. **This might be misleading because the synthetic samples are 'easy ones'. Our test scores will say if we improved or not.**
# 
# For speed, we are going to continue with just two first-level models for tuning and test evaluation: Logistic Regression and KNN. 

# <hr>

# ## <font color='darkgreen'>A.3 - Model Tuning
# Doing a single round of tuning with the same starting parameters used in chapter four. 

# In[ ]:


# Logistic Regression Initial Parameters
log_pams = [{'M__solver':['liblinear'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
             'M__penalty':['l1', 'l2'], 
             'L__apply_log':[True, False]},
            {'M__solver':['lbfgs'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
             'M__penalty':['l2'], 
             'L__apply_log':[True, False]}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
log_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', logreg)])

log_gs = GridSearchCV(log_pipe, log_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False)

train_gridsearch(log_gs, x=X_train_bal, y=y_train_bal)

log_balanced = log_gs.best_estimator_


# In[ ]:


knn_pams = {'M__n_neighbors':np.arange(2, 10),
            'M__weights':['uniform', 'distance'],
            'M__p':[1, 2, 3],
            'L__apply_log':[False, True]}

# It is important to apply the log transformer before the scaling
knn_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', knn)])

knn_gs = GridSearchCV(knn_pipe, knn_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False)

train_gridsearch(knn_gs, x=X_train_bal, y=y_train_bal)

knn_balanced = knn_gs.best_estimator_


# <hr>

# ## <font color='darkgreen'>A.4 - Test Scores

# In[ ]:


bal_names = ['Logistic Regression', 'K-Nearest Neighbors']
bal_models = [log_balanced, knn_balanced]


# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(7,4), dpi=80)
fig.subplots_adjust(wspace=0.3)

for name, mod, ax in zip(bal_names, bal_models, axes):
    pred = mod.predict(X_test_v2)
    confusion_plot(y_test, pred, ax, name)

for tick in axes[0].get_yticklabels():
    tick.set_visible(True)

axes[0].set_ylabel('True');


# Not great. Let's try reducing the threshold like we did in Chapter 6.

# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(7,4), dpi=80)
fig.subplots_adjust(wspace=0.3)

for name, mod, ax in zip(bal_names, bal_models, axes):
    pred = pd.Series(mod.predict_proba(X_test_v2)[:,1]).apply(lambda x: 1 if x>=0.25 else 0)
    confusion_plot(y_test, pred, ax, name)

for tick in axes[0].get_yticklabels():
    tick.set_visible(True)

axes[0].set_ylabel('True');


# **Conclusions**:
# - We were able to succesfully rebalance our dataset using SMOTE
# - However, the improvement on the training scores didn't manifest in the test scores. The models used in chapter 6 had better scores.
# - Many models already have a tool to deal with unbalanced datasets - such as the 'class_weight' parameter. In this dataset, creating synthetic samples wasn't useful.

# <hr>

# <hr>

# # <font color='darkgreen'> Annex - B: Dimensionality Reduction Techniques </font>
# On this annex we will try out three different methods for dimensionality reduction:
# - Principal Component Analysis
# - Autoencoder
# - Linear Discriminant Analysis
# 
# We will be using a Logistic Regression to evaluate each method. Whoever performs best will be used for model tuning and will be evaluated again on the test set.

# <hr>

# ## <font color='darkgreen'>B.1 - Principal Component Analysis
# Principal component analysis (aka PCA) is a very popular unsupervised learning model used for dimensionality reduction.
# 
# A PCA model reads the data and looks in the feature space for the directions that the data has the maximum variances. Those directions are the principal components. The first - highest variance - is the first principal component, and so on. 
# 
# Scikit-learn's PCA can have as inputs the number of components you want to build or you can specify how much of the variance in the data you want it to represent. For instance, if you want to reduce dimensionality in a way that your new features explain 50% of the variance in the data, you can use *n_components=0.5* and the model will calculate how many features you need for that.
# 
# For our case we are going for 95% explained variance.

# In[ ]:


from sklearn.decomposition import PCA

# 95% of variance Explained
pca = PCA(n_components=0.95, random_state=42)

# New train and test sets
pca_feats = pca.fit_transform(X_train_scaled)
pca_test_feats = pca.transform(X_test_scaled)


# In[ ]:


pca_log = LogisticRegression(penalty='none', solver='lbfgs', random_state=42)
pred = pca_log.fit(pca_feats, y_train).predict(pca_test_feats)

pca_train = cross_validate(pca_log, pca_feats, y_train, cv=10,
                          scoring=['accuracy', 'recall'])['test_recall']

pca_train = pca_train.mean()

print('PCA scores:')
print(f'Train Score: {pca_train:.4f}')
print(f'Test Score:  {recall_score(y_test, pred):.4f}')


# <hr>

# ## <font color='darkgreen'>B.2 - AutoEncoder Neural Network
# Autoencoder is an unique Neural Network that learns how to represent the data with less features. Put in simple words, the autoencoder:
# 1. Reads the X_train data (without the target)
# 2. Funnels the data through a hidden neuron layer with less neurons than features
# 3. Expands the data again to the same number of neurons as the first input layer (i.e. n_neurons = n_features)
# 
# We train the NN to replicate the input data and constrain it by having fewer neurons in the middle layers. That way, the network has to 'choose' which information to 'hold' in order to best represent the essence of the data. The image illustrates an autoencoder:
# ![](https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-06-at-3.17.13-PM.png)
# 
# <blockquote><font color='darkblue'>By tweaking a few parameters, an AutoEncoder can be used as a PCA model!</font></blockquote>
# 
# After the network is trained there are two things you can do with it:
# - Use the outputs of the last layer as de-noised data
# - Use the outputs of the middle layer as dimensionality reduction (what we're interested in)
# 
# We will use a Neural network with 5 layers and a 25-15-5-15-25 neurons distribution. Using Keras is the easiest way to build it. 
#     
#    

# In[ ]:


from keras.layers import Input, Dense
from keras.models import Model
from keras.backend import clear_session
from keras.regularizers import l1
from keras.callbacks import EarlyStopping


# <blockquote><font color='darkblue'>I won't comment much on the code here but if you have any question please ask in the comments!

# In[ ]:


# Best practices
clear_session()

n_cols = X_train_scaled.shape[1]

# Defining layers
input_layer = Input(shape=(n_cols,), name='Input')
encode_1 = Dense(15, activation='relu', name='Encoder',
                 kernel_regularizer=l1(0.001))(input_layer)
bottleneck = Dense(5, activation='relu', name='Middle_layer',
                 kernel_regularizer=l1(0.001))(encode_1)
decode_1 = Dense(15, activation='relu', name='Decoder',
                 kernel_regularizer=l1(0.001))(bottleneck)
output = Dense(n_cols, activation=None, name='Output',
                 kernel_regularizer=l1(0.001))(decode_1)

# Defining the network
autoencoder = Model(input_layer, output)
autoencoder.compile(optimizer='adam', loss='mse')

# Will be used for dimensionality reduction
encoder = Model(input_layer, bottleneck)

# Early stopping callback
early = EarlyStopping(restore_best_weights=True, patience=15)


# In[ ]:


# Training
autoencoder.fit(x=X_train_scaled, y=X_train_scaled,
                batch_size=40, epochs=500,
                validation_data=(X_test_scaled, X_test_scaled),
                callbacks=[early], verbose=0);


# In[ ]:


# Creating our encoded datasets 
encoded_x_train = encoder.predict(X_train_scaled)
encoded_x_test = encoder.predict(X_test_scaled)


# In[ ]:


# Creating a model and setting penalty to None since we already have removed many information
ae_log = LogisticRegression(penalty='none', solver='lbfgs', random_state=42)

pred = ae_log.fit(encoded_x_train, y_train).predict(encoded_x_test)

ae_train = cross_validate(ae_log, encoded_x_train, y_train, cv=10,
                          scoring=['accuracy', 'recall'])['test_recall']

ae_train = ae_train.mean()

print('AutoEncoder scores:')
print(f'Train Score: {ae_train:.4f}')
print(f'Test Score:  {recall_score(y_test, pred):.4f}')


# <hr>

# ## <font color='darkgreen'><font color='darkgreen'>B.3 - Linear Discriminant Analysis
# Using LDA as a dimensionality reduction is fundamentally different than the previous two: it is a supervised learning method. Similar to PCA it searches the data for the best 'direction'. But, instead of using the data's variance, it finds the best axis so the classes are as distant from eachother as possible.
# 
# The problem with LDA is we are going from 30 features to just one. It would be better if we didn't have to reduce it so much.
# 
# There is a solution we can try: we can split the dataset in three: the means, the errors and the worst features. On each of those three we can apply the LDA dimensionality reduction. This way we can have three features originated from the full dataset.

# In[ ]:


lda = LinearDiscriminantAnalysis()

mean_lda = lda.fit_transform(X_train_scaled[:,:8], y_train)
mean_lda_test = lda.transform(X_test_scaled[:,:8])

error_lda = lda.fit_transform(X_train_scaled[:,8:17], y_train)
error_lda_test = lda.transform(X_test_scaled[:,8:17])

worst_lda = lda.fit_transform(X_train_scaled[:,17:], y_train)
worst_lda_test = lda.transform(X_test_scaled[:,17:])

lda_train_feats = np.c_[mean_lda, error_lda, worst_lda]
lda_test_feats = np.c_[mean_lda_test, error_lda_test, worst_lda_test]


# In[ ]:


tlda_log = LogisticRegression(penalty='none', solver='lbfgs', random_state=42)
pred = tlda_log.fit(lda_train_feats, y_train).predict(lda_test_feats)

tlda_train = cross_validate(tlda_log, lda_train_feats, y_train, cv=10,
                          scoring=['accuracy', 'recall'])['test_recall']

tlda_train = tlda_train.mean()

print('Triple LDA scores:')
print(f'Train Score: {tlda_train:.4f}')
print(f'Test Score:  {recall_score(y_test, pred):.4f}')


# PCA had the worst result so we're dropping it. We can now tune two sets of models for the Autoencoder and the LDA data.

# <hr>

# ## <font color='darkgreen'>B.4 - Model Tuning and Testing
# Doing a quick round of parameter tuning for two models: Logistic Regression and SVC.
#     
# ### <font color='darkgreen'>Using Encoded Features

# In[ ]:


#### Logistic Regression AutoEncoder Parameters
log_pams = [{'solver':['liblinear'],
             'class_weight':[None, 'balanced'],
             'C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
             'penalty':['l1', 'l2']},
            {'solver':['lbfgs'],
             'class_weight':[None, 'balanced'],
             'C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
             'penalty':['l2']}]

log_ae = LogisticRegression(random_state=seed)
log_ae_gs = GridSearchCV(log_ae, log_pams, scoring='recall',
                           cv=skf, n_jobs=-1, iid=False)

train_gridsearch(log_ae_gs, x=encoded_x_train, y=y_train, name='Logistic Regression')
log_ae = log_ae_gs.best_estimator_

#### SVC AutoEncoder Parameters
svc_pams = {'kernel':['rbf'],
            'class_weight':[None, 'balanced'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
            'gamma':['auto', 'scale', 0.001, 0.01, 0.1]}

svc_ae = SVC(random_state=42)
svc_ae_gs = GridSearchCV(svc_ae, svc_pams, 'f1',
                      cv=skf, n_jobs=-1, iid=False,
                      refit=True)

train_gridsearch(svc_ae_gs, x=encoded_x_train, y=y_train, name='SVC')
svc_ae = svc_ae_gs.best_estimator_

ae_names = ['Logistic Regression', 'SVC']
ae_models = [log_ae, svc_ae]


# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(8,3), dpi=80)
fig.subplots_adjust(wspace=0.3)

for name, mod, ax in zip(ae_names, ae_models, axes):
    pred = mod.predict(encoded_x_test)
    confusion_plot(y_test, pred, ax, name)

for tick in axes[0].get_yticklabels():
    tick.set_visible(True)

axes[0].set_ylabel('True');


# It is a good score, considering that we did not change the threshold and our dataset is so much smaller now. However, the stacked model had a better score in Chapter Six. 
# 
# ### <font color='darkgreen'>Using Triple-LDA features

# In[ ]:


#### Logistic Regression LDA Parameters
log_pams = [{'solver':['liblinear'],
             'class_weight':[None, 'balanced'],
             'C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
             'penalty':['l1', 'l2']},
            {'solver':['lbfgs'],
             'class_weight':[None, 'balanced'],
             'C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
             'penalty':['l2']}]

log_lda = LogisticRegression(random_state=seed)
log_lda_gs = GridSearchCV(log_lda, log_pams, scoring='recall',
                           cv=skf, n_jobs=-1, iid=False)

train_gridsearch(log_lda_gs, x=lda_train_feats, y=y_train, name='Logistic Regression')
log_lda = log_lda_gs.best_estimator_

#### SVC LDA Parameters
svc_pams = {'kernel':['rbf'],
            'class_weight':[None, 'balanced'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
            'gamma':['auto', 'scale', 0.001, 0.01, 0.1]}

svc_lda = SVC(random_state=42)
svc_lda_gs = GridSearchCV(svc_lda, svc_pams, 'f1',
                      cv=skf, n_jobs=-1, iid=False,
                      refit=True)

train_gridsearch(svc_lda_gs, x=lda_train_feats, y=y_train, name='SVC')
svc_lda = svc_lda_gs.best_estimator_

lda_names = ['Logistic Regression', 'SVC']
lda_models = [log_lda, svc_lda]


# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(8,3), dpi=80)
fig.subplots_adjust(wspace=0.3)

for name, mod, ax in zip(lda_names, lda_models, axes):
    pred = mod.predict(lda_test_feats)
    confusion_plot(y_test, pred, ax, name)

for tick in axes[0].get_yticklabels():
    tick.set_visible(True)

axes[0].set_ylabel('True');


# Amazing!! This Logistic Regression Model has the best score we had so far. With the LDA features we managed to reduce 60% of the errors from our previous best score!
# 
# **Conclusions:**
# - All the three methods had decent untuned results, considering how much smaller the reduced datasets are
# - PCA is the simplest method for dimensionality reduction and had the worst result between them
# - Autoencoder had a promising first result and a very good tuned result, getting a really close score to our stacked model
# - The triple LDA features performed above the expectations. **Even though we are going from 30 to 3 features (10x less data), we got the best result of all models in this analysis**.
