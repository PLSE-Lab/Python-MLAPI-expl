#!/usr/bin/env python
# coding: utf-8

# # Ensembling / Stacking HR with python
# 
# Hello everyone, this notebook is a simple example of how to apply a method known as stacking in order to improve predictions on employees churn in a fictional company.
# 
# Stacking is a model ensembling method, in which information from multiple predictive models (level one models) is combined to create a new predictive model (level two modelsl). These new models often outperforms the individual models since they attempt to mitigate aspects of the basic models that might be causing poor performances. For further information I recommend checking this post:
# 
# Ben Gorman: [A Kaggler's Guide to Model Stacking in Practice](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
# 
# This notebook is divided in four main parts:
# 1. A quick exploratory data analysis;
# 2. Pre-processing and feature engineering;
# 3. Level one models;
# 4. Level two models.
# 
# Lastly but of most importance, I strongly recommend checking out this kaggle kernel by Anisotropic, which served as a model for this notebook:
# 
# Anisotropic: [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
# 
# Any feedback, advice, corrections, critics, questions are welcome.

# # Introduction
# 

# In[ ]:


# Importing used libraries
# Analysis and data processing
import pandas as pd # organization
import numpy as np # maths
import matplotlib.pyplot as plt # Viz
import seaborn as sns # More viz
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split # Splitting the dataset into training and test sets
from sklearn.linear_model import LogisticRegression # Logistic regression
from sklearn.naive_bayes import GaussianNB # Naive-Bayes
from sklearn.model_selection import GridSearchCV # Grid Search for model optimization
# The four algorithms that will be used in the stacking
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.svm import SVC # Support vector machines
from sklearn.model_selection import KFold; # Cross validation
from sklearn.model_selection import cross_val_score # Cross validation score
from sklearn.ensemble import VotingClassifier # Voting Classifier
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# Fixed parameters
seed = 18
nfolds = 5
colors = ['faded green','faded red']


# In[ ]:


# Functions needed to create radar plots

# Radar plot functions (Source:https://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart)
# Credit to the user Kyler Brown
def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle 
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)           


# # Quick EDA
# ---
# 
# Familiarization with the dataset is pivital. This section will show several basic caracteristics of our features, while also creating hypothesis in order to understand what makes a employee churn or not.
# 
# We start off by importing the dataset and checking traditional problems. The dataset seems to be clean from the get go, so only null values and dataset balance will be verified.

# In[ ]:


# Importing the data
data = pd.read_csv('../input/HR_comma_sep.csv')
ds = pd.DataFrame(data = data)
ds.info()


# In[ ]:


# Checking dataset dependent variable balance
sns.set(style = "whitegrid")
sns.countplot(x = 'left', data = ds, palette = sns.xkcd_palette(colors))
plt.title('Employee churn')
plt.xlabel('Churn')
print(data.left.value_counts() / len(ds))


# In[ ]:


ds.head()


# The dataset seems balanced, with 76% negatives, 24% positives, and no null values.
# 
# We can observe that we have 4 continuous, 3 binary and 3 categorical variables:
# * Continuous: Satisfaction level, last evaluation, average montly hours and time spend on the company;
# * Binary: Work accidents, promotions in the last 5 years and if the employee left or not;
# * Categorical: Number of projects, the deparment the belong to (sales) and the salary.

# **Satisfaction Level**
# 
# Satisfaction level ranges from 0 to 1, expressing the happiness of the employee, having 0 as the lowest value and 1 as the highest.

# In[ ]:


fg = sns.FacetGrid(ds, hue = 'left', aspect = 4, palette = sns.xkcd_palette(colors))
fg.map(sns.kdeplot, 'satisfaction_level', shade = True)
fg.set(xlim=(0, ds['satisfaction_level'].max()))
plt.title('Churn - Satisfaction Level')
fg.add_legend(), fg.set_axis_labels('Satisfaction Level', 'Frequency')


# We can see from the kdeplot that:
# * People who stayed show a standard behavior, employees with high satisfaction levels usually stay in the company;
# * People who left show a multimodal distribution, having their peaks at levels 0.1, 0.4 and 0.8;
# * There is a strange occurrence on the bounds of the 0.2 range, where there is a very low frequency on the churn rate which can't be explained by this kdplot alone.
# * It would be a good idea to categorize this column into 10 diferent levels and checking the countplot in order to try to understand if it's a sample issue.

# ** Last evaluation**
# 
# Last evaluation is the amount of time since last performance evaluation (in years). I feel this is a mistake in the data description, since the way it's shown (ranging from 0 to 1) is not coherent with what it is stated to be.
# 
# Therefor, it will be considered that last evaluation is the evaluation score, ranging from 0 to 1, each employee got from a last evaluation.

# In[ ]:


le = sns.FacetGrid(ds, hue = 'left', aspect = 4, palette = sns.xkcd_palette(colors))
le.map(sns.kdeplot, 'last_evaluation', shade = True)
le.set(xlim=(0.35 , 1.1))
plt.title('Churn - Last Evaluation')
le.add_legend(), le.set_axis_labels('Last Evaluation', 'Frequency')


# What can we see here:
# * Lower evaluations show a high churn rate, which can be expected;
# * High evaluations also show a high churn rate, this can be explain by the employee not feeling valued by the company;
# * It can be interesting for future analysis to cross the last evaluation with the satisfaction level in order to verify these hypothesis.

# **Average montly hours**
# 
# The average montly hours each employee spent on the workplace, with a range from 84 to 350.

# In[ ]:


# Average Montly Hours
amh = sns.FacetGrid(ds, hue = 'left', aspect = 4, palette = sns.xkcd_palette(colors))
amh.map(sns.kdeplot, 'average_montly_hours', shade = True)
amh.set(xlim=(85 , 350))
plt.title('Churn - Average Montly Hours')
amh.add_legend(), amh.set_axis_labels('Average Montly Hours', 'Frequency')


# The plot shows that:
# * Employees with the least hours seem to have a high churn rate, maybe due to lack of projects for them to participate in;
# * Employees with the most hours also have a high churn rate, probably because they are being overworked;
# * This pattern can also be seen in the feature 'Number of projects'

# **Number of projects**
# 
# Contains the number of projects each employee completed at work. This feature ranges from 2 to 7.
# 
# For this feature, I rather view it as currently enrolled projects, as it should be more important to predict if an employee would leave or not.

# In[ ]:


num_p = sns.countplot(x = 'number_project', hue = 'left', data = ds, palette = sns.xkcd_palette(colors))
plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, title = 'left')
plt.xlabel('Number of completed projects')
plt.title('Churn - Number of projects ')


# For the number of completed projects:
# * High churn rates for the lowest number of projects (2), this can be due the employee either not being able to complete enrolled projects or feeling underworked;
# * Employees with more than 5 projects seem to be likely to leave the company, probably because they are overworking;
# * It would be interesting to know the mean number of hours for employees with more projects and then compare it to the medium number of projects, as well as their satisfaction level.

# **Salary**
# 
# Salary is refered to the payscale in which the employee is in. Salary is split in 3 categories, low, medium and high.

# In[ ]:


sales_depart = sns.countplot(x = 'salary', data = ds, hue = 'left', palette = sns.xkcd_palette(colors))
plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, title = 'left')
plt.title('Churn - Salary Class')
plt.xlabel('Salary Class')


# This doesn't give us much insights, the frequency of low and medium payscales employees is much higher than the high payscale. Further EDA would have to be done to understand possible trends.

# [](http://)**Sales**
# 
# Sales tells us the department of each employee. There are 10 unique classes.

# In[ ]:


sales_depart = sns.countplot(y = 'sales', data = ds, hue = 'left', palette = sns.xkcd_palette(colors))
plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, title = 'left') 
plt.title('Churn - Department')
plt.yticks(rotation=45, size = 12)


# Again there isn't much usefull information on this plot alone, the churn / not churn ratio seems to be proportional among all deparments. Further EDA would have to be done.

# **Work accidents**
# 
# Whether the employee had a workplace related accident. Binary feature, where 0 means no and 1 means yes.

# In[ ]:


# Work accidents
wa = sns.countplot(x = 'Work_accident', data = ds, hue = 'left', palette = sns.xkcd_palette(colors))
plt.title('Churn - Work accident')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, title = 'left')
plt.xlabel('Work accidents')


# As the last 2 features, there is no relevant information from this plot alone.

# **Time spent on company**
# 
# Number of years the employee is at the company, with a 2 to 10 range.

# In[ ]:


# Time Spent on Company
tsc = sns.countplot(x = 'time_spend_company', data = data, hue = 'left', palette = sns.xkcd_palette(colors))
plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, title = 'left')
plt.title('Churn - Years spent on the company')
plt.xlabel('Years spent on the company')


# The countplot shows that:
# * There is a strange reason at the fifth year that makes employees leave the company;
# * There is a very substancial increase in churn rate from the second year to the third;
# * Employees with a lot of years in the company tend to stay.

# **Promotion in the last 5 years**
# 
# A binary feature, stating if the employee was promoted in the last 5 years (1) or not (0).

# In[ ]:


# If employee had a promotion in the last 5 years
pl5y = sns.countplot(x = 'promotion_last_5years', data = ds, hue = 'left', palette = sns.xkcd_palette(colors))
plt.title('Churn - Promotion in the last 5 years')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, title = 'left')
plt.xlabel('Promotion in the last 5 years')


# Doesn't seem to have a relevant impact in the churn rate.

# **Removing irrelevant columns**
# 
#  Simple models usually outperform more complex models. According to the posteriorly done analysis and a analysis done with some algorithms and their feature contribution (not covered in the notebook) I decided to remove columns that had little contribution to the goal.

# In[ ]:


del ds['Work_accident']
del ds['promotion_last_5years']
del ds['sales']
del ds['salary']
ds.head(5)


# **Correlation matrix and associated heatmap**
# 
# It's important that in a classification problem there aren't high correlations between features, since this usually allows the model to make stronger predictions.

# In[ ]:


data_corr = ds.corr()
sns.heatmap(data_corr, annot = True, cmap = 'viridis', linewidths = 1, linecolor = 'white')


# We can see that there is:
# * Moderately positive correlation between the number of projects, last evaluation and average montly hours. This probably means that the higher the time spent working and number of projects, the higher the evaluation would be;
# * High negative correlation between churning and level of satisfaction, showing that the people least satisfied are more likely to leave the company.

# # Pre-processing and feature engineering
#  ---
# In this section the data will be pre-processed in order to be used by the algorithms. Continuous variables will be encoded (feature engineering) and the dataset will be split into the training set and the test set.
# 
# Afterwards some useful functions and a class, that will help create the data needed for our models will be shown.
# 
# **Feature engineering**

# In[ ]:


# Creating 3 satisfaction level bins
sf_holder = pd.cut(ds['satisfaction_level'], 3)
sf_holder.unique()


# In[ ]:


# Creating 4 last evaluation bins
le_holder = pd.cut(ds['last_evaluation'], 4)
le_holder.unique()


# In[ ]:


# Creating 4 average montly hours bins
avgmh = pd.cut(ds['average_montly_hours'], 4)
avgmh.unique()


# In[ ]:


# Enconding according to the bins limits

# Satisfaction level
ds.loc[(ds['satisfaction_level'] > 0.697) & (ds['satisfaction_level'] <= 1.0), 'satisfaction_level'] = 2
ds.loc[(ds['satisfaction_level'] > 0.393) & (ds['satisfaction_level'] <= 0.697), 'satisfaction_level'] = 1
ds.loc[(ds['satisfaction_level'] <= 0.393), 'satisfaction_level'] = 0
del sf_holder # Droping holder column

# Last evaluation
ds.loc[(ds['last_evaluation'] > 0.84) & (ds['last_evaluation'] <= 1.0), 'last_evaluation'] = 3
ds.loc[(ds['last_evaluation'] > 0.68) & (ds['last_evaluation'] <= 0.84), 'last_evaluation'] = 2
ds.loc[(ds['last_evaluation'] > 0.52) & (ds['last_evaluation'] <= 0.68), 'last_evaluation'] = 1
ds.loc[(ds['last_evaluation'] <= 0.52), 'last_evaluation'] = 0
del le_holder # Droping holder column

# Average montly hours
ds.loc[(ds['average_montly_hours'] <= 149.5), 'average_montly_hours'] = 0
ds.loc[(ds['average_montly_hours'] > 149.5) & (ds['average_montly_hours'] <= 203.0), 'average_montly_hours'] = 1
ds.loc[(ds['average_montly_hours'] > 203.0) & (ds['average_montly_hours'] <= 256.5), 'average_montly_hours'] = 2
ds.loc[(ds['average_montly_hours'] > 256.5) & (ds['average_montly_hours'] <= 310.0), 'average_montly_hours'] = 3
del avgmh # Droping holder column

ds.head()


# **Splitting the dataset into a train and test set**

# In[ ]:


# Creating the classification features
x = ds.iloc[:, 0:5].values
y = ds.iloc[:, 5].values

# Creating the training set (80%) and the test set (20%), random_state for repeatability 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = seed)


# **Helping tools**
# 
# The created class (clfBuilder) and function (Out-of-folds) shown later in the notebook are derivative from a script created by the user Fargo and I highly suggest checking it out:
# 
# Fargo: [Stacking Starter](https://www.kaggle.com/mmueller/stacking-starter)
# 
# **Class clfBuilder**
# 
# The clfBuilder class will help in avoiding  extended coding everytime a different classifier is used. This class will use common SKlearn classifiers inbuilt methods (train, predict, feature importance and score).

# In[ ]:


class clfBuilder(object):
    def __init__(self, clf, seed = seed, params = None):
        params['random_state'] = seed
        self.clf = clf(**params)
        
    def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)
        
    def predict(self, x_test):
            return self.clf.predict(x_test)
        
    def feature_importances(self, x, y):
            print(self.clf.fit(x, y).feature_importances_)
        
    def get_score(self, x, y):
            print(self.clf.score(x, y))
    
    def save_score(self, x, y):
        return self.clf.score(x, y)


# **SmallTimeEnsembler**
# 
# The idea behind it is creating a function that will allow us to quickly see results from multiply classifiers, removing redundant coding, while also trying to mitigate wrong predictions that can result from the algorithms modeling properties.

# In[ ]:


def smallTimeEnsembler(x, y):

    lgr = LogisticRegression(C = 0.5, random_state = 18,)
    rfc = RandomForestClassifier(random_state=18)
    nvb = GaussianNB()
    svc = SVC(C = 0.5)
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    clf_names = ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Kernel SVC',
                 'KNNeighbors', 'Ensemble']
    
    ensembler = VotingClassifier(estimators = [('lr', lgr), ('rf', rfc), ('nvb', nvb),
                                               ('SVC', svc), ('knn', knn)], voting = 'hard')
    for clf, label in zip([lgr, rfc, nvb, svc, knn, ensembler], clf_names):
        scores = cross_val_score(clf, x, y, cv = 5, scoring = 'accuracy')
        print('Accuracy: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), scores.std(), label))
        
smallTimeEnsembler(x_train, y_train)


# Unfortunately it's hard to evaluate the effectiveness of the smallTimeEnsembler since there were serveral high percent predictions.
# 
# I found that trying to input a parameter optimizing field would greatly increase the complexity of the function. Therefor any parameter optimizing needs to be inserted into the function code.

# # **Level one models**
# 
# The four used algorithms for the stacking, are ensemblers themselfs:
# 
# * *AdaBoost* - Boosting classifier that uses the output of the other learning algorithms and combines them into a weighted sum that represents the final output of the boosted classifier;
# 
# * *Random* *Forest* - Constructs many decision trees then outputs the mode or the mean of the predictions;
# 
# * *Gradient Boost* -  Optimizes arbitrary differentiable loss functions;
# 
# * *Extra Trees* -  meta estimator that fits a number of randomized decision trees on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.
# 
# For more information on this topic, click this [link](http://scikit-learn.org/stable/modules/ensemble.html).
# 
# These models have different optimization parameters which can boost the prediction score. In order to find the optimal parameters, GridSearchCV will be used. GridSearchCV searchs specified parameters values of an estimator then these parameters are optimized by cross-validation grid-search. For more information check [this link](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
# 
# Again, in order to reduce the amout of redudant code the function *gridsearchEngine* was created as shown below.

# In[ ]:


def gridsearchEngine(clf, params, data):
    # clf is the classifier, params is a dict with the paramters.
    
    x = data.iloc[:, [0,4]].values
    y = (data.iloc[:, 5]).ravel()
    grid = GridSearchCV(estimator= clf,
                        param_grid= params,
                        scoring = 'accuracy',
                        cv = 5,
                        n_jobs = -1)
    grid.fit(x, y)
    print('Best Score:', grid.best_score_)
    best_params = grid.best_estimator_
    return best_params


# In[ ]:


# GridSearch parameters

# RandomForests parameters
rfc_params = {
    'n_estimators': range(100, 500, 100),
    'criterion': ('gini','entropy'),
    'max_depth': range(15, 30, 5),
    'max_features' : ['sqrt', 'auto', 'log2']}

# ExtraTrees parameters
extree_params = {
    'n_estimators':(100, 250, 500), 
    'max_features': [0.5, 1],
    'max_depth': [5, 8, 12],
    'min_samples_leaf': [1, 2]}    

# GradientBoost parameters (takes forever)
gbm_params = {'learning_rate': (0.05, 0.1, 0.2, 0.5), 
              'n_estimators': (250, 500, 750), 
              'min_samples_split':range(200,1001,200),
              'subsample':[0.7,0.75,0.8,0.85,0.9], 
              'max_depth': [3,4,5]}

# AdaBoost parameters
adaboost_params = {'n_estimators':[25,50,100,500],
    'learning_rate':[0.5, 1.0, 3]}


# In[ ]:


# Creating the gridsearch for each classifier (commented for notebook speed)

#rfc_gs = gridsearchEngine(RandomForestClassifier(), rfc_params, ds)
#gbm_gs = gridsearchEngine(GradientBoostingClassifier(), gbm_params, ds)
#extree_gs = gridsearchEngine(ExtraTreesClassifier(), extree_params, ds)
ada_gs = gridsearchEngine(AdaBoostClassifier(), adaboost_params, ds) # As an example


# In[ ]:


ada_gs


# In[ ]:


# Optimal parameters according to the grid search

# Random Forest optimal parameters
rfc_optimal = {'criterion':'gini',
               'max_depth': 15,
               'max_features': 'sqrt',
               'n_estimators': 100
              }

# Extra tree optimal parameters
extree_optimal = {'criterion': 'gini',
           'max_depth': 5, 
           'max_features': 0.5,
           'min_samples_leaf': 1,
           'n_estimators': 100
                 }

# GradientBoost optimal parameters
gbm_optimal = {'criterion':'friedman_mse',
              'learning_rate': 0.05, 
              'loss': 'deviance', 
              'max_depth': 3,
              'min_samples_leaf': 1, 
              'min_samples_split': 200,
              'n_estimators': 250,
              'subsample':0.7}

# AdaBoost optimal parameters
adaboost_optimal = {'n_estimators':500,
                    'learning_rate':0.5,
                    'random_state': 18}

# For later use
lvl1_scv_optimal = {'C':0.1,
              'kernel':'rbf',
              'degree':3, 
              'shrinking': True}


# Having the optimal parameters defined, creating our 4 level one boosting model objetcs:

# In[ ]:


model_label = ['Adaboost', 'GradientBoost', 'ExtraTrees', 'RandomForest', 'Kernel SVC']
rfc = clfBuilder(RandomForestClassifier, seed, rfc_optimal)
gbm = clfBuilder(GradientBoostingClassifier, seed, gbm_optimal)
extree = clfBuilder(ExtraTreesClassifier, seed, extree_optimal)
adaboost = clfBuilder(AdaBoostClassifier, seed, adaboost_optimal)


# It's important to verify how did our features contribute to the models predictions, because if all the classifiiers prioritize the same feature we might be feeding our ensembler redudant information.
# 
# Using the *featureCheck* function, the feature importance will be calculated for each classifier and then used to plot a radar  chart.

# In[ ]:


lvl1_models = [rfc, gbm, extree, adaboost]
model_name = ['RandomForest', 'GradientBoost', 'Extra Trees', 'AdaBoost']
def featureCheck(clf_list, clf_names, x, y):
    for clf, label in zip(clf_list, clf_names):
        print(label, 'feature importance:')
        clf.feature_importances(x, y)
        print()
        
featureCheck(lvl1_models, model_name, x_train, y_train)


# For some unknown reason, the radar chart labels aren't rotating as intended (which still work on jupyter notebook), will try to fix it later.

# In[ ]:


# Visualizing the feature contribution for each classifier

rfc_fi = (0.13626274, 0.13096811, 0.32626718, 0.11477031, 0.29173167)
gbm_fi = (0.20941982, 0.13287447, 0.24964535, 0.11205148, 0.29600888)
extree_fi = (0.21779109, 0.14936934, 0.33238019, 0.14609129, 0.15436808)
adaboost_fi = (0.094, 0.168, 0.348, 0.15, 0.24)

ranges = [(0.1, 0.4), (0.1, 0.4), (0.1, 0.4),
         (0.1, 0.4), (0.1, 0.4), (0.1, 0.4), (0.1, 0.4)]     
label = ('Satisfaction Level', 'Last Evaluation', 'Number of Projects',
       'Average Montly Hours', 'Time Spend Company')


# In[ ]:


# Random Forest Classifier
fig1 = plt.figure(figsize=(6, 6))
radar = ComplexRadar(fig1, label, ranges)
radar.plot(rfc_fi)
radar.fill(rfc_fi, alpha=0.20)
plt.title('RFC')
plt.show()


# In[ ]:


# GradientBoost Classifier
fig2 = plt.figure(figsize=(6, 6))
radar = ComplexRadar(fig2, label, ranges)
radar.plot(gbm_fi)
radar.fill(gbm_fi, alpha=0.2)
plt.title('GB')
plt.show()


# In[ ]:


# Extra Trees classifier
fig3 = plt.figure(figsize=(6, 6))
radar = ComplexRadar(fig3, label, ranges)
radar.plot(extree_fi)
radar.fill(extree_fi, alpha=0.2)
plt.title('ExT')
plt.show()


# In[ ]:


# AdaBoost classifier
fig4 = plt.figure(figsize=(6, 6))
radar = ComplexRadar(fig4, label, ranges)
radar.plot(adaboost_fi)
radar.fill(adaboost_fi, alpha=0.2)
plt.title('ADA')
plt.show()


# **Out-of-folds function**
# 
# The Out-of-folds function (here named foldGenerator) uses cross-validation to create a new feature set with the the predictions for each row then stores them in new matrices so they can be used to train and test our new classifiers and produce a new set of predictions.

# In[ ]:


# Creating k-fold object with nfolds splits
kfolds = KFold(n_splits = nfolds, random_state = seed)

def foldGenerator(clf, x_train, y_train, x_test):
    # Clf is the classifier object

    xtrain_shape = x_train.shape[0] # Size of x_train
    xtest_shape = x_test.shape[0] # Size of x_test 
    
    # Creating 2 matrices with the size of x_train and x_test
    train_matrice = np.zeros((xtrain_shape,))
    test_matrice = np.zeros((xtest_shape,))

    test_kf = np.empty((nfolds, xtest_shape))
    
    # Creating matrices with indices for both x's
    for i, (train_index, test_index) in enumerate(kfolds.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index] 

        clf.train(x_tr, y_tr)
        
        # Filling the matrices with the predictions
        train_matrice[test_index] = clf.predict(x_te)
        test_kf[i, :] = clf.predict(x_test)
        
    test_matrice[:] = test_kf.mean(axis=0)
    
    # Reshaping with 1 column and allowing numpy to figure out the rest
    return train_matrice.reshape(-1, 1), test_matrice.reshape(-1, 1)


# In[ ]:


# Creating the first level outputs
rfc_fg_train, rfc_fg_test = foldGenerator(rfc, x_train, y_train, x_test) # Random Forest
extree_fg_train, extree_fg_test = foldGenerator(extree, x_train, y_train, x_test) # Extra Trees
gbm_fg_train, gbm_fg_test = foldGenerator(gbm, x_train, y_train, x_test) # Gradient Boost
ada_fg_train, ada_fg_test = foldGenerator(adaboost, x_train, y_train, x_test) # Ada Boost


# In[ ]:


# For later use
svc_normal = clfBuilder(SVC, seed = seed, params = {})
svc_normal.train(x_train, y_train)
svcn_train_score = svc_normal.save_score(x_train, y_train)
svcn_test_score = svc_normal.save_score(x_test, y_test)

svc_lvl1_optimal = clfBuilder(SVC, seed = seed, params = lvl1_scv_optimal)
svc_lvl1_optimal.train(x_train, y_train)
svc1_train_score = svc_lvl1_optimal.save_score(x_train, y_train)
svc1_test_score = svc_lvl1_optimal.save_score(x_test, y_test)


# In[ ]:


# I think it's very important to understand what exacly the new feature set looks in order to
# understand what is happening
firstlevel_train = pd.DataFrame({
    'RandomForest': rfc_fg_train.ravel(),
     'ExtraTrees': extree_fg_train.ravel(),
     'GradientBoost': gbm_fg_train.ravel(),
      'AdaBoost': ada_fg_train.ravel()
    })

# firstlevel_train is a feature like dataset that contains the 
# predictions from each classifier on each row of the training set
firstlevel_train.head(10)


# **New correlation matrix**

# In[ ]:


fltrain_corr = firstlevel_train.corr()
sns.heatmap(fltrain_corr, annot = True, cmap = 'viridis', linewidths = 1, linecolor = 'white')


# As before mentioned, we want low correlation between the variables in order to provide the most distinct information to the ensembler. Unfortunately our variables are highly correlated which can indicate that the ensembler won't be able to greatly improve our predictions.

# # Level two models
# ---
# After stacking the new features, they can be used as inputs in a new classifier to verify if the accuracy improves due to the ensembling method or not.
# 
# The chosen ensemblers were XGBoost, Artificial Neural Network and Support Vector Machines, but I wasn't able to make XGBoost work on Python due to a installation issue (my guess).
# 
# This section will be divided in the implentation of the artificial neural network and SVC ( Support Vector Machines), comparing the predictions scores with a optimized SVC with the former feature set and a new  SVC with the new feature set. I think this will be helpfull to start to have an grasp of what are the magnitude of the improvements there can be with using a more complex predictor (ANN) vs a simpler one (SVC) and optimizing the classifiers paremeters.
# 

# In[ ]:


# Creating the new x_train and x_test
x_train = np.concatenate((rfc_fg_train, extree_fg_train, 
                          gbm_fg_train, ada_fg_train), axis = 1)
x_test = np.concatenate((rfc_fg_test, extree_fg_test, 
                          gbm_fg_test, ada_fg_test), axis = 1)


# **Artificial Neural Network**
# 
# There was a lot that could be said about ANNs,but it would take too much time to correctly **try** to explain how they work. If you are curious about how they work you can follow this [Wikipedia link](https://en.wikipedia.org/wiki/Artificial_neural_network).

# In[ ]:


# Keras ANN

# Initialising the ANN
ann = Sequential()

# Adding the input layer and the first hidden layer
ann.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
# Adding the second hidden layer
ann.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer (1 output layer because it's a classification problem)
ann.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN (Performing stochastic gradient descendent to the ANN)
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


ann.fit(x_train, y_train, batch_size = 10, epochs = 100, validation_data = (x_test, y_test), shuffle=False)


# The ANN returns a pretty good prediction score with an accuracy of 0.9655 on the training set and 0.9647 on the test set. If we look at the epochs, we see that the loss function was soon minimize and the accuracy had little variation which may be related to the high correlation in our new feature set.
# 
# It is possible that if you re-rerun this your results will be very different, since the ANN weights are initialised randomly according to their initialization. I got some very low scores (74% training set, 73% test set) and very high as shown (96% training set, 96% test set).
# 
# **Support vector machine**
# 
# The SVM classifier maps the features as points in a multi dimensional space, in a way that the features of the different categories are divided by the widest gap possible. The new inputs are mapped into the new dimensional space, predicting to which region they belong. 
# 
# Still, the SVM is an linear classifier, in order to perform non-linear classification it is implemented what is known as the kernel trick is used. The kernel trick maps the features into a higher-dimensional feature space without the need to compute the coordinates of our features, computing instead the inner products between features. This method is often refered as SVC (C-Support Vector Classification).
# 
# Previously a SVC object (svc_lvl1) was trained with the original feature set and the performance on the training and test set saved in order to compare to our new predictive models. 

# In[ ]:


# Creating a level 2 SVC with the new data
svc_lvl2_optimal = {'C': 0.1,
              'kernel': 'linear',
              'cache_size': 100, 
              'shrinking': True,
              'decision_function_shape' : 'ovo'}

svc_lvl2 = clfBuilder(SVC, seed = seed, params = svc_lvl2_optimal)
svc_lvl2.train(x_train, y_train)
svc2_train_score = svc_lvl2.save_score(x_train, y_train)
svc2_test_score = svc_lvl2.save_score(x_test, y_test)


# In[ ]:


ensembler_score = {
    'SVC Level 1 (no params)': pd.Series([svcn_train_score, svcn_test_score, (svcn_train_score + svcn_test_score)/2], index = ['Training Score', 'Test Score', 'Average Score']),
    'SVC Level 1': pd.Series([svc1_train_score, svc1_test_score, (svc1_train_score + svc1_test_score)/2], index = ['Training Score', 'Test Score', 'Average Score']),
    'SVC Level 2': pd.Series([svc2_train_score, svc2_test_score, (svc2_train_score + svc2_test_score)/2], index = ['Training Score', 'Test Score', 'Average Score']),
    'Artificial Neural Network': pd.Series([0.9652,  0.9647, (0.9647+0.9652) / 2 ], index = ['Training Score', 'Test Score', 'Average Score'])
}

ensembler_accuracy = pd.DataFrame(ensembler_score)
ensembler_accuracy


# We can see that:
# * By optimizing the SVC on the orginal dataset we get a 0.004% decrease in the prediction score. The reason behind this is a mystery to me;
# * There is a 0.005% increase by using the new feature set instead of the old one, meaning that our goal was achieved;
# * The ANN achieved the best score with 96.495%, improving the stacked SVC by 0.0004%.
# 
# 

# # **Conclusions**
# ---
# * The results are satisfactory, the ensemble method allow us to improve our predictions as intended, but in a very small percentage.
# 
# * For this specific prediction problem, the ANN consumes almost as much time to train and predict as the rest of the notebook, with a very small increase in accuracy making, in my opinion, the level two SVC the best suited classifier.
# 
# * The decrease in accuracy after optimizing the classifier parameters leads me to believe there may be a possible overlooked error.

# # Final thoughts
# 
# I hope I was able to explain a moderately hard topic in a simple way, unfortunately I lack the time at the moment to produce an R code just to test out the predictions with XGBoost.
# 
# This is my first kaggle notebook, again, any feedback, advice, corrections, critics, questions are very welcome.
# 
# So ye, thanks for your time, upvote if you enjoyed it!
