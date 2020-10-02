#!/usr/bin/env python
# coding: utf-8

# # Identifying fraud from Enron e-mails

# ## Table of Contents
# * <a href="#sec1">1. Introduction</a>
#   * <a href="#sec1.1"> 1.1. Initial statements </a>
# 
# * <a href="#sec2">2. Data Wrangling</a>
#   * <a href="#sec2.1"> 2.1. Loading the data </a>
#   * <a href="#sec2.2"> 2.2. Analyzing string data </a>
#   * <a href="#sec2.3"> 2.3. Converting string "NaN" to numpy.NaN </a>
#   * <a href="#sec2.4"> 2.4. Counting valid data </a>
#   * <a href="#sec2.5"> 2.5. Outlier removal </a>
#   * <a href="#sec2.6"> 2.6. Checking the class distribution </a>
#   * <a href="#sec2.7"> 2.7. Stratified data split </a>
# 
# * <a href="#sec3">3. Feature selection</a>
#   * <a href="#sec3.1"> 3.1. Verifying correlation among variables </a>
#   * <a href="#sec3.2"> 3.2. Applying PCA on correlated features to generate a new one </a>
#   * <a href="#sec3.3"> 3.3. Univariate feature selection </a>
#   * <a href="#sec3.4"> 3.4. Alternative feature selection </a>
#   * <a href="#sec3.5"> 3.5. Feature scaling </a>
#   
# * <a href="#sec4">4. Machine Learning</a>
#   * <a href="#sec4.1"> 4.1. Naive Bayes classifier </a>
#   * <a href="#sec4.2"> 4.2. AdaBoost classifier </a>
#   * <a href="#sec4.3"> 4.3. SVM classifier </a>
#   * <a href="#sec4.4"> 4.4. Conclusions</a>

# <a id='sec1'></a>
# ## 1. Introduction
# This project is part of the Udacity Data Analyst Nanodegree and refers to Intro to Machine Learning module. Its main scope is to build a person of interest (POI) identifier based on financial and email data made public as a result of the Enron scandal, as explained below.
# 
# In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives

# <a id='sec1.1'></a>
# ### 1.1. Initial statements
# This section sets up import statements for all the packages that will be used throughout this python notebook.

# In[ ]:


# Udacity statements
import pickle

# Data analysis packages:
import pandas as pd
import numpy as np
#from datetime import datetime as dt

# Visualization packages:
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## Importing "manually" some functions provided by Udacity and available at 
## https://github.com/tbnsilveira/DAND-MachineLearning/blob/master/tools/feature_format.py
def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """
    return_list = []
    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )
    return np.array(return_list)

def targetFeatureSplit( data ):
    """ given a numpy array like the one returned from featureFormat, separate out the first feature and put it into its own list
    (this should be the quantity you want to predict) return targets and features as separate lists (sklearn can generally 
    handle both lists and numpy arrays as input formats when training/predicting) """
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )
    return target, features


# In[ ]:


## Forcing pandas to display any number of elements
pd.set_option('display.max_columns', None)
pd.options.display.max_seq_items = 2000


# <a id='sec2'></a>
# ## 2. Data wrangling
# Analyzing the data integrity, the expected values and removing outliers.

# <a id='sec2.1'></a>
# ### 2.1. Loading the data

# In[ ]:


data_dict = pd.read_pickle("../input/final_project_dataset.pkl")


# In[ ]:


## What is the data type and length?
print('Dataset type: ',type(data_dict))
print('Dataset length: ',len(data_dict))


# In[ ]:


## Exploring the dataset through pandas.Dataframe
dataset = pd.DataFrame.from_dict(data_dict, orient='index')
dataset.head()


# <a id='sec2.2'></a>
# ### 2.2 Analyzing string data:

# In[ ]:


dataset.describe()


# In[ ]:


## Checking the feature data type:
features_to_check = []
for col in dataset.columns:
    datatype = type(dataset[col][0])
    ## Uncomment the line below for a verbose mode:
    # print '{} has type {}'.format(col,datatype)
    ## Here we select those attributes which have string type data:
    if datatype is str:
        features_to_check.append(col)


# In[ ]:


## Printing out the features that must be checked (string types are not iterable!)
features_to_check


# From the features above, only *email_address* is expected to contain string type data. In this way, an in-depth look must be done in the other ones.

# <a id='sec2.2.1'></a>
# #### 2.2.1 Checking *loan_advances* data:
# This attribute type is originally *(str type)*. However, it was expected to have financial values. 

# In[ ]:


dataset['loan_advances'].unique()


# Only four instances have *loan_advances* valid values. Checking them out:

# In[ ]:


dataset[dataset['loan_advances']!='NaN']


# The first outlier pops out from this data -- the **TOTAL** instance must be removed. Besides it, this feature occurs only for three valid instances, in the way maybe it's not the best feature to feed our classifier.

# <a id='sec2.2.2'></a>
# #### 2.2.1 Checking *director_fees* data:
# As for *loan_advances*, this attribute type is originally *(str type)*. However, it was expected to have financial values too. 

# In[ ]:


dataset['director_fees'].unique()


# In[ ]:


dataset[dataset['director_fees']!='NaN']


# Regarding the ***director_fees*** feature, only 17 instances contains valid value. What calls attention in this case is that most of the other features has **NaN** values, which brings suspection that maybe they refers to false names. Surely something that must be checked later. 

# <a id='sec2.3'></a>
# ### 2.3 Converting string "NaN" to numpy.NaN
# From the previous output, it was clear there are some 'NaN' in string type instead of numerical or numpy type, which causes some troubles when plotting data or using some classifier. Due to this, the next step is to scan the dataset for 'NaN' string and replace it by numpy.NaN.

# In[ ]:


for column in dataset.columns:
    dataset[column] = dataset[column].apply(lambda x: np.NaN if x == 'NaN' else x)


# In[ ]:


## Checking the dataset information:
dataset.info()


# <a id='sec2.4'></a>
# ### 2.4 Counting valid data
# As seen before, there are many null data in our dataset. In order to select the most appropriate features to explore, we will look for those that are present at least in 70% of the dataset. Considering there are 21 features (from which 70% is approximate to 15 features), we will first observe which instances have more than 15 not null values and choose the most complete features from this selection.

# In[ ]:


notNullDataset = dataset.dropna(thresh=15)


# In[ ]:


notNullDataset.info()


# From the output above, the features named *deferral_payments; restricted_stock_deferred;* and *director_fees* are removed. Doing so the pre-selected features list can be defined accordingly to its context:

# In[ ]:


## Only numerical features are being considered here
financialFeatures = ['salary','bonus', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi',
                     'total_payments', 'expenses', 'total_stock_value', 'deferred_income', 'long_term_incentive']
behavioralFeatures = ['to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'other']
allFeatures = ['poi','salary','bonus', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi',
               'total_payments', 'expenses', 'total_stock_value', 'deferred_income', 'long_term_incentive',
               'to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'other']


# Since in the *feature_format()* method the "NaN" strings are replaced by zeros, we need to make this adjustment here.

# In[ ]:


dataset.fillna(0,inplace=True)


# <a id='sec2.5'></a>
# ### 2.5 Outlier removal
# In this section we look for identifying and, if it is the case, removing outliers in our dataset.

# <a id='sec2.5.1'></a>
# ### 2.5.1 Visualizing data by features
# The objective here is to visualize how the data is distributed accordingly to each feature, as well as identifying outliers through visual inspection. For this we make use of a function build based on the code available in (http://stamfordresearch.com/outlier-removal-in-python-using-iqr-rule/)

# In[ ]:


def visualizeFeat(series, figsize):
    ''' series = pandas.series, which can be inputed as "dataframe['feature']
        figsize = (width,length)'''
    fig, axes = plt.subplots(2,1,figsize=figsize, sharex=True)
    series.plot(kind='kde', ax=axes[0])
    sns.boxplot(x=series, ax=axes[1])
    plt.xlim(series.min(), series.max()*1.1)
    return


# The function above seems useful to get information for only one variable. Since we want to explore the whole dataset (or at least a good chunk of it), we will code a function to show them all:

# In[ ]:


def visualize3Feats(dataset, features):
    '''Shows the distribution and the boxplot for the given features of a pandas.Dataframe:
        dataset = pandas dataframe.
        features = list of features of interest'''
    ## Building the Figure:
    fig, axes = plt.subplots(2,3,figsize=(15,6), sharex=False)
    for col, feat in enumerate(features):
        dataset[feat].plot(kind='kde', ax=axes[0,col])
        sns.boxplot(x=dataset[feat], ax=axes[1,col])
        axes[0,col].set_xlim(dataset[feat].min(), dataset[feat].max()*1.1);
        axes[1,col].set_xlim(dataset[feat].min(), dataset[feat].max()*1.1);
    return


# In[ ]:


### Visualizing financial features:
numPlots = int(np.ceil(len(financialFeatures)/3.))
for i in range(numPlots):
    shift = i*3
    visualize3Feats(dataset,financialFeatures[0+shift:3+shift])


# From the charts above, it becomes evident there is at least one strong outlier sample (which in fact was identified during the lesson as the TOTAL instance). Even after removing it (see Section 2.4.2) there are still remaining outliers. However, since they are probably related to what we are looking for, they won't be removed.

# <a id='sec2.5.2'></a>
# ### 2.5.2 Removing **TOTAL** instance
# As observed before, the "TOTAL" instance must be removed, since we are interested only on POIs.

# In[ ]:


dataset.drop('TOTAL',inplace=True)  #Removing the anomalous instance


# <a id='sec2.6'></a>
# ### 2.6 Checking the class distribution

# In[ ]:


## Counting gender classes
dataset['poi'].value_counts()


# <a id='sec2.7'></a>
# ### 2.7 Stratified data split
# Besides we can extract features from the whole dataset, when we are training machine learning algorithms is really important to split the data into training and testing subsets, in order to avoid overfitting. But as seen above, our data is unbalanced and so it is important to split data in a stratified way, i.e., each subset must have the same proportion of each class.

# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


## For pandas.Dataframe the train_test_split is given in a straight way:
trainData, testData = train_test_split(dataset, test_size=0.3, random_state=42, stratify=dataset['poi'])


# In[ ]:


## Converting boolean data into int:
dataset['poi'] = dataset['poi'].apply(lambda x: int(x))
trainData['poi'] = trainData['poi'].apply(lambda x: int(x))
testData['poi'] = testData['poi'].apply(lambda x: int(x))


# In[ ]:


## Evaluating the class distribution:
fig2, axes2 = plt.subplots(1,3,figsize=(15,3), sharex=False);
dataset['poi'].plot(kind='hist', ax=axes2[0], title='Total dataset');
trainData['poi'].plot(kind='hist', ax=axes2[1], title='Train subset');
testData['poi'].plot(kind='hist', ax=axes2[2], title='Test subset');


# <a id='sec3'></a>
# ## 3. Feature extraction
# **Note:** the initial approach was to split the dataset and to build out new features from the test data. However, since the 'poi_id.py' suggested by Udacity splits the data only after the feature engineering, we will then use the original dataset, splitting it up only before machine learning training.

# <a id='sec3.1'></a>
# ### 3.1 Verifying correlation among features
# Considering feature extraction, high correlated variables usually are useless for machine learning classification. In this case, it's better to use uncorrelated variables as features, in the way they are orthogonal to each other and so brings on different information aspects from data.  
# 
# To check which features are correlated or not we will then use a method from pandas library (http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.corr.html) and show it through a heatmap to make easier the visualization, as follow:

# In[ ]:


## Calculating the correlation among features by Pearson method
correlationDataframe = dataset[allFeatures].corr()

# Drawing a heatmap with the numeric values in each cell
fig1, ax = plt.subplots(figsize=(14,10))
fig1.subplots_adjust(top=.945)
plt.suptitle('Features correlation from the Enron POI dataset', fontsize=14, fontweight='bold')

cbar_kws = {'orientation':"vertical", 'pad':0.025, 'aspect':70}
sns.heatmap(correlationDataframe, annot=True, fmt='.2f', linewidths=.3, ax=ax, cbar_kws=cbar_kws);


# <a id='sec3.2'></a>
# ### 3.2 Applying PCA on correlated features to generate a new one
# Since the financial features are highly correlated, as seen above, we will now apply PCA to generate an only one new feature from them. 

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


## Listing the financial features
financialFeatures


# In[ ]:


## Defining only one resulting component:
pca = PCA(n_components=1)
pca.fit(dataset[financialFeatures])


# In[ ]:


pcaComponents = pca.fit_transform(dataset[financialFeatures])


# In[ ]:


dataset['financial'] = pcaComponents


# In[ ]:


sns.pairplot(dataset,hue='poi',vars=['salary','bonus'], diag_kind='kde');


# In[ ]:


sns.pairplot(dataset,hue='poi',vars=['salary','financial'], diag_kind='kde');


# <a id='sec3.3'></a>
# ### 3.3 Univariate feature selection

# In[ ]:


## Adding up the new 'financial' feature to the 'allFeatures' list:
allFeatures.append('financial')
financialFeatures.append('financial')


# In[ ]:


allFeatures


# The code below is based on the example available in http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-py, in which an univariate feature selection is applied. 

# In[ ]:


from sklearn.feature_selection import SelectPercentile, f_classif

selectorDataset = dataset[financialFeatures]
selectorLabel = dataset['poi']

# #############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 5% most significant features
selector = SelectPercentile(f_classif, percentile=5)
selector.fit(selectorDataset, selectorLabel)


# In[ ]:


## Plotting the features selection: 
X_indices = np.arange(selectorDataset.shape[-1])
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')


# In[ ]:


len(scores)


# In[ ]:


## Printing out the selected financial features: 
selectedFeatures = ['poi']  #'poi' must be the first one due to the evaluation methods defined by Udacity.
for ix, pval in enumerate(scores):
    print(financialFeatures[ix],': ',pval)
    if (pval >= 0.45):
        selectedFeatures.append(financialFeatures[ix])


# In[ ]:


selectedFeatures


# <a id='sec3.4'></a>
# ### 3.4. Alternative feature selection
# Since the PCA applied to the financial features creates a new variable ('financial') which contains the most important components of the others, one alternative is to append it to the behavioral features selected before, in what we call *strategicFeatures*. In the next sections we will evaluate the performance of applying machine learning in this two classes of features.

# In[ ]:


strategicFeatures = ['poi'] + behavioralFeatures + ['financial']


# In[ ]:


strategicFeatures


# <a id='sec3.5'></a>
# ### 3.5. Feature scaling
# Besides some of the machine learning algorithms chosen for this analysis (Naive Bayes and Adaboost) are invariant to feature scaling (Ref: https://stats.stackexchange.com/questions/244507/what-algorithms-need-feature-scaling-beside-from-svm), since we are also interested in using SVM it is important to apply feature scaling in our dataset. Furthermore, the "strategicFeatures" mixes two types of data, the financial data from PCA and the behavioral data from the number of emails sent.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(dataset[allFeatures])
dataset[allFeatures] = scaler.transform(dataset[allFeatures])


# <a id='sec4'></a>
# ## 4. Machine Learning

# In this section we select some classification algorithms to apply on the features we extracted from our dataset. After applying each of the selected machine learning algorithms -- naive Bayes; Adaboost with decision trees; and SVM -- it is important to evaluate its classification results. Considering we have an unbalanced dataset, i.e. the number of samples for each class are distinct, we cannot use *accuracy* for measuring its performance. In this case we choose *precision* and *recall* for performance measurements. 

# In[ ]:


## Converting back the pandas Dataframe to the dictionary structure, in order to use the Udacity evaluating code.
my_dataset = dataset.to_dict(orient='index')
features_list = selectedFeatures
#features_list = strategicFeatures

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[ ]:


## Splitting the data:
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[ ]:


## Defining an evaluation metric based on (http://scikit-learn.org/stable/modules/model_evaluation.html)
from sklearn.metrics import classification_report
def evaluateClassif(clf):
    classes=['Non-POI','POI']  ## Defining the classes labels
    predTrain = clf.predict(features_train)
    print('################### Training data ##################')
    print(classification_report(labels_train, predTrain, target_names=classes))
    
    predTest = clf.predict(features_test)
    print('################### Testing data ###################')
    print(classification_report(labels_test, predTest, target_names=classes))
    
    return


# In[ ]:


## Importing GridSearch algorithm for parameter selection:
from sklearn.model_selection import GridSearchCV


# <a id='sec4.1'></a>
# ### 4.1 Naive Bayes classifier

# In[ ]:


#%%## Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_parameters = {}


# In[ ]:


clf_nb = GridSearchCV(nb, nb_parameters)
clf_nb.fit(features_train, labels_train)


# In[ ]:


evaluateClassif(clf_nb)


# <a id='sec4.2'></a>
# ### 4.2 AdaBoost classifier

# In[ ]:


### Adaboost Classifier
### http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


## Defining the Adaboost parameters for GridSearch:
abc_parameters = {"learning_rate" : [0.5, 1., 2., 5., 10., 100.],
                 "n_estimators": [10,50,100,200,500,900,2000],
                 "algorithm": ['SAMME','SAMME.R']}

dtc = DecisionTreeClassifier(random_state = 42, max_features = "auto", max_depth = None)
abc = AdaBoostClassifier(base_estimator=dtc)

# run grid search
clf_adaboost = GridSearchCV(abc, param_grid=abc_parameters)


# In[ ]:


clf_adaboost.fit(features_train, labels_train)


# In[ ]:


evaluateClassif(clf_adaboost)


# <a id='sec4.3'></a>
# ### 4.3. SVM classifier

# In[ ]:


from sklearn import svm
svm_parameters = {'kernel':['linear','rbf','poly','sigmoid'], 
                  'C':[0.5,1.,5.,10.,50.,100.,1000.], 'gamma':['scale']}
svr = svm.SVC()


# In[ ]:


clf_svc = GridSearchCV(svr, svm_parameters);


# In[ ]:


clf_svc.fit(features_train, labels_train)


# In[ ]:


evaluateClassif(clf_svc)


# <a id='sec4.4'></a>
# ### 4.4. Conclusions

# Evaluating the output of the classifiers above, it's effort to choose **Naive Bayes Classifier**, since it had the best considered metrics (precision and recall) for POI in the testing data. It must be considered, though, that we could use more strategies to iterate the test and to certify this would be the best machine learning algorithm to be used. 

# In[ ]:




