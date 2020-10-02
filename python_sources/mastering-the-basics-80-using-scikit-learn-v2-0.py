#!/usr/bin/env python
# coding: utf-8

# # Mastering the basics: How to get ~80% using scikit-learn.
# 
# **Author:** tommyod, mattemagisk
# 
# > **Scored 0.79241 on Kaggle using logistic regression.**
# 
# **Abstract:** In this notebook, we will demonstrate how to get close to $80\%$ ROC AUC using the [scikit-learn](http://scikit-learn.org/stable/) library for Python. In contrast to some of the other kernels, the primary intention of this notebook is not to demonstrate new, novel methods, but rather to show that one can get far by observing the basics. Specifically, we will choose numerical, categorical and text features, build a scikit-learn pipeline, perform a hyperparameter search and present the results.
# 
# 
# # Table of contents
# 
# - <a href="#choosing">Choosing features</a>: Importing data and preparing features
# - <a href="#building">Building a scikit-learn pipeline</a>: How to build a data processing pipeline
# - <a href="#hyperparameter">Hyperparameter search</a>: Easy search for hyperparameters over the pipeline
# - <a href="#results">Results and references</a>: The results and some further reading

# # <a id="choosing">Choosing features</a>

# ## Importing, setting up packages and loading data

# ### Library imports and settings

# In[1]:


# Python library imports
import numpy as np # All numerical libraries in Python built on NumPy
import pandas as pd # Pandas provides DataFrames for data wrangling
import matplotlib.pyplot as plt # The de facto plotting library in Python
import itertools # For iterations
import string # For strings
import re # For regular expression (regex)
import os # Operating system functions

plt.style.use('Solarize_Light2') # Set a non-default visual aesthetic for plots
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns", 2**10) # View more columns than pandas default


# ### Importing the test and train data, concatenate together

# We start by importing the data as [pd.DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) objects.

# In[2]:


# Import the train and test data as pandas DataFrame instances
date_cols = ['project_submitted_datetime'] # Convert to datetime format automatically
train = pd.read_csv(os.path.join(r'../input', 'train.csv'), low_memory=False, parse_dates=date_cols)
test = pd.read_csv(os.path.join(r'../input', 'test.csv'), low_memory=False, parse_dates=date_cols)


# We want to apply a set of identical functions to both. The most efficient way is to make a new column in each data set indicating the source and concatenating the DataFrames with [pd.concat()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html).

# In[3]:


# Keep track of the original data source by adding a 'source' column
train['source'] = 'train'
test['source'] = 'test'
test_train = pd.concat((test, train))
print(f'The shape of the data is {test_train.shape}.') # Showing off Python f-strings


# In[4]:


# Checking for missing values
print("="*60)
print("Detecting NaN values in data:")
print("="*60)
print(test_train.isnull().sum(axis=0)[test_train.isnull().sum(axis=0) > 0])


# There are many missing third and fourth essays, as expected (see [competition page](https://www.kaggle.com/c/donorschoose-application-screening) for more info. We are also missing $78035$ target values. These are from the test data. Finally, there are $5$ missing values in the `teacher_prefix` column, fill it using [DataFrame.value_counts](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html), [DataFrame.idxmax](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.idxmax.html) (this is the [arg max function](https://en.wikipedia.org/wiki/Arg_max)) and [DataFrame.fillna](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html):

# In[5]:


most_common = test_train.teacher_prefix.value_counts().idxmax() # Compute argument maximizing value counts
test_train.teacher_prefix = test_train.teacher_prefix.fillna(most_common)


# ## Features selection

# Algorithms aren't very useful without good data. Going from a raw data set to a set of usable features for an algorithm is is process of [feature selection](https://en.wikipedia.org/wiki/Feature_selection). There are three types of available features in this competition:
# 
# - **Numerical**: E.g. the total price of the request
# - **Categorical**: E.g. the gender of the requester
# - **Text**: E.g. the written text of the second essay
# 
# We will build a data processing pipeline which heeds these types of variables.

# In[6]:


numerical_cols = []
dummy_categorical_cols = []
text_cols = []


# ### Features selection

# Since essays 3 and 4 were dropped, we will apply the following mapping to the essays:
# 
# - $\text{Essay}_1 := \text{Essay}_1 \oplus \text{Essay}_2$
# - $\text{Essay}_2 := \text{Essay}_3 \oplus \text{Essay}_4$
# 
# Where $\oplus$ denotes concatenation. This is easily accomplished using a mask in pandas, along with the clever [DataFrame.assign](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.assign.html) method for assigning new columns as part of a [pandas method chain](https://tomaugspurger.github.io/method-chaining). For more resources related to pandas, see [awesome-pandas](https://github.com/tommyod/awesome-pandas). Also note the use of `lambda` keyword. `lambda` is used to create [anonymous functions](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions), and is very useful.

# In[7]:


# Find the rows where essays 3 and 4 are not null
mask_four_essays = ~(test_train.project_essay_3.isnull() & test_train.project_essay_4.isnull())

# Assign them to columns 1 and 2 by concatenation
test_train[mask_four_essays] = (test_train[mask_four_essays]
                 .assign(project_essay_1 = lambda df: df.project_essay_1 + df.project_essay_2)
                 .assign(project_essay_2 = lambda df: df.project_essay_3 + df.project_essay_4))

# Drop columns related to essay 3 and 4
test_train = test_train.drop(columns=['project_essay_3', 'project_essay_4'])


# ### Importing data from `resources.csv`, join in 

# Let's import data from `resources.csv`:

# In[8]:


# Load resources
resources = pd.read_csv(os.path.join(r'../input', 'resources.csv'), low_memory=False)
print(f'The shape of the data is {resources.shape}.')


# In[9]:


# Checking for missing values
print("="*30)
print("Detecting NaN values in data:")
print("="*30)
print(resources.isnull().sum(axis=0)[resources.isnull().sum(axis=0) > 0])


# We see that 292 descriptions are missing text. Our approach is to simply fill these with the character 'X':

# In[10]:


# Fill NAs
resources = resources.fillna('X')


# In[11]:


# Previewing data
resources.head(3)


# Now it's time to show off [method chaining](https://tomaugspurger.github.io/method-chaining):

# In[12]:


def concatenate(series, sep=' '):
    return sep.join(series) # Preferred to lambda, since this function has a name

# Create a lot of possible numerical features, each starting with 'p_'
resource_stats = (resources
.assign(p_desc_len = lambda df: df.description.str.len()) # Length of description text
.assign(p_total_price = lambda df: df.quantity * df.price) # Total price per item
.groupby('id') # Grouping by teacher ID and aggregating the following columns (note we pass function as a list): 
.agg({'description': [pd.Series.nunique, concatenate], # Number of unique items asked for
'quantity': [np.sum], # Total number of items asked for
'price': [np.sum, np.mean], # Prices per item added and averaged (quantity not included)
'p_desc_len': [np.mean, np.min, np.max], # Average description length
'p_total_price': [np.mean, np.min, np.max]})
)


# Again, we check for missing values:

# In[13]:


# Checking for missing values
print("="*30)
print("Detecting NaN values in data:")
print("="*30)
print(resource_stats.isnull().sum(axis=0)[resource_stats.isnull().sum(axis=0) > 0])


# Note that the new DataFrame has a MultiIndex. In the following we collapse it to a flat index:

# In[14]:


# Collaps to flat index
resource_stats.columns = ['_'.join([col, func]) for col, func in resource_stats.columns.values]
numerical_cols += list(resource_stats.columns.values)


# Let's join in information from `resources.csv`. We'll use [DataFrame.merge](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html) (which corresponds to a [SQL JOIN](https://blog.codinghorror.com/a-visual-explanation-of-sql-joins/) operation).

# In[15]:


# Merge the resources statistics into the test and train sets
test_train = test_train.merge(resource_stats, how='left', left_on='id', right_index=True)
test_train.sample(1).T.tail(6)


# ## Encode categorical features

# At this point we have some numerical features. Now it's time to look at the categorical ones. We'll try adding the following:
# 
# - The **month of the request**, added using the [pandas.dt accessor](https://pandas.pydata.org/pandas-docs/stable/basics.html#dt-accessor). Pandas has some pretty powerful datetime (`dt`) functionality. Check out the [documentation](https://pandas.pydata.org/pandas-docs/stable/timeseries.html) for details.
# - The **time of day**. Using [np.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html) for a blazingly fast, vectorized `IF-THEN-ELSE` condition, we will find out if the application was sent during the **morning hours** or not.
# - The **gender of the applicant** and the **school state**, using [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) which converts a categorical variable to a dummy variable representation.

# In[16]:


# Get the month from the datetime, convert it to a string, and add it as a new column
test_train['month'] = test_train.project_submitted_datetime.dt.month.apply(str)

# Does submitting during the morning hours help?
test_train['daytime'] = pd.Series(np.where( 
    ((7 <= test_train.project_submitted_datetime.dt.hour) & 
     (test_train.project_submitted_datetime.dt.hour <= 10)), 1, 0)).apply(str)


# In[17]:


# Simple dummy variables, i.e. every entry has one value
dummy_colnames = ['teacher_prefix', 'month', 'school_state', 'daytime']
dummies = pd.get_dummies(test_train.loc[:, dummy_colnames])
dummy_categorical_cols += dummies.columns.tolist()

# Concatenate along the columns
test_train = pd.concat((test_train, dummies), axis=1)


# ### Create categorical features from `project_subject_categories` and `project_subject_subcategories`

# The following columns are special:
# 
# - `project_subject_categories`
# - `project_subject_subcategories`
# 
# Each entry may contain several categories - clearly a violation of the priniples of [tidy data](https://en.wikipedia.org/wiki/Tidy_data) if you consider a single category as a variable. There are $51$ different combinations of categories, and $416$ different combinations of subcategories. In reality, there are way less categories to choose from: $9$ categories and $30$ subcategories in total.
# 
# In the following we will define a function that creates a list of unique categories and sub-categories. Then we will create dummy variables from the resulting variables:    

# In[18]:


# The following libray is for representing (printing) long strings and lists
import reprlib

def set_of_categories(col_name):
    """Retrieve a set of category names from a column"""
    list_train = test_train[col_name].tolist()
    list_test = test_train[col_name].tolist()
    return set(', '.join(list_train + list_test).split(', '))

unique_categories = set_of_categories('project_subject_categories')
unique_subcategories = set_of_categories('project_subject_subcategories')
unique_cats_total = list(unique_categories.union(unique_subcategories))
dummy_categorical_cols += unique_cats_total
print('Categories:', reprlib.repr(unique_cats_total))


# Let's create the dummy encoding:

# In[19]:


project_cat_colnames = ['project_subject_categories', 'project_subject_subcategories']

df_cats = test_train.loc[:, project_cat_colnames]

# Create a new column for each category: put 1 if it's mentioned, 0 if not
for category in unique_categories:
    df_cats[category] = np.where(df_cats.project_subject_categories.str.contains(category), 1, 0)
for category in unique_subcategories:
    df_cats[category] = np.where(df_cats.project_subject_subcategories.str.contains(category), 1, 0)
    
df_cats = df_cats.drop(columns=project_cat_colnames)
df_cats.head(1).T.head(5)


# Again we'll use [pandas.concat](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html) to concatenate the results back into the DataFrame.

# In[20]:


test_train = pd.concat((test_train, df_cats), axis=1)
print(f'The dataset now has ~{len(test_train.columns)} features.')
test_train.head(1)


# ## Investigate text features

# In[21]:


# Extracting text columns
text_cols += ['project_essay_1', 'project_essay_2', 'project_resource_summary', 'description_concatenate']


# ### Does polarity and subjectivity matter?

# The TextBlog library includes functions for [sentiment analysis](http://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis), namely scoring **polarity** and **subjectivity** from a text. We'll investigate if these could be a useful features:

# In[22]:


from textblob import TextBlob

df_subset = test_train[test_train.source == 'train']

# Create a plot
plt.figure(figsize=(14, 10))

plt_num = 1
for column in text_cols:
    
    # Get data corresponding to the columns which we will use
    data = df_subset.loc[:, ('project_is_approved', column)]
        
    for feature in ['polarity', 'subjectivity']:
    
        # Create a new subplot, set the title
        plt.subplot(4, 2, plt_num)
        plt.title(f'{column} - {feature.capitalize()}', fontsize=12)
        plt_num += 1

        # Function to get features from text using TextBlob
        feature_from_txt = lambda x : getattr(TextBlob(x).sentiment, feature)

        # Sample some data, apply the feature extraction function
        approved_mask = (data.project_is_approved == 1)
        approved = data[approved_mask].sample(1000).assign(feat=lambda df: df[column].apply(feature_from_txt))
        not_approved = data[~approved_mask].sample(1000).assign(feat=lambda df: df[column].apply(feature_from_txt))
        
        # Plot the subplot
        bandwidth = 0.225
        ax = approved.feat.plot.kde(bw_method=bandwidth, label='Approved')
        ax = not_approved.feat.plot.kde(ax=ax, bw_method=bandwidth, label='Not approved')
        plt.xlim([-.5, 1])
        plt.legend(loc='best')
        
# Show the full figure
plt.tight_layout()
plt.show()


# It seems like the following could be reasonable predictors of whether a request is approved or not:
# 
# - Subjectivity of `description`
# - Polarity of `description`
# - Subjectivity of `project_resource_summary`
# - Polarity of `project_resource_summary`
# - Polarity of `project_essay_1`
# - Polarity of `project_essay_2`
# 
# **WARNING:** Slow code ahead. Applying the polarity and subjectivity scoring takes a while. Skip ahead if you do not wish to run slow cells.

# --------------------

# In[23]:


get_ipython().run_cell_magic('time', '', "\nsubj = lambda x: TextBlob(x).sentiment.subjectivity\npolar = lambda x: TextBlob(x).sentiment.polarity\n\ntest_train['description_subjectivity'] = test_train['description_concatenate'].apply(subj)\ntest_train['description_polarity'] = test_train['description_concatenate'].apply(polar)\n\ntest_train['project_resource_summary_subjectivity'] = test_train['project_resource_summary'].apply(subj)\ntest_train['project_resource_summary_polarity'] = test_train['project_resource_summary'].apply(polar)\n\ntest_train['project_essay_2_polarity'] = test_train['project_essay_2'].apply(polar)\ntest_train['project_essay_1_polarity'] = test_train['project_essay_1'].apply(polar)")


# There are a few more obvious numerical features we can extract from the text data.
# We'll extract:
# 
# - The number of unique words.
# - The total number of words.
# - Their ratio, defined as the *vocabularity* of a request.
# 
# To accomplish this in a fast way, we'll make use of the scikit-learn [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), which converts a collection of text documents into a (sparse) matrix of token counts. The matrix will be very sparse, so scikit-learn will return a [scipy Compressed Sparse Row matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html). I mistakenly called [.todense()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.todense.html#scipy.sparse.csr_matrix.todense) on it, and it crashed my computer. 
# 
# First we will take a quick look at `text_cols` to remind ourselves of which features we're iterating over:

# In[24]:


text_cols += ['project_title']
print(text_cols)


# In[25]:


get_ipython().run_cell_magic('time', '', "\n# This code is a bit slow: approximately ~50 seconds on my computer\nfrom sklearn.feature_extraction.text import CountVectorizer\nvectorizer = CountVectorizer()\n\nfor text_col in text_cols: # Title, essays, summary, title\n    \n    # Get a sparse SciPy matrix of words counts\n    X = vectorizer.fit_transform(test_train[text_col].values)\n    \n    col_new_name = text_col.replace('project_', '')\n    \n    # Compute some basic statistics and add to dataset\n    unique_words = (X > 0).sum(axis=1) # Sum of words appearing more than zero times\n    num_words = (X).sum(axis=1) # Sum of occurences of words\n    test_train[col_new_name + '_unique_words'] = unique_words\n    test_train[col_new_name + '_num_words'] = num_words\n    test_train[col_new_name + '_vocab'] = np.exp(unique_words / (num_words + 10e-10))")


# ### Save and load the data 
# 
# The code above is slow. We do not wish to run it every time we run the notebook, so
# we'll store the data, then re-load it from the hard drive.

# In[26]:


from csv import QUOTE_ALL

# We're going to quote the fields using ", so remove it from the text just in case
for text_col in text_cols:
    test_train[text_col] = test_train[text_col].str.replace('"', ' ')
   
# Save the data
# test_train.to_csv('preprocessed_test_train.csv', quoting=QUOTE_ALL)


# In[27]:


# Load the data from disk
# test_train = pd.read_csv('preprocessed_test_train.csv', index_col=0, quoting=QUOTE_ALL)


# In[28]:


numeric_cols = [
 'teacher_number_of_previously_posted_projects',
 'description_nunique',
 'quantity_sum',
 'price_sum',
 'price_mean',
 'p_desc_len_mean',
 'p_desc_len_amin',
 'p_desc_len_amax',
 'p_total_price_mean',
 'p_total_price_amin',
 'p_total_price_amax',
 'project_resource_summary_subjectivity',
 'project_resource_summary_polarity',
 'project_essay_2_polarity',
 'project_essay_1_polarity',
 'essay_1_unique_words',
 'essay_1_num_words',
 'essay_1_vocab',
 'essay_2_unique_words',
 'essay_2_num_words',
 'essay_2_vocab',
 'resource_summary_unique_words',
 'resource_summary_num_words',
 'resource_summary_vocab',
 'description_concatenate_unique_words',
 'description_concatenate_num_words',
 'description_concatenate_vocab',
 'title_unique_words',
 'title_num_words',
 'title_vocab']


# # <a id="building">Building a scikit-learn pipeline</a>
# 
# ## General theory on scikit-learn estimators and pipelines
# 
# We have three types of data: 
# 
# 1. **numerical**, 
# 2. **categorical** (now encoded using dummy variables), and 
# 3. **text**.
# 
# The next step is to build a scikit-learn [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to handle the flow of data.
# There are at least three good reasons for building pipelines, they include:
# 
# 1. **Modular components**: Individual estimators may be switched out for other components easily.
# 2. **Object oriented**: Clean, readable code.
# 3. **Efficient hyperparameter search**: Hyperparameter searches for every estimator in the pipeline may be automatized.
# 
# The genius of scikit-learn is it's API, described in detail in the paper [API design for machine learning software](https://arxiv.org/abs/1309.0238) on [arXiv.org](https://arxiv.org/). The three important objects are:
# 
# 1. **Estimators**: Subclasses [BaseEstimator](http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html), must implement a **fit** method.
# 2. **Transformers**: Subclasses [BaseEstimator](http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html) and [TransformerMixin](http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html), must implement a **fit** and **transform** method.
# 3. **Predictors**: Subclasses [BaseEstimator](http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html) and [RegressorMixin](http://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html) (or another predictor-mixin), must implement a **fit** and **predict** method.
# 
# Every transformer is an estimator, and every predictor is an estimator. The converse is not true.
# A [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) is simply a composition:
# 
# $$\text{Pipeline}(D) = (\text{Estimator} \circ \text{Transformer}_N \circ \dots \circ \text{Transformer}_2 \circ \text{Transformer}_1 )(D)$$
# 
# Where $\circ$ denotes composition of functions and $D$ denotes data. 
# 
# This may all be a bit abstract, but hopefully the following image may clarify things (it is just a visual representation of the pipeline described above):
# 
# ![pipe.png](https://raw.githubusercontent.com/skaug/Kaggle-Club/master/DonorsChoose/python/pipe.png?token=AJm_qDboqiFisXoZ4e04f-L6fTA3fIsmks5a26vvwA%3D%3D)
# 
# There are two types of pipelines: pipelines where the final estimator is a **transformer** (making the entire pipeline a transformer), and pipelines where the final estimator is a **predictor** (making the entire pipeline a predictor).
# 
# $$\text{Pipeline}_\text{trans}(D) = (\text{Transformer}_{N+1} \circ \text{Transformer}_N \circ \dots \circ \text{Transformer}_2 \circ \text{Transformer}_1 )(D)$$
# 
# $$\text{Pipeline}_\text{pred}(D) = (\text{Predictor} \circ \text{Transformer}_N \circ \dots \circ \text{Transformer}_2 \circ \text{Transformer}_1 )(D)$$
# 
# Pipelines are sequential. Their complement is the [FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html), which combines data from pipelines by concatenation of the columns.
# 
# 
# ## Our pipeline - an overview
# 
# We're going to build the following pipeline.
# 
# ![full_pipe.png](https://raw.githubusercontent.com/skaug/Kaggle-Club/master/DonorsChoose/python/full_pipe.png?token=AJm_qBC8s6OAl_bVWB3lgxfZ6AlA0l7Dks5a26uowA%3D%3D)

# We'll import most of the estimators: 
#    1. [RobustScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) (computes robust statistics by scaling the data according to some given quantile range),
#    2. [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) (a technique to re-weight words so that more meaningful words are given more weight), and 
#    3. [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (used when we want to model a the probability of a binary outcome).<br>
#     
# We'll create some custom transformers: 
#    1. **ColSplitter** to split the columns of the dataset to send data to parallel paths in the main pipeline, 
#    2. **LogTransform** which computes $f(x;\alpha) = \alpha \log(1 + x) + (1 - \alpha) x$ on the numerical data, where we can search for optimal $\alpha$ using hyperparameter search, and 
#    3. **ParallelPipe**, which will let us apply Tfidf to each text feature individually.
# 
# We'll make liberal use of [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) objects to control data flow.

# In[29]:


# Imports for the pipeline
from tempfile import mkdtemp
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import RobustScaler


# ## Custom classes for the pipeline

# In[30]:


class ColSplitter(BaseEstimator, TransformerMixin):
    """Estimator to split the columns of a pandas dataframe."""
    
    def __init__(self, cols, ravel=False):
        self.cols = cols
        self.ravel = ravel # If it's a text features, we ravel for TF-IDF

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return (x.loc[:, self.cols].values.ravel() if self.ravel
                else x.loc[:, self.cols].values)
    
class LogTransform(BaseEstimator, TransformerMixin):
    """Take linear combination of f(x) = x and g(x) = ln(x)"""
    
    def __init__(self, alpha):
        """alpha = 1 -> log
        alpha = 0 -> linear"""
        self.alpha = alpha
        
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return self.alpha * np.log(np.log(x + 2.001) + 1.001) + (1 - self.alpha) * x
    
class ParallelPipe(BaseEstimator, TransformerMixin):
    """Put similar pipes in parallel. This has two effects:
    (1) When transforming, the final result is concatenated (Feature Union)
    (2) Setting hyperparameters on this pipe will set it on every sub-pipe."""
    
    def __init__(self, pipes, *args, **kwargs):
        self.pipes = pipes
        
    def fit(self, x, y=None):
        [p.fit(x) for p in self.pipes]
        return self

    def transform(self, x):
        try:
            return hstack([p.transform(x) for p in self.pipes])
        except:
            return np.concatenate(tuple([p.transform(x) for p in self.pipes]), axis=1)
    
    def _get_param_names(self, *args, **kwargs):
        return ['pipes']
    
    def set_params(self, *args, **kwargs):
        [p.set_params(*args, **kwargs) for p in self.pipes]
        return None
    


# ## Building the pipeline

# In[40]:


# --------------------------------------------------
# ----- (1) Numerical pipeline ---------------------
# --------------------------------------------------
num_colsplitter = ColSplitter(cols=numeric_cols)
logtransform = LogTransform(alpha=1)
scaler = RobustScaler(quantile_range=(5.0, 95.0))
numerical_pipe = Pipeline([('num_colsplitter', num_colsplitter), 
                           ('logtransform', logtransform),
                           ('scaler', scaler)])

# --------------------------------------------------
# ----- (2) Categorical pipeline -------------------
# --------------------------------------------------
dummy_colsplitter = ColSplitter(cols=dummy_categorical_cols)
categorical_pipe = Pipeline([('dummy_colsplitter', dummy_colsplitter)])

# --------------------------------------------------
# ----- (3) Text pipeline --------------------------
# --------------------------------------------------
text_subpipes = []
for text_col in text_cols:
    text_colsplitter = ColSplitter(cols=text_col, ravel=True)
    
    tf_idf = TfidfVectorizer(sublinear_tf=False, norm='l2', stop_words=None, 
                             ngram_range=(1, 1), max_features=None)
    
    text_col_pipe = Pipeline([('text_colsplitter', text_colsplitter),
                     ('tf_idf', tf_idf)])
    text_subpipes.append(text_col_pipe)

text_pipe = ParallelPipe(text_subpipes)

# --------------------------------------------------
# ----- (4) Final pipeline - Logistic regression ---
# --------------------------------------------------
#cachedir = mkdtemp() # Creates a temporary directory
estimator = LogisticRegression(penalty="l2", C=0.21428)

pipeline_logreg = Pipeline([('union', FeatureUnion(transformer_list=[
    ('numerical_pipe', numerical_pipe),
    ('categorical_pipe', categorical_pipe),
    ('text_pipe', text_pipe)
])), 
                     ('estimator', estimator)])


# Note the penalty term in `estimator` in the final Logistic Regression. [Regularization](https://en.wikipedia.org/wiki/Regularization_%28mathematics%29) is very important in machine learning. Here we use $L_2$-regularization, also known as **weight decay** or **ridge regression**.

# # <a id="hyperparameter">Hyperparameter search </a>

# In[41]:


# Testing that all numeric values are finite
assert np.all(np.isfinite(test_train[numeric_cols].values))


# Time to run hyperparameter search using [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) on the entire pipeline. 
# 
# **WARNING:** The code below is not representative of what we tried. We experimented with several hyperparameters and several estimators which we liberally removed and added to the full pipeline.

# ## Prepare grid search

# The [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) (Receiver operating characteristic) curve is a useful model validation tool when dealing with binary classifiers. We will use the [area under the curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) (AUC) of the ROC curve to determine the best hyperparameters:

# In[45]:


from sklearn.model_selection import GridSearchCV

# Dictionary with parameters names to try during search
# We tried a lot of parameters, you may uncomment the code an experiment
param_grid = {"estimator__C": np.linspace(0.24285-0.1, 0.24285+0.1, num=6)
             # "union__numerical_pipe__logtransform__alpha": [0.8, 1],
             # "union__text_pipe__tf_idf__stop_words": [None, 'english']
             }

# run randomized search
grid_search = GridSearchCV(pipeline_logreg, param_grid=param_grid,
                                    scoring='roc_auc',
                                    n_jobs=1,
                                    verbose=1,
                                    cv=3)


# ## Run grid search
# 
# We'll grab an equal amount of data: $50\%$ approved and $50\%$ not approved.

# In[46]:


# Grab 2 subsets of the data
n = 25000
subset_A = test_train.loc[lambda df: (df.project_is_approved == 1)].sample(n)
subset_B = test_train.loc[lambda df: (df.project_is_approved == 0)].sample(n)
test_train_subset = pd.concat((subset_A, subset_B))
test_X = test_train_subset
test_y = test_X.project_is_approved


# In[47]:


import warnings
perform_grid_search = True

if perform_grid_search:
    with warnings.catch_warnings():
        # UserWarning: Persisting input arguments took 0.70s to run.
        warnings.simplefilter("ignore", category=UserWarning)
        grid_search.fit(test_X, test_y.values.ravel())
    best_estimator = grid_search.best_estimator_
    print('Best score:', grid_search.best_score_)
    print('Best params:\n', grid_search.best_params_)
else:
    # If we do not run grid search, we use the pipeline as initialized
    best_estimator = pipeline_logreg


# ## Fit on 100% of the data and predict on test data

# In[48]:


# Grab 100% of the train data, run fitting using the best estimator
full_train = test_train.loc[lambda df: df.source == 'train']
y_pred = best_estimator.fit(full_train, full_train.project_is_approved.ravel())


# In[52]:


# Predict on test data
test = test_train[test_train.source == 'test']
test.loc[:, ('project_is_approved', )] = best_estimator.predict_proba(test)[:, 1]


# Finally, we save the predictions to a .csv-file and submit it to Kaggle:

# In[53]:


test[['id','project_is_approved']].shape


# In[54]:


test[['id','project_is_approved']].to_csv('submission.csv', index=False)


# # <a id="results">Results and references</a>
# 
# ## Results and further work
# 
# In the end, this kernel (notebook) achieved a ROC AUC of $\sim 80\%$ using relatively well known, simple techniques.
# When we drop parts of the pipeline (by commenting out a line of code), we saw that $\geq 70\%$ could be achieved using only simple numerical features.
# Although this kernel is still a few percent from the highest scoring submissions, it is our hope that it can be intructional for people wanting to learn about Python, pandas and scikit-learn.
# If time allows, it would be interesting to experiment with other models than logistic regression. To do so efficiently, we might have to reduce the number of features from the text pipeline.
# 
# ## References
# 
# - [awesome-pandas](https://github.com/tommyod/awesome-pandas) - A collection of resources for pandas (Python) and related subjects.
# - [API design for machine learning software: experiences from the scikit-learn project](https://arxiv.org/abs/1309.0238) - Paper describing the scikit-learn API.
# - [Official pandas documentation](https://pandas.pydata.org/pandas-docs/stable/)
# - [Official scikit-learn documentation](http://scikit-learn.org/stable/)
# 
# **Question, comment or feedback? Please leave a comment below!**
