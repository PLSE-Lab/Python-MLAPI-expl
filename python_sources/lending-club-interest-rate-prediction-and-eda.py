#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# <a id="introduction**"></a>
# ***
# 
# Hello and welcome to a simple exploration and model of the Lending Club Loan Dataset. 
# 
# Today we will first be performing exploratory data analysis and visualizations including but not limited to
# * Violin plots
# * Box Plots
# * Correlation Heatmaps
# * Etc.
# 
# We will then build different models using Scikit-Learn, once we have made the basic models from RandomForests to Support Vector Machines, we will attempt to make our models stronger using
# * Ensemble Learning Techniques
# * Boosting Techniques
# 
# The notebook itself is divided into 3 parts
# 1. [Introduction and basic EDA]
# 2. [Data Visualization]
# 3. [Data Processing]
# 4. [Data Modelling]
# 5. [Model Assesment]

# In[ ]:


import time
start_time = time.time()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Notice there is a new datafile I have uploaded that we will use in this notebook


# Before we start let us make a utility class to help time the various steps of the program

# In[ ]:


class Timing:
    """
    Utility class to time the notebook while running. 
    """
    def __init__(self, start_time):
        self.start_time = start_time
        self.counter = 0

    def timer(self, message=None):
        """
        Timing function that returns the time taken for this step since the starting time. Message is optional otherwise we use a counter. 
        """
        if message:
            print(f"{message} at {time.time()-self.start_time}")
        else:
            print(f"{self.counter} at {time.time()-self.start_time}")
            self.counter += 1
        return
    
timing = Timing(start_time)


# Now we create a loading data function. This specifically has the ability to load only the columns that are mentioned in the datafile time_of_issue.csv. The data in this file is nothing more than the names of the variables/ information which the bank has access to at the time of actually issuing the loan

# In[ ]:


def load_data(number_of_rows:int =None, purpose=None)->pd.DataFrame:
    """
    Returns a pandas DataFrame with the loan data inside
    number_of_rows: Controls the number of rows read in, default and maximum is 22,60,668 rows
    restriction: Restricts the columns read in to correct for information you should not have depending on the task at hand
        "time_of_issue": Returns only the data that the lender has access to during the issuing of the loan
    """
    root = "../input/lending-club-loan-data"
    use_cols= None
    if purpose not in [None, 'time_of_issue']:
        raise ValueError(f"Invalid Purpose {purpose}")
    if purpose:
        col_root = "../input/columns-available-at-time-of-loan"
        columnframe = pd.read_csv(os.path.join(col_root, purpose+".csv"))
        illegals = ['sec_app_fico_range_low ', 'sec_app_inq_last_6mths ', 'sec_app_earliest_cr_line ', 'revol_bal_joint ', 'sec_app_mths_since_last_major_derog ', 'sec_app_revol_util ', 'sec_app_collections_12_mths_ex_med ', 'sec_app_open_acc ', 'fico_range_low', 'sec_app_fico_range_high ', 'verified_status_joint', 'last_fico_range_low', 'sec_app_chargeoff_within_12_mths ', 'fico_range_high', 'total_rev_hi_lim \xa0', 'sec_app_mort_acc ', 'sec_app_num_rev_accts ', 'last_fico_range_high']
        use_cols = [x for x in list(columnframe['name']) if x not in illegals]



    path = os.path.join(root, "loan.csv")

    maximum_rows = 2260668
    if not number_of_rows:
        return pd.read_csv(path, low_memory=False, usecols=use_cols)
    else:
        if number_of_rows > maximum_rows or number_of_rows < 1:
            raise ValueError(f"Number of Rows Must be a Number between 1 and {data.shape[0]}")
        else:
            return pd.read_csv(path, low_memory=False, nrows=number_of_rows, usecols=use_cols)


# In[ ]:


data = load_data(number_of_rows=None, purpose="time_of_issue")


# ## Data Exploration
# <a id="exploration**"></a>
# ***
# 
# Let us start by creating a function to look at the data

# In[ ]:


def investigate(data)->None:
    print(data.shape)
    print(data.info())
    print(data.describe())


# In[ ]:


investigate(data)


# Looking at the columns and their details then we can segment them into three types

# In[ ]:


def type_list_generator(data, separated=False):
    """
    Prints out 3 list to store which columns are of which type.
    Interest rate can be in the list or not depending on the seperated variable
    """
    numericals = ['loan_amnt','funded_amnt','funded_amnt_inv', 'annual_inc','mort_acc','emp_length', 'int_rate']
    if separated:
        numericals.pop()
    strings = ['issue_d', 'zip_code']
    categoricals = [x for x in data.columns if x not in numericals and x not in strings] # ['term', 'grade', 'sub_grade', 'emp_title', 'home_ownership', 'verification_status', 'purpose', 'title', 'addr_state', 'initial_list_status', 'application_type', 'disbursement_method']
    return numericals, strings, categoricals


# In[ ]:


numericals, strings, categoricals = type_list_generator(data)


# From an initial look at the data it seems like some columns are entirely nan columns (e.g. desc, there at least 2 such columns). We should probably drop these columns which have too many NaNs entirely as opposed to dropping rows

# In[ ]:


def drop_nan_columns(data, ratio=1.0)->pd.DataFrame:
    """
    The ratio parameter (0.0<=ratio<1.0) lets you drop columns which has 'ratio'% of nans. (i.e if ratio is 0.8 then all columns with 80% or more entries being nan get dropped)
    Returns a new dataframe
    """
    col_list = []
    na_df = data.isna()
    total_size = na_df.shape[0]
    for col in na_df:
        a = na_df[col].value_counts()
        if False not in a.keys():
            col_list.append(col)
        elif True not in a.keys():
            pass
        else:
            if a[True]/total_size >= ratio:
                col_list.append(col)
    print(f"{len(col_list)} columns dropped- {col_list}")
    return data.drop(col_list, axis=1)


# In[ ]:


data = drop_nan_columns(data, ratio=0.5)


# Now we've taken out the really useless columns, let's check the other ones so that we get a sense of how many NaN entries the rest of our data has

# In[ ]:


def investigate_nan_columns(data)->None:
    """
    Prints an analysis of the nans in the dataframe
    """
    col_dict = {}
    na_df = data.isna()
    total_size = na_df.shape[0]
    for col in na_df:
        a = na_df[col].value_counts()
        if False not in a.keys():
            col_dict[col] = 1.0
        elif True not in a.keys():
            pass
        else:
            col_dict[col] =  a[True]/total_size
    print(f"{col_dict}")
    return


# In[ ]:


investigate_nan_columns(data)


# This function us that employment title has over 7% NaN values, while length has less than 7%. The others are nearly negligible Let us look at the employment title to get a sense of the best strategy for dealing with these values.

# In[ ]:


data['emp_title'].value_counts()


# It seems a bit suspicious that there is no employment title for people without a job, very possibly these are what the NaN values represent. To confirm this let us create a list of titles that could feasibly represent the unemployed and see if any of them appear in the column

# In[ ]:


unemployed = ['unemployed', 'none', 'Unemployed', 'other', 'Other']
for item in unemployed:
    if item in data['emp_title']:
        print("Found It at ", item)


# It seems for employment title NaNs are likely all unemployed (it is highly unlikely that no one who recieved loans was unemployed.). 
# * For length, mortgage account, annual income and zip code we will use mode filling.
# * For title we will cast it to other.

# In[ ]:


def handle_nans(data)->None:
    """
    Handle the nans induvidually per column
    emp_title: make Nan -> Unemployed
    emp_length: make Nan - > 10+ years this is both mode filling and value filing
    title: make Nan -> Other
    """
    data['emp_title'] = data['emp_title'].fillna("Unemployed")
    data['title'] = data['title'].fillna('Other')
    mode_cols = ['emp_length', 'annual_inc', 'mort_acc', 'zip_code']
    for col in mode_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    return
handle_nans(data)


# We can now confirm that there are no NaN values in our dataset

# In[ ]:


any(data.isna().any()) # True iff there some NaN values anywhere in the dataset


# Now we can look at the data again and actually understand it

# In[ ]:


investigate(data)


# Looking at the datatypes it's easy to tell a few things need to be done regarding typing:-
# * There are a lot of categorical columns (e.g. grade) but right now these are only being read as object or strings, we must convert them to categorical variables.
# * We can also safely convert employment length to numbers so that the model can use it as a numerical column.
# * We also need to convert date to a datetime datatype to best use it

# In[ ]:


def handle_types(data, numericals, strings, categoricals):
    def helper_emp_length(x):
        if x == "10+ years": return 10
        elif x == "2 years": return 2
        elif x == "< 1 year": return 0
        elif x == "3 years": return 3
        elif x == "1 year": return 1
        elif x == "4 years": return 4
        elif x == "5 years": return 5
        elif x == "6 years": return 6
        elif x == "7 years": return 7
        elif x == "8 years": return 8
        elif x == "9 years": return 9
        else:
            return 10
    data['emp_length'] = data['emp_length'].apply(helper_emp_length)

    for category in categoricals:
        try:
            data[category] = data[category].astype('category')
        except:
            pass
    data['issue_d'] = data['issue_d'].astype('datetime64')
    return


# In[ ]:


handle_types(data, numericals, strings, categoricals)


# And that's it! We have finished an extremely basic cleaning of the dataset. 
# 

# ## Data Visualization
# <a id="visualization**"></a>
# ***
# We can now start Exploratory Data Analysis to find deeper patterns in the data.

# **Correlation Heatmap**
# 
# * Correlation Heatmaps are a great way to spot linear relations between numerical columns in your dataset.
# * The basic theory is that you use an inbuilt pandas function *corr* to calculate each variables correlation to every other variable
# * Plotting this resulting correlation matrix in a heatmap gives you a sense of which features are correlated to the target variables, hinting at the features you should select or are more important in the model you will make.

# In[ ]:


def correlation_heatmap(data):
    corrmat = data.corr()
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.title("Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.show()
    timing.timer("Heatmap")
    return

correlation_heatmap(data)


# From this plot we can see a few things
# * Clearly there is a huge correlation (nearly one to one) with the loan_amnt (Which is the amount requested by the borrower), and the funded amounts (amount funded by investors). This suggests that we probably want to merge these columns as they add dimensionality but do not provide that much extra information
# * From looking at the variables related to interest rates the first observation is that some variables like mortgage account balance and (surprisingly) annual income seem to have nearly no correlation
# * In general the most correlated variable seems to be employment length, we could plot these two variables against each other to get a clearer sense of their relationship
# 
# Unfortunately overall it seems the numerical variables are not the most correlated, either there is a non-linear relationship in the data or our categorical features are where the bulk of our useful features will be
# 

# **Distribution Plot**
# 
# * Distribution Plots are very similar to histograms and essentially show how data is distributed across its different values. 
# * They are extremely useful in finding skews in the data. Most models perform best when the data they deal with is normally distributed (especially linear models). So if we find a skew we may want to apply a skew solution function the variable in order to make it resemble a normal distribution
# 

# In[ ]:


def distplot(data):
    """
    Reveals a positive skew
    """
    from scipy.stats import norm
    sns.distplot(data['int_rate'], fit=norm)
    plt.title("Distribution and Skew of Interest Rate")
    plt.xlabel("Interest Rate in %")
    plt.ylabel("Occurance in %")
    plt.show()
    timing.timer("Skew with distplot")
    return

distplot(data)


# The extended tail forward gives a clear sign of a positive skew in our interest rate. This means that there are much more lower values than there are high values.
# Possible solutions we could apply include the square root and log functions

# **Boxplots**
# 
# * Boxplots are an extremely useful way to plot outliers, while also seeing how numerical data varies across different categories. 
# 
# 

# In[ ]:


def boxplot(data):
    """
Creates 4 boxplots
            
    """
    fig, axes = plt.subplots(2,2) # create figure and axes
    col_list = ['annual_inc', 'loan_amnt', 'int_rate', 'emp_length']
    by_dict = {0: 'home_ownership', 1:"disbursement_method", 2:"verification_status", 3:"grade"}

    for i,el in enumerate(col_list):
        a = data.boxplot(el, by=by_dict[i], ax=axes.flatten()[i])

    #fig.delaxes(axes[1,1]) # remove empty subplot
    plt.tight_layout()
    plt.title("Various Boxplots")
    plt.show()
    timing.timer("Boxplot")
    return

boxplot(data)


# The insights we can take from each plot:- 
# * (0,0) - This graph tells us nothing about the relation to interest rates, but gives us interesting insights on the economy from which the data was extracted, namely it is likely not a savings based economy. You can tell this by looking at how people who own their houses are not that much wealthier than those who have it on a mortgage. This implies that even when induviudals have enough income to perhaps save an eventually buy a house or a buy a lower grade house they could afford they are opting to take a loan and commit to this expenditure. It is also an extremely unequal economy, with the outliers being so high it makes the averages look like they are at the zero mark.
# * (0,1) - This graph tells us the intutive idea that cash loans on average are of a smaller sum than DirectPay loans, presumably for security reasons. The suprising observation is the lack of any significant outliers, implying that this lending institution is a relatively safe one which caps the loans it gives, meaning there isn't a single loan so high that it would count as a significant outlier.
# * (1,0) - This graph suggests that verification status does seem to have a relationship with interest rate. The average steadily increases the more verfified the borrower is.
# * (1,1) - This graph suggest there is no relationship between the length of employment and the Grade of the loan

# **LinePlots**
# 
# Good for seeing trends in data between induvidual variables

# In[ ]:


def lines(data):
    """
    Employment length vs interest rate
    """
    sns.lineplot(x=data['emp_length'], y=data['int_rate'])
    plt.title("Employment Length vs Interest Rate")
    plt.xlabel("Employment Length in yrs")
    plt.ylabel("Interest Rate in %")
    plt.show()
    timing.timer("Lines")
    return

lines(data)


# It seems interest rate vs employment length shows some non-linear relation with a clear drop in average interest rate from working 7 years to working 10 years, probably because stability in occupation is a sign of lower risk.
# The interest rate for people who have worked less than a year seems low though, possibly this is because these are small buisness or enterprise loans that are valued at a lower interest rate so that the buisness itself has a greater chance of success and thus repaying the loan.

# **Scatter Plot**
# 
# The most basic type of plot, but we will scatter averages because otherwise the graph will be too dense for us to actually learn anything

# In[ ]:


def scatter(data):
    """
    Scatter Sub_Grade vs Risk
    """
    info = data.copy()
    a = info.groupby('sub_grade').mean()
    
    sns.scatterplot(x=a.index, y=a['int_rate'])
    plt.title("Subgrade vs Interest Rate ScatterPlot")
    plt.xlabel("Subgrade")
    plt.ylabel("Interest Rate in %")
    plt.show()
    timing.timer("Scatter")
    return

scatter(data)


# Clearly there is a strong linear relationship between subgrade and interest rate. This makes it the best feature we have seen so far, and understandably so because the interest rate is in most cases a function of the risk of a loan.

# The rest of the plots are various different plots which show relationships in the data
# * 3D scatter
# * Violin Plot
# * Bubble plot

# In[ ]:


# 3D Scatterplot
def three_D_scatter(data):
    """
    Loan Amount vs Employment Length vs Interest Rate
    """
    from mpl_toolkits import mplot3d
    import numpy as np
    info = data[:1000]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xs = info['loan_amnt']
    zs = info['emp_length']
    ys = info['int_rate']
    ax.scatter(xs, ys, zs, s=1, alpha=1)


    ax.set_xlabel('Loan Amount')
    ax.set_ylabel('Interest Rate')
    ax.set_zlabel('Employment Length')
    plt.title("3D Scatterplot")
    plt.show()
    timing.timer("3D Scatter")
    return

three_D_scatter(data)


# In[ ]:


# Violin Plot
def violin_plot(data):
    sns.violinplot(x="home_ownership", y="int_rate", data=data, hue="term")
    plt.title("Violin Plot")
    plt.xlabel("Home Ownership")
    plt.ylabel("Interest Rate in %")
    plt.show()
    timing.timer("Violin")
    return

violin_plot(data)


# In[ ]:


# Bubble Chart
def bubble_chart(data):
    info = data[:1000]
    sns.lmplot(x="loan_amnt", y="int_rate",data=info,  fit_reg=False,scatter_kws={"s": info['annual_inc']*0.005})
    plt.title("Bubble Chart")
    plt.xlabel("Loan Amount")
    plt.ylabel("Interest Rate in %")
    plt.show()
    timing.timer("bubble")
    return

bubble_chart(data)


# Exploratory Data Analysis done. Now we will prepare our data for the model

# ## Data Processing
# <a id="processing**"></a>
# ***
# First let us define a function to split data and return it to us. This is useful because we want to be very sure of what manipulations we are doing to test data, in order to ensure we aren't cheating

# In[ ]:


def load_split_data(number_of_rows=None, purpose=None, column='int_rate', test_size=0.2):
    from sklearn.model_selection import train_test_split
    data = load_data(number_of_rows=number_of_rows, purpose=purpose)
    target = data[column]
    data.drop(column, axis=1, inplace=True)
    return train_test_split(data, target, test_size=test_size)


# Model training becomes too long with the entire dataset and so we will use only a subset of the data

# In[ ]:


X_train, X_test, y_train, y_test = load_split_data(50000, purpose="time_of_issue")
numericals, strings, categoricals = type_list_generator(X_train, separated=True)


# We use the data cleaning methods from above in order to prepare the data for processing

# In[ ]:


X_train = drop_nan_columns(X_train, ratio=0.5)
X_test = drop_nan_columns(X_test, ratio=0.5)
handle_nans(X_train)
handle_nans(X_test)
handle_types(X_train, numericals, strings, categoricals)
handle_types(X_test, numericals, strings, categoricals)
# For this notebook we will ignore the string variables, however there are ways to use them using other prepreocessing techniques if desired
X_train = X_train.drop(strings, axis=1)
X_test = X_test.drop(strings, axis=1)
timing.timer("Cleaned Data")


# First, let us fix the skew that we saw in the distribution plot using the square root transformation

# In[ ]:


def manage_skews(train_target, test_target):
    """
    Applying Square Root in order
    """
    timing.timer("Unskewed Data")
    return np.sqrt(train_target), np.sqrt(test_target)

y_train, y_test = manage_skews(y_train, y_test)


#  Next, we should normalize all of our data, this once again simply makes most models more effective and speeds up convergence

# In[ ]:


def scale_numerical_data(X_train, X_test, numericals):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[numericals] = sc.fit_transform(X_train[numericals])
    X_test[numericals] = sc.transform(X_test[numericals])
    timing.timer("Scaled Data")
    return

scale_numerical_data(X_train, X_test, numericals)


# Finally we will encode all of our categorical variables so that models can process them, but before we do that, we need to realize something about the size of our dataset
# i.e if you look at the columns such as employment title you can already see a whole bunch of different occupations - 432653 to be precise

# In[ ]:


X_train['emp_title'].value_counts()


# There are other columns, like purpose, that have this same issue. When there are so many different categories the model is likely to get extremely confused or is just unlikely to generalize well.
# Additionally there is also a harm that once we fit an encoder onto our categorical columns, then there will be a completely new profession in the test set that the encoder hasn't seen before, this would throw an exception
# To solve this problem we keep only the instances that make up the top 15 categories of that variable, and cast the rest to a standard value like "Other"

# In[ ]:


def shrink_categoricals(X_train, X_test, categoricals, top=25):
    """
    Mutatues categoricals to only keep the entries which are the top 25 of the daframe otherwise they become other
    """
    for category in categoricals:
        if category not in X_train.columns:
            continue
        tops = X_train[category].value_counts().index[:top]
        def helper(x):
            if x in tops:
                return x
            else:
                return "Other"
        X_train[category] = X_train[category].apply(helper)
        X_test[category] = X_test[category].apply(helper)
    timing.timer("Shrunk Categories")
    return


# In[ ]:


shrink_categoricals(X_train, X_test, categoricals)

X_train['emp_title'].value_counts()


# In[ ]:


def encode_categorical_data(X_train, X_test, categoricals):
    from sklearn.preprocessing import LabelEncoder
    for category in categoricals:
        if category not in X_train.columns:
            continue
        le = LabelEncoder()
        X_train[category] = le.fit_transform(X_train[category])
        X_test[category] = le.transform(X_test[category])
    timing.timer("Encoded Categoricals")
    return

encode_categorical_data(X_train, X_test, categoricals)


# We are nearly done with our data processing. The final step is to run a dimensionality reduction algorithm.This is because having many dimensions to data can make training models significantly slower and also sometimes less accurate as well. 
# 
# We will use PCA, an algorithm which tries to project data into lower dimensions while preserving as much entropy as possible. Instead of stating how many dimensions the PCA algorithm should project down to, we will simply state what percentage of entropy we would like preserved, 95% is a good standard bar.
# The reason we are doing the step last is because after the PCA it is virtually impossible to figure out what each column represents in terms of the original data.

# In[ ]:


def dimensionality_reduction(X_train, X_test):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    timing.timer("Dimensionality Reduced")
    return X_train, X_test

X_train, X_test = dimensionality_reduction(X_train, X_test)
X_train.shape


# From over 20 columns to only 5! While it may seem like we ought to have lost a lot of data the 95% of entropy being saved ensures we have preserved most core patterns.
# You could do the notebook without the above step to see how much slower the training of the models is without this step

# ## Data Modelling
# <a id="modelling**"></a>
# ***
# We are finally onto the modelling stage. We will create 4 different models - Random Forest, Support Vector Machine, Linear Regression, KNearestNeighbors and fine-tune them using scikit-learn

# The optimal model to return for each estimator was entered after running this section with a size of 500,000. It is computationally infeasible to run the Hyperparameter Tuning Algorithms on the full dataset

# In[ ]:


def random_forest(X_train, y_train, optimal=False):
    """
    Optimal = True returns an untrained model
    """
    from sklearn.ensemble import RandomForestRegressor
    if optimal:
        return RandomForestRegressor(n_estimators=120, max_depth=25, bootstrap=True, max_features=3)
    from sklearn.model_selection import GridSearchCV
 
    param_grid = [{'n_estimators':[60, 70, 80, 100, 120], 'max_depth':[15, 20, 25, None], 'bootstrap':[True, False], 'max_features':[None, 2, 3]}]
    forest = RandomForestRegressor()
    grid_search = GridSearchCV(forest, param_grid, cv=3, scoring="r2")
    grid_search.fit(X_train, y_train)
    timing.timer("Forest Grid Search Complete")
    final = grid_search.best_params_
    print(final)
    return grid_search.best_estimator_
    

def regression(X_train, y_train, optimal=False):
    """
    Optimal = True returns an untrained model
    """
    from sklearn.linear_model import ElasticNetCV
    if optimal:
        return ElasticNetCV(alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l1_ratio=0.0)
    from sklearn.model_selection import GridSearchCV

    elastic_net = ElasticNetCV()
    param_grid = {'alphas':[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], 'l1_ratio':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    grid_search = GridSearchCV(elastic_net, param_grid, scoring="r2", cv=3)
    grid_search.fit(X_train, y_train)
    timing.timer("Regression Grid Search Complete")
    print(grid_search.best_params_)
    return grid_search.best_estimator_ 


def knn(X_train, y_train, optimal=False):
    """
    Optimal = True returns an untrained model
    """
    from sklearn.neighbors import KNeighborsRegressor
    if optimal:
        return KNeighborsRegressor(n_neighbors=10, weights='distance')
    
    from sklearn.model_selection import GridSearchCV
    
    model = KNeighborsRegressor()
    param_grid = {'n_neighbors':[2,4,6,8,10,12,14], 'weights':['uniform', 'distance']}
    grid_search = GridSearchCV(model, param_grid, scoring="r2", cv=3)
    grid_search.fit(X_train, y_train)
    timing.timer("Neighbors Grid Search Complete")
    print(grid_search.best_params_)
    return grid_search.best_estimator_ 

def svm(X_train, y_train, optimal=False):
    """
    Optimal = True returns an untrained model
    """
    from sklearn.svm import SVR
    if optimal:
        return SVR()
    from sklearn.model_selection import RandomizedSearchCV

    svr = SVR()

    param_grid = {'kernel':['rbf', 'sigmoid', 'poly', 'linear'], 'C':[0.8, 1.0, 1.2]}
    n_iter = 2
    rsv = RandomizedSearchCV(svr, param_grid, n_iter=n_iter, scoring="r2")
    rsv.fit(X_train, y_train)
    timing.timer("SVR Random Search Complete")
    final = rsv.best_params_
    print(final)
    return rsv.best_estimator_


# The below code let's you create the model and tune its hyperparameters, it will not run if you use the full dataset. Either add a size parameter and set it below 20,000 when loading split data above or skip this block of code to proceed with the notebook. Now let us actually create the models and see what the best hyperparameters are

# In[ ]:


"""
model_creators = [random_forest,regression, knn, svm]
models = []
for creator in (model_creators):
    models.append(creator(X_train, y_train))
    
for model in models:
    model.score(X_test, y_test)
"""


# Apart from these models we will add two other models. The first is an ensemble learner, let us use an averaging ensemble technique to combine the models and see if it improves the R^2 Score

# In[ ]:


def ensemble(model_list):
    from sklearn.ensemble import VotingRegressor
    vtr = VotingRegressor(model_list)
    return vtr


# Finally we will try to use a Boosting technique -Gradient Boosting, 
# Gradient Boosting has a simple theory :-
# * A prelimnary model is trained on a dataset, and its residual errors are recorded. 
# * Another model is then trained to model these residual errors, 
# * The sum of the induvidual predictions is considered the prediction of the overall system.
# * This can chain to multiple models and not just two.
# 
# Scikit-Learn has an inbuilt GradientBoostingRegressor, but this uses Decision trees as its fundamental unit. We would rather use the SVM that is performing so poorly induvidually (comapared to the others), so we will have to manually define the class

# In[ ]:


class GradientBoost:
    def __init__(self, model_class, example, n_estimators=2):
        self.model_class = model_class
        self.parameters = example.get_params()
        self.n_estimators = n_estimators
        self.estimators = []
        for n in  range(n_estimators):
            model = model_class()
            model.set_params(**self.parameters)
            self.estimators.append(model)

    def fit_helper(self, X, y, i=0):
        if i >= len(self.estimators):
            return
        else:
            self.estimators[i].fit(X, y)
            preds = self.estimators[i].predict(X)
            error = y - preds
            self.fit_helper(X, error, i=i+1)

           
    def fit(self, X, y):
        self.fit_helper(X, y)

    def predict(self, X):
        prediction = self.estimators[0].predict(X)
        for estimator in self.estimators[1:]:
            prediction += estimator.predict(X)
        return prediction

    def score(self, X, y):
        from sklearn.metrics import r2_score
        preds = self.predict(X)
        return r2_score(y, preds)


# In[ ]:


order = {0: "rfr", 1:"lin_reg", 2:"knn", 3:"svr", 4:"vtr", 5:"gb"}
model_creators = [random_forest,regression, knn, svm]
model_list = []
models = []
for i, creator in enumerate(model_creators):
    model_list.append( (order[i] , creator(X_train, y_train, optimal=True) ) )
    models.append(creator(X_train, y_train, optimal=True))
    timing.timer(f"Appended Model {i}")
models.append(ensemble(model_list))
from sklearn.svm import SVR
models.append(GradientBoost(SVR, models[3], 2))


# Now we can create a training function

# In[ ]:


def train_and_test(models, order):
    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        timing.timer(f"Finished Fitting model {i}")
    scores = []
    for model in models:
        scores.append(model.score(X_test, y_test))
    final = {}
    for score_no in range(len(scores)):
        final[order[score_no]] = scores[score_no]
    return final


# In[ ]:


train_and_test(models, order)


# As we can see, both the ensemble and Gradient Boosting could reduce the error beyond the SVM threshold. The best model we have is clearly the KNN at around 99.8% R^2 score. However the SVM itself is quite a good model, 

# ## Model Assesment
# <a id="assesment**"></a>
# ***
# Let us look at our models accuracy now to get a sense of how close we are to the true interest rate.
# 

# In[ ]:


best_model = models[2]
# We take the first 20 inputs and compare the predictions with the outputs
truths = y_test[0:20]**2 # Squaring to undo the skew solution in order to truly reflect the data
preds = best_model.predict(X_test[0:20])**2
residual_error = truths - preds
print(residual_error)


# As you can see we have a pretty good model that can predict within around a 1% accuracy what the interest rate will be. We can also visualize this accuracy in a graph

# In[ ]:


plt.scatter(truths, preds)
plt.plot([7.0, 22], [7.0, 22], c = "red")
plt.title("Model Analysis")
plt.xlabel("Truth")
plt.ylabel("Prediction")
plt.show()


# In[ ]:


truths, preds


# ## Conclusion
# ***
# 
# Thank you for viewing this Kernel. The notebook is a work in progress and will be updated as per feedback and future ideas.

# In[ ]:




