#!/usr/bin/env python
# coding: utf-8

# ## Telecom Churn: XGBoost (Extreme Gradient Boosting)
# 
# 
# 
# 
# Data was downloaded from [IBM Sample Data](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/).

# In[ ]:


import os, logging, gc
from time import time
import pandas as pd
import numpy as np
import time

pd.set_option("display.max_columns", 50)

import warnings

warnings.filterwarnings(action='ignore')

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams 

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 15, 8

seed = 515
np.random.seed(seed)


# #### 1. Define UDFs
# 
# In this section, I will setup some local utilities to load the data and find the missing features. The data will be loaded and checked the null values as well.

# In[ ]:


def missing_values_table(df):
    #
    # Function to explore how many missing values (NaN) in the dataframe against its size
    # Args:
    #   df: the input dataframe for analysis
    # 
    # Return:
    #   mis_val_table_ren_columns: dataframe table contains the name of columns with missing data, # of missing values and % of missing against total
    #
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns and rows of " + str(df.shape[0]) + ".\n" "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.\n")
    return mis_val_table_ren_columns


def read_data(filename, nrows=10):
    #
    # Function to read the csv file onto the panda dataframe
    # Args:
    #   filename: The name of csv file
    #   nrows: number of rows to be read. Default is 10 rows. None will read all rows
    #
    # Return:
    #  df: panda dataframe containing the data from csv file
    #
    if(os.path.isfile(filename)):
        print("\nReading file:: {}\n".format(filename))
        df = pd.read_csv(filename, sep = ',', nrows = nrows)
        df.columns = [x.lower() for x in df.columns]
        print("\n=======================================================================")
        print("Sample records: \n", df.head(2))
        print("\n=======================================================================")
        print("The data type: \n", df.columns.to_series().groupby(df.dtypes).groups)
        print("\n=======================================================================")
        print("Checking missing data (NaN): \n", missing_values_table(df))
        
    else:
        logging.warning("File is not existed")
        df = None
        
    return df


def one_way_tab (df, col):
    #
    # Function to compute one way table
    # Args:
    #   df: pandas dataframe
    #   col: column name to tabulate
    #
    # Return:
    #   df: the tabulate pandas of the outcome
    #
    sns.countplot(x = col, data = df)
    plt.show();
    df = pd.crosstab(index = df[col], columns = "count")
    df['percent'] = df/df.sum() * 100
    return df


# In[ ]:


data_file = "../input/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = read_data(data_file, nrows = None)


# #### 2. Data Sensemaking
# After the data loaded, we can see that there is no missing data, however, **totalcharges** feature has data types as *object*. Based on the sample data, this should be numeric data.

# In[ ]:


display(df.head(5))


# For each observation, this represents one subscriber, with his / her details, product holding and other services subscription.
# 
# In the end, we want to predict whether the subscribers become **churn** or not. Let's quickly look at the distribution of the target variable.

# In[ ]:


one_way_tab(df, 'churn')


# Is there any duplicated Customer ID in the dataframe? Based on the below result, there is no duplicated **customer id** in the data.

# In[ ]:


df[df.duplicated(['customerid'], keep=False)]


# Based on the above finding, **total charges** column should not be an object. Let's start by looking at this column.

# In[ ]:


df['totalcharges'] = df['totalcharges'].replace(r'\s+', np.nan, regex=True)
df['totalcharges'] = pd.to_numeric(df['totalcharges'])


# In[ ]:


missing_values_table(df)


# By exploring the missing total charges column, we can see that all of them has **tenure** of 0. Hence, this is the newbie subscriber (as tenure represents in a month unit).
# 
# This is the case depends on each operator in each country, because sometimes they will pro-rate based on day and some will wait until specific days (i.e. 15 days after usage) and charge the full amount.
# 
# However, we are given ***tenure = 0***, we cannot correctly decipher the values. Hence, we will fix this value to become 0.

# In[ ]:


df[df.totalcharges.isnull()]


# In[ ]:


df.loc[df.totalcharges.isnull(), 'totalcharges'] = 0
sns.distplot(df.totalcharges)
plt.show();


# #### Univariate Analysis
# 
# Let's look at each feature (except **customerid**) to see the unique values and the distribution of the features.
# 
# I have created a function to quickly plot the data points based on its type by separating **object** and **non-object** from each other.

# In[ ]:


def display_plot(df, col_to_exclude, object_mode = True):
    """ 
     This function plots the count or distribution of each column in the dataframe based on specified inputs
     @Args
       df: pandas dataframe
       col_to_exclude: specific column to exclude from the plot, used for excluded key 
       object_mode: whether to plot on object data types or not (default: True)
       
     Return
       No object returned but visualized plot will return based on specified inputs
    """
    n = 0
    this = []
    
    if object_mode:
        nrows = 4
        ncols = 4
        width = 20
        height = 20
    
    else:
        nrows = 2
        ncols = 2
        width = 14
        height = 10
    
    
    for column in df.columns:
        if object_mode:
            if (df[column].dtypes == 'O') & (column != col_to_exclude):
                this.append(column)
                
                
        else:
            if (df[column].dtypes != 'O'):
                this.append(column)
     
    
    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(width, height))
    for row in range(nrows):
        for col in range(ncols):
            if object_mode:
                sns.countplot(df[this[n]], ax=ax[row][col])
                
            else:
                sns.distplot(df[this[n]], ax = ax[row][col])
            
            ax[row,col].set_title("Column name: {}".format(this[n]))
            ax[row, col].set_xlabel("")
            ax[row, col].set_ylabel("")
            n += 1

    plt.show();
    return None


# In[ ]:


display_plot(df, 'customerid', object_mode = True)


# In[ ]:


display_plot(df, 'customerid', object_mode = False)


# Based on the value of the services the subscribers subscribed to, there are **yes**, **no**, and **no phone / internet service**. These are somewhat related to primary products. Examples are illustrated through *panda crosstab* function below:
# 
# 1. **Phone service (Primary) and Multiple lines (Secondary)**
#  
#  - If the subscribers have phone service, they may have multiple lines (yes or no). 
#  - But if the subscribers don't have phone service, the subscribers will never have multiple lines.
#  
#  
# 2. **Internet Service (Primary) and other services, let's say streaming TV (secondary)**
# 
#  - If the subscribers have Internet services (either DSL or Fiber optic), the subscribers may opt to have other services related to Internet (i.e. streaming TV, device protection).
#  - But if the subscribers don't have the Internet services, this secondary service will not be available for the subscribers.
#  
# 
# With this conclusion, I opt to transform the feature value of **No Phone / Internet service** to be the same **No** because it can be used another features (hence, **phone service** and **internet service** column) to explain.

# In[ ]:


pd.crosstab(index = df["phoneservice"], columns = df["multiplelines"])


# In[ ]:


pd.crosstab(index = df["internetservice"], columns = df["streamingtv"])


# In[ ]:


def convert_no_service (df):
    col_to_transform = []
    for col in df.columns:
        if (df[col].dtype == 'O') & (col != 'customerid'):
            if len(df[df[col].str.contains("No")][col].unique()) > 1:
                col_to_transform.append(col)
    
    print("Total column(s) to transform: {}".format(col_to_transform))
    for col in col_to_transform:
        df.loc[df[col].str.contains("No"), col] = 'No'
        
    return df


# In[ ]:


df = convert_no_service(df)


# Let's see the data after transformation.

# In[ ]:


display_plot(df, 'customerid', object_mode = True)


# We will start to transform the data for building the predictive model in the next section.
# 
# - Encode Yes / No into 1 / 0, respectively
# - Encode Male / Female into 1/0, respectively (for **gender** feature)
# - Create dummy variables for **internet service, contract, and payment method** features.

# In[ ]:


df.gender = df.gender.map(dict(Male=1, Female=0))
display(df.gender.value_counts())


# In[ ]:


def encode_yes_no (df, columns_to_encode):
    for col in columns_to_encode:
        df[col] = df[col].map(dict(Yes = 1, No = 0))
        
    return df


# In[ ]:


encode_columns = []
for col in df.columns:
    keep = np.sort(df[col].unique(), axis = None)
    
    if ("Yes" in keep) & ("No" in keep):
        encode_columns.append(col)

del keep
print("Encode Columns Yes/No: {}".format(encode_columns))
        
    


# In[ ]:


df = encode_yes_no(df, encode_columns)
display(df.head(5))


# In[ ]:


df = pd.get_dummies(df, columns = ['internetservice', 'contract', 'paymentmethod'], prefix = ['ISP', 'contract', 'payment'])
display(df.head(5))


# As we start to develop the model, let's quickly drop **customerid** column and assign it to new dataframe.

# In[ ]:


df2 = df.drop('customerid', axis = 1, inplace = False)
df2.columns = df2.columns.str.replace(" ", "_")


# #### 3. Machine Learning 
# 
# Prior to model build, let's quickly observe the correlation of the data we have.
# 
# - Based on this, looks like there are positive correlation between churn and those Month-to-month contracts.
#    - Assumption is that the month-to-month contract doesn't require a commitment from the subscribers, hence they can easily churn out (stop using the services)
# - The second and third variables with positive correlation are **Fiber Optic ISP** and **monthly charges**.

# In[ ]:


df2.corr()['churn'].sort_values(ascending=False)


# In[ ]:


corr = df2.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(16, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.6, cbar_kws={"shrink": .5})
plt.show();


# In[ ]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
import pickle
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import make_scorer


# In[ ]:


X = df2.drop('churn', axis = 1, inplace = False)
y = df2['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed)
print("Training target distribution:\n{}".format(y_train.value_counts()))
print("\nTesting target distribution:\n{}".format(y_test.value_counts()))


# In[ ]:


def xgb_f1(y, t):
    #
    # Function to evaluate the prediction based on F1 score, this will be used as evaluation metric when training xgboost model
    # Args:
    #   y: label
    #   t: predicted
    #
    # Return:
    #   f1: F1 score of the actual and predicted
    #
    t = t.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]   # change the prob to class output
    return 'f1', f1_score(t, y_bin)


def plot_evaluation_metric (y_true, y_prob):
    #
    # Function to plot the evaluation metric (cumulative gain, lift chart, precision and recall) on the screen
    # Args:
    #   y_true: array of y true label
    #   y_prob: array of y predicted probability (outcome of predict_proba() function)
    #
    # Return:
    #   None
    #
    skplt.metrics.plot_cumulative_gain(y_true, y_prob)
    plt.show();
    skplt.metrics.plot_precision_recall(y_true, y_prob)
    plt.show();
    skplt.metrics.plot_lift_curve(y_true, y_prob)
    plt.show();
    return 


def print_evaluation_metric (y_true, y_pred):
    #
    # Function to print out the model evaluation metrics
    # Args:
    #   y_true: array of y true label
    #   y_pred: array of y predicted class
    #
    # Return:
    #   None
    #
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F-score: {}".format(fscore))
    print("Support: {}".format(support))
    return 


def get_confusion_matrix (y_true, y_pred, save=0, filename="this.csv"):
    #
    # Function to print out the confusion matrix on screen as well as print to csv file, if enabled
    # Args:
    #   y_true: array of y true label
    #   y_pred: array of y prediction
    #   save: to enable the write to csv file (default = 0)
    #   filename: the name of the file to be saved (default = this.csv)
    #
    # Return:
    #   None
    #
    from sklearn.metrics import confusion_matrix
    get_ipython().magic('matplotlib inline')
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred),
                      columns = ['Predicted False', 'Predicted True'],
                      index = ['Actual False', 'Actual True']
                      )
    display(cm)
    if(save):
        cm.to_csv(filename, index = True)
    
    return 


def my_plot_roc_curve (y_true, y_prob, filename="img.png", dpi = 200):
    #
    # Function to plot the ROC curve by computing fpr and tpr as well as save the plot to file
    # Args:
    #   y_true: array of y true label
    #   y_prob: the output of y probability prediction (outcome for predict_proba() function)
    #   filename: the name of the file to be saved
    #   dpi: the resolution of the figure
    # Return:
    #   None
    #
    fpr, tpr, threshold = roc_curve(y_true, y_prob[:, 1])
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    fig.savefig(filename, dpi = dpi)
    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return


# #### Build the classifiers
# 
# Let's try (very) basic classifiers, using only default setting of each classifier's algorithm. I will try on 3 different algorithms:
# 
# 1. K-NN (K-Nearest Neighbors) classifier
# 2. Random Forest Classifier
# 3. XGBoosting Classifier
# 
# Then I will print out the accuracy score and F1 score of each classifier.

# In[ ]:


classifiers = [
    KNeighborsClassifier(n_jobs = 4),
    RandomForestClassifier(n_jobs = 4),
    XGBClassifier(n_jobs = 4)
]

# iterate over classifiers
for item in classifiers:
    classifier_name = ((str(item)[:(str(item).find("("))]))
    print (classifier_name)
    
    # Create classifier, train it and test it.
    clf = item
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print ("Score: ", round(score,3),"\nF1 score: ", round(f1_score(y_test, pred), 3), "\n- - - - - ", "\n")
    


# In[ ]:


param_grid = {
    'silent': [False],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.001, 0.01, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [0.5, 1.0, 3.0],
    'gamma': [0, 0.25, 0.5, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
    'n_estimators': [50, 100, 150],
    'scale_pos_weight': [1, 1.5, 2],
    'max_delta_step': [1, 2, 3]
}

clf = XGBClassifier(objective = 'binary:logistic')


# In[ ]:


fit_params = {'eval_metric': 'logloss',
              'early_stopping_rounds': 10,
              'eval_set': [(X_test, y_test)]}

rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=50,
                            n_jobs=4, verbose=2, cv=5,
                            fit_params=fit_params,
                            scoring= 'f1_macro', refit=True, random_state=seed)


print("Randomized search..")
search_time_start = time.time()
rs_clf.fit(X_train, y_train)
print("Randomized search time:", time.time() - search_time_start)

best_score = rs_clf.best_score_
best_params = rs_clf.best_params_
print("Best score: {}".format(best_score))
print("Best params: ")
for param_name in sorted(best_params.keys()):
    print('%s: %r' % (param_name, best_params[param_name]))


# In[ ]:


best_xgb = XGBClassifier(objective = 'binary:logistic',
                         colsample_bylevel = 0.7,
                         colsample_bytree = 0.8,
                         gamma = 1,
                         learning_rate = 0.15,
                         max_delta_step = 3,
                         max_depth = 4,
                         min_child_weight = 1,
                         n_estimators = 50,
                         reg_lambda = 10,
                         scale_pos_weight = 1.5,
                         subsample = 0.9,
                         silent = False,
                         n_jobs = 4
                        )

best_xgb.fit(X_train, y_train, eval_metric = xgb_f1, eval_set = [(X_train, y_train), (X_test, y_test)], 
             early_stopping_rounds = 20)


# In[ ]:


xgb.plot_importance(best_xgb, max_num_features = 15)
plt.show();


# In[ ]:


y_pred = best_xgb.predict(X_test)
y_prob = best_xgb.predict_proba(X_test)
print_evaluation_metric(y_test, y_pred)
get_confusion_matrix (y_test, y_pred, save=0, filename="this.csv")
my_plot_roc_curve (y_test, y_prob, filename="ROC.png", dpi = 200)
plot_evaluation_metric (y_test, y_prob)


# In[ ]:


from sklearn.metrics import classification_report
ev = classification_report(y_test, y_pred, target_names = ['Not Churn', 'Churn'])
print(ev)


# In[ ]:


from xgboost import plot_tree
import graphviz

plot_tree(best_xgb, num_trees = 0)
fig = plt.gcf()
fig.set_size_inches(300, 100)
fig.savefig('tree.png')


# #### 4. Application
# 
# Normally in business setting, we will use the prediction score, rather than the class prediction. This score is used to associate with the subscriber profiles, and can be used to adjust for campaign targeting.

# In[ ]:


y_all_prob = best_xgb.predict_proba(X)
df['churn_prob'] = y_all_prob[:, 1]
sns.distplot(df['churn_prob'])
plt.show();


# In[ ]:


df[['customerid', 'churn', 'churn_prob']].head(10)


# #### Appendix: Explain XGBoost model
# 
# I will add **SHAP** package to help visualization and explain the ***XGBoost Classifier***. 
# 
# First, we will need to use *TreeExplainer* to the XGBoost model object.

# In[ ]:


import shap
shap.initjs()

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_train)


# When we train the model, we can get the SHAP values, rotate the data and then stack them, hence we can see the explanations for the entire dataset and we can see which features influence the output (**Note**: The plot is interactive).

# In[ ]:


shap.force_plot(explainer.expected_value, shap_values, X_train)


# We can display how each feature influenced the output of the model. SHAP values sum to the difference between the expected output of the model and the current output for the observation. Note that for the Tree SHAP implementation, the margin output of the model is explained, not the transformed output (i.e. output from **predict** function).
# 
# This means the units of the SHAP values for this model are log odd rations. Large positive values mean a subscriber is likely to churn.
# 
# Below can be used in place of Feature importances plot. 

# In[ ]:


shap.summary_plot(shap_values, X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")


# Furthermore, we can use **dependence_plot** to understand the effects of the single feature to the model output. Since SHAP values represent a feature's responsibility for a change in the model output, the plot below represents the change in predicted churn as the subscribers have **ISP - Fiber Optic** product. 
# 
# Vertical dispersion at a single value of ISP Fiber Optic represents the interaction effects with another features, in this plot, I assign **monthly charges** for coloring. Based on this plot, we can see that the Fiber Optic subscribers who have higher monthly charges are likely to churn more than those who don't have Fiber optic.

# In[ ]:


shap.dependence_plot("ISP_Fiber_optic", shap_values, X_train, interaction_index="monthlycharges")


# In[ ]:





# In[ ]:


shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], link="logit")

