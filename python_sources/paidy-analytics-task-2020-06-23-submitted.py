#!/usr/bin/env python
# coding: utf-8

# # Key insights summary
# 
# ### (1) For predicting the 2-years default (<code>SeriousDlqin2yrs</code>), the following 7 columns are useful:
# * <code>RevolvingUtilizationOfUnsecuredLines</code> correlation with the target 0.28
# * <code>DebtRatio</code>
# * <code>NumberOfOpenCreditLinesAndLoans</code>
# * <code>age</code>
# * <code>NumberOfTime30-59DaysPastDueNotWorse</code> correlation with the target 0.28
# * <code>NumberOfTime60-89DaysPastDueNotWorse</code> correlation with the target 0.27
# * <code>NumberOfTimes90DaysLate</code>: correlation with the target 0.32
#     
# In other words, there is no need to use extra resources to obtain and/or increase the quality of the <code>MonthlyIncome</code> and <code>NumberOfDependents</code> data.
# 
# ### (2) RevolvingUtilizationOfUnsecuredLines is a great predictor for the 2-year default, effective for all age groups
# More than half of all defaulted users have the <code>RevolvingUtilizationOfUnsecuredLines</code> above 0.8,<br>but only 18% of non-defaulted users are above that value.
# 
# 
# ### (3) More users default at younger age
# Half of all defaults are happening before 45, while the average are of the non-default user is around 52.<br>
# Noticeably, younger age groups (20 ~ 40) have the highest concentration of users with high <code>RevolvingUtilizationOfUnsecuredLines</code>.
# 
# ### (4) Average DebtRatio of the default users is above 0.4 for over 40 age group
# For the non-default users, the average <code>DebtRatio</code> is at 0.33 for the 40~50 age group, and from there decreases<br>
# to below 0.2 for users over 80. 

# # Step 0: Environment preparation, data load, brief quality check

# In[ ]:


# Preparing essentials environment
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn import preprocessing
from itertools import cycle
pd.options.mode.chained_assignment = None


# In[ ]:


# Reading CSV data
df = pd.read_csv('../input/give-me-some-credit-dataset/cs-training.csv')


# In[ ]:


# Checking the data types and column names
df.dtypes


# In[ ]:


# Checking the data top 15 rows 
df.head(15)


# **Observations from the sample:**
# * <code>Unnamed: 0</code> is most likely the "id" column
# * <code>DebtRatio</code> has unexpected (> 1) values
# * <code>MonthlyIncome</code> has unexpected (NaN, 0) values
# * <code>NumberOfDependants</code> has unexpected (NaN) values

# In[ ]:


# Checking basic stats
rs = round(df.describe(), 2)
rs


# In[ ]:


# Renaming the first column to "id"
df.rename(
    columns = {'Unnamed: 0':'id'},
    inplace = True
)


# In[ ]:


# Counting empty cells in each column
df.isnull().sum()


# #### Quick notes about the data quality:
# * <code>RevolvingUtilizationOfUnsecuredLines</code>: contains wrong data (max = 50708.00), needs cleaning
# * <code>age</code>: contains wrong data (min = 0.00), needs cleaning
# * <code>NumberOfTime30-59DaysPastDueNotWorse</code>: seems fine
# * <code>DebtRatio</code>: seems off when MonthlyIncome is "NaN", contains wrong data (max = 329664.00), needs cleaning
# * <code>MonthlyIncome</code>: 19.8% are "NaN" values, some outliers (min = 0, max = 3,008,750) >> needs cleaning, possibly splitting the dataset in 2 to build the model if this is a good predictor
# * <code>NumberOfOpenCreditLinesAndLoans</code>: seems fine
# * <code>NumberOfTimes90DaysLate</code>: seems fine 
# * <code>NumberRealEstateLoansOrLines</code>: seems fine 
# * <code>NumberOfTime60-89DaysPastDueNotWorse</code>: seems fine
# * <code>NumberOfDependents</code>: 2.6% are "NaN" values, needs cleaning

# # Step 1: Data cleaning: wrong values, nulls, outliers, etc.

# ### 1) RevolvingUtilizationOfUnsecuredLines

# In[ ]:


# Checking the issues with RevolvingUtilizationOfUnsecuredLines
df[df["RevolvingUtilizationOfUnsecuredLines"] > 2].sample(n = 30)


# In[ ]:


# There seems to be no clear source of the issue (i.e. no connection with other columns) for the wrong values
# Checking how many RevolvingUtilizationOfUnsecuredLines values are over 2
df["id"][df["RevolvingUtilizationOfUnsecuredLines"] >= 2].count()


# In[ ]:


# Checking RevolvingUtilizationOfUnsecuredLines for outliers (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
plt.figure(figsize = (4,8))
a = sns.boxplot(
    y = "RevolvingUtilizationOfUnsecuredLines", 
    x = "SeriousDlqin2yrs",
    data = df
)
a.set(
    ylim = (0, 2)
);


# #### Data prep actions memo:
# * <code>RevolvingUtilizationOfUnsecuredLines</code> limit at 1.9

# ### 2) age

# In[ ]:


df.sort_values(by = ["age"]).head(5)


# In[ ]:


df.sort_values(by = ["age"], ascending = False).head(5)


# #### Data prep actions memo:
# * <code>RevolvingUtilizationOfUnsecuredLines</code> limit at 1.4
# * <code>age</code> remove one row with 0

# ### 3) NumberOfTime30-59DaysPastDueNotWorse, NumberOfTime60-89DaysPastDueNotWorse, NumberOfTimes90DaysLate

# In[ ]:


df['NumberOfTime30-59DaysPastDueNotWorse'].value_counts().sort_index()


# In[ ]:


df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts().sort_index()


# In[ ]:


df['NumberOfTimes90DaysLate'].value_counts().sort_index()


# #### Data prep actions memo:
# * <code>RevolvingUtilizationOfUnsecuredLines</code> limit at 1.9
# * <code>age</code> remove one row with 0
# * <code>NumberOfTime30-59DaysPastDueNotWorse</code>: clip at 5 to exclude values "96" and "98"
# * <code>NumberOfTime60-89DaysPastDueNotWorse</code>: clip at 5 to exclude values "96" and "98"
# * <code>NumberOfTimes90DaysLate</code>: clip at 6 to exclude values "96" and "98"

# ### 4) MonthlyIncome and DebtRatio

# In[ ]:


# Checking the data when MonthlyIncome is null 
rs = round(df[df["MonthlyIncome"].isnull()].describe(), 2)
rs


# In[ ]:


# Seems like when MonthlyIncome is null, DebtRatio is 100% wrong (i.e. way over 1, which should be it's max value by definition)...
plt.figure(figsize = (16,5))

a = sns.boxplot(
    y = "DebtRatio",
    x = pd.qcut((df[df["MonthlyIncome"].isnull()]["age"]), 15),
    data = df[df["MonthlyIncome"].isnull()]
)

a.set(
    ylim = (0, 9000)
)

plt.setp(
    a.get_xticklabels(), 
    rotation = 55
);


# In[ ]:


# ... and when MonthlyIncome is not null, DebtRatio is behaving as expected and could be considered as "correct"
plt.figure(figsize = (16,10))

a = sns.boxplot(
    y = "DebtRatio",
    x = pd.qcut((df[df["MonthlyIncome"] > 0]["age"]), 15),
    data = df[df["MonthlyIncome"] > 0]
)

a.set(
    ylim = (0, 10)
)

plt.setp(
    a.get_xticklabels(), 
    rotation = 55
);


# In[ ]:


df[df["MonthlyIncome"] > 0]['DebtRatio'].value_counts().sort_index()


# In[ ]:


df[df["MonthlyIncome"] > 0]['DebtRatio'].describe()


# In[ ]:


df[df["MonthlyIncome"] > 0]['MonthlyIncome'].quantile(.02)


# In[ ]:


df[df["MonthlyIncome"] > 0]['MonthlyIncome'].quantile(.98)


# #### Data prep actions memo:
# * <code>RevolvingUtilizationOfUnsecuredLines</code> limit at 1.9
# * <code>age</code> remove one row with 0
# * <code>NumberOfTime30-59DaysPastDueNotWorse</code>: clip at 5 to exclude values "96" and "98"
# * <code>NumberOfTime60-89DaysPastDueNotWorse</code>: clip at 5 to exclude values "96" and "98"
# * <code>NumberOfTimes90DaysLate</code>: clip at 6 to exclude values "96" and "98"
# * <code>MonthlyIncome</code>: remove nulls for analysis, clip between 800 and 20,000
# * <code>DebtRatio</code>: after removing MonthlyIncome null data the distribuion should be back to as expected, limit at 50 and clip at 1.2
# 

# ### 5) NumberOfOpenCreditLinesAndLoans

# In[ ]:


# Checking NumberOfOpenCreditLinesAndLoans for outliers (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
plt.figure(figsize = (4,8))

a = sns.boxplot(
    y = "NumberOfOpenCreditLinesAndLoans",
    x ="SeriousDlqin2yrs",  
    data = df
);


# ### 6) NumberRealEstateLoansOrLines

# In[ ]:


# Checking NumberRealEstateLoansOrLines for outliers (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
plt.figure(figsize = (4,8))

a = sns.boxplot(
    y = "NumberRealEstateLoansOrLines",
    x ="SeriousDlqin2yrs",  
    data = df
);


# ### 7) NumberOfDependents

# In[ ]:


# Checking NumberOfDependents for outliers (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
plt.figure(figsize = (4,8))

a = sns.boxplot(
    y = "NumberOfDependents",
    x ="SeriousDlqin2yrs",  
    data = df
);


# #### Data prep actions memo:
# * <code>RevolvingUtilizationOfUnsecuredLines</code> limit at 1.9
# * <code>age</code> remove one row with 0
# * <code>NumberOfTime30-59DaysPastDueNotWorse</code>: clip at 5 to exclude values "96" and "98"
# * <code>NumberOfTime60-89DaysPastDueNotWorse</code>: clip at 5 to exclude values "96" and "98"
# * <code>NumberOfTimes90DaysLate</code>: clip at 6 to exclude values "96" and "98"
# * <code>MonthlyIncome</code>: remove nulls for analysis, clip between 800 and 20,000
# * <code>DebtRatio</code>: after removing MonthlyIncome null data the distribuion should be back to as expected, limit at 50 and clip at 1.2
# * <code>NumberOfOpenCreditLinesAndLoans</code>: clip at 20
# * <code>NumberRealEstateLoansOrLines</code>: clip at 5
# * <code>NumberOfDependents</code>: clip at 5

# # Step 2: Making a clean dataset, EDA

# In[ ]:


# Creating a clean dataset for EDA according to the Actions memo
df_clean = df[
    df['RevolvingUtilizationOfUnsecuredLines'].notnull() & 
    (df['RevolvingUtilizationOfUnsecuredLines'] <= 1.9) &
    (df['age'] > 0) &
    (df['DebtRatio'] < 50) &
    (df['MonthlyIncome'] > 0) &
    (df['MonthlyIncome'].notnull())
]

df_clean["RevolvingUtilizationOfUnsecuredLines"] = df_clean["RevolvingUtilizationOfUnsecuredLines"].clip(upper = 1.2)
df_clean["age"] = df_clean["age"].clip(upper = 95)
df_clean["NumberOfTime30-59DaysPastDueNotWorse"] = df_clean["NumberOfTime30-59DaysPastDueNotWorse"].clip(upper = 5)
df_clean["NumberOfTime60-89DaysPastDueNotWorse"] = df_clean["NumberOfTime60-89DaysPastDueNotWorse"].clip(upper = 5)
df_clean["NumberOfTimes90DaysLate"] = df_clean["NumberOfTimes90DaysLate"].clip(upper = 6)
df_clean["MonthlyIncome"] = df_clean["MonthlyIncome"].clip(upper = 20000, lower = 800)
df_clean["DebtRatio"] = df_clean["DebtRatio"].clip(upper = 1.2)
df_clean["NumberOfOpenCreditLinesAndLoans"] = df_clean["NumberOfOpenCreditLinesAndLoans"].clip(upper = 20)
df_clean["NumberRealEstateLoansOrLines"] = df_clean["NumberRealEstateLoansOrLines"].clip(upper = 5)
df_clean["NumberOfDependents"] = df_clean["NumberOfDependents"].clip(upper = 5)
df_clean["NumberOfDependents"].fillna(0, inplace = True)


# In[ ]:


# Adding a custom predictor
df_clean["Custom1"] = df_clean["NumberOfTime30-59DaysPastDueNotWorse"] + df_clean["NumberOfTime60-89DaysPastDueNotWorse"] * 1.6 + df_clean["NumberOfTimes90DaysLate"] * 2


# In[ ]:


# Checking the clean dataset 
rs = round(df_clean.describe(), 2)
rs


# In[ ]:


# Checking the distribution of the target variable SeriousDlqin2yrs
sns.countplot(
    x = "SeriousDlqin2yrs",
    data = df_clean
);
# Seems like Accuracy is not a good metric for ML models


# In[ ]:


# Plotting the pair grid to better understand connections between columns 
grid = sns.pairplot(
    df_clean[["SeriousDlqin2yrs",
              "RevolvingUtilizationOfUnsecuredLines",
              "age",
              "DebtRatio",
              "MonthlyIncome",
              "NumberOfOpenCreditLinesAndLoans",
              "NumberRealEstateLoansOrLines",
              "NumberOfDependents"
             ]].sample(n = 3000),
    hue = "SeriousDlqin2yrs",
    height = 3,
    kind = "reg",
    plot_kws = {'scatter_kws': {'alpha': 0}}
)
grid = grid.map_upper(plt.scatter)
grid = grid.map_lower(
    sns.kdeplot, 
    shade = True,
    shade_lowest = False,
    alpha = 0.6,
    n_levels = 5
);


# #### Explore further memo:
# * Defaulted (<code>SeriousDlqin2yrs</code> = 1) users are generally younger than non-defaulted users 
# * Defaulted users have <code>RevolvingUtilizationOfUnsecuredLines</code> much closer to maximum (1), which is opposite for the non-defaulted users
# * Defaulted users' <code>DebpRatio</code> tend to be increasing with <code>age</code>, and for non-defaulted users the trend is the opposite 

# In[ ]:


# Checking most highly correlated variables
def highestcorrelatedpairs (df, top_num):
    correl_matrix = df.corr()
    correl_matrix *=np.tri(*correl_matrix.values.shape, k = -1).T
    correl_matrix = correl_matrix.stack()
    correl_matrix = correl_matrix.reindex(correl_matrix.abs().sort_values(ascending = False).index).reset_index()
    correl_matrix.columns = [
        "Variable 1",
        "Variable 2",
        "Correlation"
    ]
    return correl_matrix.head(top_num)

highestcorrelatedpairs(df_clean, 16)


# #### Explore further memo - part 2:
# * <code>MonthlyIncome</code> has an unexpected medium positive correlation with <code>NumberRealEstateLoansOrLines</code>

# In[ ]:


# Preparations for ECDF plot 
def ecdf_plot(df, col, split):
    x0 = np.sort(df[(df[split] == 0) | (df[split] == -1)][col])
    x1 = np.sort(df[df[split] == 1][col])
    y0 = np.arange(1, len(x0)+1) / len(x0)
    y1 = np.arange(1, len(x1)+1) / len(x1)
    _ = plt.plot(x0, y0, marker = '.', linestyle = 'none')
    _ = plt.plot(x1, y1, marker = '.', linestyle = 'none')
    plt.margins(0.04) 
    plt.legend([split + ": 0", split + ": 1"])
    plt.xlabel(col, fontsize = 12)
    plt.grid()
    plt.show()


# In[ ]:


# 1st variable for ECDF: Age
plt.figure(figsize = (8.5,6))
ecdf_plot(df_clean, "age", "SeriousDlqin2yrs")


# ![](http://)Average age of the defaulted user is around **45**, and non-default user is around **52**

# In[ ]:


# 2nd pair for ECDF: RevolvingUtilizationOfUnsecuredLines
plt.figure(figsize = (8.5,6))
ecdf_plot(df_clean, "RevolvingUtilizationOfUnsecuredLines", "SeriousDlqin2yrs")


# More than half of defaulted users have the <code>RevolvingUtilizationOfUnsecuredLines</code> above **0.8**<br>
# Only 18% of non-defaulted users have such high <code>RevolvingUtilizationOfUnsecuredLines</code>

# In[ ]:


# Defining ageRange for easier visualization
ageRange = pd.interval_range(
    start = 20, 
    freq = 10, 
    end = 90
)
df_clean['ageRange'] = pd.cut(df_clean['age'], bins = ageRange)


# In[ ]:


# Exploring the connections between RevolvingUtilizationOfUnsecuredLines and age
plt.figure(figsize = (16,8))
sns.violinplot(
    y = "RevolvingUtilizationOfUnsecuredLines",
    x = "ageRange",
    data = df_clean
);


# Out of all age groups, users between 20 and 30 have the highest ratio of very high (close to 1) <code>RevolvingUtilizationOfUnsecuredLines</code>

# In[ ]:


# Explorning the RevolvingUtilizationOfUnsecuredLines by age groups for both categories of the target variable
plt.figure(figsize = (16,8))
sns.boxplot(
    y = "RevolvingUtilizationOfUnsecuredLines",
    x = "ageRange",
    hue ="SeriousDlqin2yrs",  
    data = df_clean
);


# For all age groups, <code>RevolvingUtilizationOfUnsecuredLines</code> is noticably different between defaulted and non-defaulted users

# In[ ]:


# Explorning the DebtRatio by age groups for both categories of the target variable
plt.figure(figsize = (16,8))
sns.boxplot(
    y = "DebtRatio",
    x = "ageRange",
    hue ="SeriousDlqin2yrs",  
    data = df_clean
);


# While the average <code>DebtRatio</code> is decreasing with age for the groups from 40 to 80 for non-defaulted users, it almost doesn't change for defaulted users

# In[ ]:


# Checking the RevolvingUtilizationOfUnsecuredLines distribution differences by age group and both categories of the target variable
plt.figure(figsize = (16,8))
sns.violinplot(
    y = "RevolvingUtilizationOfUnsecuredLines",
    x = "ageRange",
    hue = "SeriousDlqin2yrs",  
    data = df_clean,
    split = True,
    inner = "quart"
);


# For higher age groups, it's easier to tell if the user would default if the <code>RevolvingUtilizationOfUnsecuredLines</code> is high (around "1")

# In[ ]:


# Checking the RevolvingUtilizationOfUnsecuredLines distribution differences by NumberOfTime30-59DaysPastDueNotWorse and both categories of the target variable
plt.figure(figsize = (16,8))
sns.violinplot(
    y = "RevolvingUtilizationOfUnsecuredLines",
    x = "NumberOfTime30-59DaysPastDueNotWorse",
    hue = "SeriousDlqin2yrs",  
    data = df_clean,
    split = True,
    inner = "quart"
);


# When the <code>NumberOfTime30-59DaysPastDueNotWorse</code> is **4 or higher**, it becomes less effective to use <code>RevolvingUtilizationOfUnsecuredLines</code> as a predictor of default.

# In[ ]:


g = sns.FacetGrid(
    df_clean,
    col = "SeriousDlqin2yrs", 
    row = "ageRange", 
    height = 2.5,
    aspect = 1.6
)
g.map(sns.kdeplot, "RevolvingUtilizationOfUnsecuredLines", "DebtRatio");
plt.ylim(-0.5, 1.5);


# In[ ]:


# Defining MonthlyIncomeRanges for easier visualization
incomeRange = pd.interval_range(
    start = 0, 
    freq = 2500, 
    end = 25000
)
df_clean['MonthlyIncomeRanges'] = pd.cut(df_clean['MonthlyIncome'], bins = incomeRange)


# In[ ]:


# Explorning the NumberOfOpenCreditLinesAndLoans by income groups for both categories of the target variable
plt.figure(figsize = (16,8))
a = sns.boxplot(
    y = "DebtRatio",
    x = "MonthlyIncomeRanges",
    hue ="SeriousDlqin2yrs",  
    data = df_clean
)
plt.setp(
    a.get_xticklabels(), 
    rotation = 55
);


# For all income groups, average <code>DebtRatio</code> of defaulted users is higher than non-defaulted users, with the difference becoming bigger as the income increases

# In[ ]:


sns.jointplot(
    "MonthlyIncome",
    "NumberRealEstateLoansOrLines",
    data = df_clean.sample(n = 3000),
    kind = 'kde'
);


# In[ ]:


# Checking medium and strong correlations in preparation for ML 
corr = df_clean.corr()
plt.subplots(figsize = (11, 9))
sns.heatmap(
    corr[(corr >= 0.25) | (corr <= -0.25)], 
    cmap = 'viridis', 
    vmax = 1.0, 
    vmin = -1.0, 
    linewidths = 0.1,
    annot = True, 
    annot_kws = {"size": 10}, 
    square = True
);


# # Step 3: ML

# In[ ]:


# ML environment for Random Forest with Random Search optimization 

from numpy import arange
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import itertools

# Set a random seed
seed = 87

# Define evaluation function (source: https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/Random%20Forest%20Tutorial.ipynb)
def evaluate_model(predictions, probs, train_predictions, train_probs):
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 12
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rt'); plt.ylabel('True Positive Rt'); plt.title('ROC Curves');
    
# Define confusion matrix function (source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
def plot_confusion_matrix(cm,
                          classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Oranges):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (4, 4))
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title, size = 10)
    plt.colorbar(aspect = 3)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, size = 10)
    plt.yticks(tick_marks, classes, size = 10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 12,
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 10)
    plt.xlabel('Predicted label', size = 10)


# #### ML-relevant observations summary:
# * <code>MonthlyIncome</code> has >20% of missing values AND is not a strong predictor (correlation is less than 0.2), so consider removing it from the model

# In[ ]:


# Test data preparations
# a) Dropping MonthlyIncome and id
df_ml = df_clean.drop(
    columns = [
        "MonthlyIncome",
        "id",
        "MonthlyIncomeRanges",
        "ageRange",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents"
    ]
);
# b) Splitting the dataset (30%)
labels = np.array(df_ml.pop('SeriousDlqin2yrs'))
train, test, train_labels, test_labels = train_test_split(
    df_ml, 
    labels, 
    stratify = labels,
    test_size = 0.3, 
    random_state = seed
);
# c) Saving features
features = list(train.columns);


# In[ ]:


train.shape


# In[ ]:


test.shape


# ### Take 1 - RandomForest

# In[ ]:


model = RandomForestClassifier(
    n_estimators = 230, 
    random_state = seed, 
    max_features = 'sqrt',
    n_jobs = -1,
    verbose = 1
)

# Fitting on the Train dataset, predicting
model.fit(train, train_labels)

train_rf_pred = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

rf_pred = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]


# In[ ]:


# Evaluating
evaluate_model(rf_pred, rf_probs, train_rf_pred, train_rf_probs)


# In[ ]:


roc_auc_score(test_labels, rf_probs)


# In[ ]:


cm = confusion_matrix(test_labels, rf_pred)
plot_confusion_matrix(
    cm, 
    classes = ['0', '1'],
    normalize = True,
    title = 'Confusion Matrix'
);
plt.grid(None);


# In[ ]:


print(classification_report(test_labels, rf_pred))


# In[ ]:


# Checking variables importance
rf_model = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
rf_model.head(10)


# ### Take 2 - RandomForest with Random Search optimization

# In[ ]:


# Hyperparameters
params = {
    'n_estimators': np.linspace(100, 210).astype(int),
    'max_depth': [None] + list(np.linspace(4, 24).astype(int)),
    'max_features': ['auto', 'sqrt', None] + list(np.arange(0.2, 0.4)),
    'max_leaf_nodes': [None] + list(np.linspace(16, 48, 80).astype(int)),
    'min_samples_split': [1, 2, 3],
    'bootstrap': [True, False]
}

# Estimator
estimator = RandomForestClassifier(random_state = seed)

# Random search model
rs = RandomizedSearchCV(
    estimator,
    params,
    n_jobs = -1, 
    scoring = 'recall',
    cv = 3, 
    n_iter = 10, 
    verbose = 1,
    random_state = seed
)

# Fitting
rs.fit(train, train_labels)


# In[ ]:


# Predicting
train_rs_pred = rs.predict(train)
train_rs_probs = rs.predict_proba(train)[:, 1]

rs_pred = rs.predict(test)
rs_probs = rs.predict_proba(test)[:, 1]


# In[ ]:


# Evaluating
evaluate_model(rs_pred, rs_probs, train_rs_pred, train_rs_probs)


# In[ ]:


roc_auc_score(test_labels, rs_probs)


# In[ ]:


rs.best_params_


# In[ ]:


cm = confusion_matrix(test_labels, rs_pred)
plot_confusion_matrix(
    cm, 
    classes = ['0', '1'],
    normalize = True,
    title = 'Confusion Matrix'
);
plt.grid(None);


# In[ ]:


print(classification_report(test_labels, rs_pred))


# In[ ]:


# Checking variables importance
rs_model = pd.DataFrame({'feature': features,
                   'importance': rs.best_estimator_.feature_importances_}).\
                    sort_values('importance', ascending = False)
rs_model.head(10)

