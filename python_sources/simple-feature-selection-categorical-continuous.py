#!/usr/bin/env python
# coding: utf-8

# <font color=blue>
# Updates 2.0: I made a huge mistake in my prior version of kernel by dropping the entire set of BINARY VARIABLES. I have just realized that these are truly likely One-Hot Encoded variables and contain much inferential information; in the end, I did a univariate (pair-wise) linear regression on X and y, ending up getting a dozen statistically significant variables. 
# 
# 
# Updates 1.0: Mutual Information Regression is added to supplement ANOVA in Categorical feature selection, results reveal robust evidence that these three variables, '0f49e0f05', '7bf58da23', 'c16a7d537', are the only ones significant to be considered within the category. 
# </font>
# ## In a nutshell:
# 
# This notebook aims to extract a small number of features from the vast pool of unlabelled variables. 
# 
# It first subsets features into univariate, binary, categorical and continuous. Then it goes on to drop both the univariate and binary (501 in total). For the categorical and continuous, the following are kept.
# > - Categorical: ['0f49e0f05', '7bf58da23', 'c16a7d537']
# > - Continuous: ['fb0f5dbfe', 'eeb9cd3aa', '58e2e02e6', '9fd594eec', '241f0f867', 'b43a7cfd5', 'd6bb78916', '66ace2992', '402b0d650', '58232a6fb', '20aa07010', 'f190486d6', '6eef030c1', '15ace8c9f']
# 
# 
# 
# 
# Enjoy!

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import plotly.offline as pyo
import plotly.graph_objs as go


# In[ ]:


train_df = pd.read_csv('../input/train.csv', index_col=0)
test_df = pd.read_csv('../input/test.csv', index_col=0)


# In[ ]:


target_df = train_df['target']
target_df_log = np.log(target_df)
train_df.drop('target', axis = 1, inplace = True)


# ## Dimension and Data Type
# 1. Target Distribution
# 2. Check Missing Values
# 3. Check Data Types
# 4. Reconcile Data Types btw Train and Test

# In[ ]:


variable_dict = {'Train Set': train_df, 'Test Set': test_df, 'Target': target_df}
for i in ['Train Set', 'Test Set', 'Target']:
    print('The Dimension (row x col) of %s is: ' %i, variable_dict[i].shape)


# #### Distribution of Target Variable
# > Descriptive: 
#   > - Target is nowhere near bar bell shaped and has very extreme fat tails
#   
#   > - The most common values are mutliples of million, e.g. 2M, 10M, etc.
#   
#   > - Large extreme values make predictions harder
# 
# 
# ***
# - ** Based on the magnitude of the value, combined with the business nature of Santander, the transaction amount is likely to be determined more heavily by qualitative characteristics.**

# In[ ]:


target_df.describe()


# In[ ]:


# Sort most common values
from collections import Counter
cnt = Counter(target_df)
cnt.most_common(20)


# In[ ]:


pyo.init_notebook_mode()
data = [go.Histogram(x = target_df)]
pyo.iplot(figure_or_data= data, filename='Histogram_Target')


# #### Check Missing Value
# - Luckily we have no missing values between train and test

# In[ ]:


print('Sum of NAs in Train = ', train_df.isnull().sum(axis = 0).unique())
print('Sum of NAs in Test = ',  test_df.isnull().sum(axis = 0).unique())


# In[ ]:


## Train set contains a portion of int64, most of which are categorical values, i.e.[0, some_other_value].
train_df.dtypes.value_counts()


# In[ ]:


## Test set only contains float64 dtype
test_df.dtypes.value_counts()


# In[ ]:


# So we convert all Train variables to float64
train_df = train_df.astype(dtype = 'float64',copy = True)
train_df.dtypes.value_counts()


# ## Feature Importance: Categorizing variable types before selection
#  > ### Subset features into 'Constant, Binary, Categorical, and Continuous
#  - Constant: only has value zero
#  - Binary: embodies no meaningful binary information
#  - Categorical: univariate ANOVA test to identify features
#  - Continuous: regularized regression, random forest, gradient boosting

# <font color=blue>_ This function filters features with n or less unique values and hence serves to subset feature types _</font>

# In[ ]:


def categorical_filter(df, low_exclusive = 2, high_inclusive = 15):
    """function returns features (col_names) that have unique values
    less than or equal to n_categories
    
    """ 
    list_of_features = []
    for i in df.columns:
        if low_exclusive == high_inclusive:
            if df[i].nunique() <= low_exclusive :
                list_of_features.append(i)
        else:
            if df[i].nunique() <= high_inclusive and df[i].nunique() > low_exclusive :
                list_of_features.append(i)
    return list_of_features


# In[ ]:


category_1_cols = categorical_filter(train_df, 1, 1 )
print('# of Constant Variables = ',len(category_1_cols))


# In[ ]:


category_2_cols = categorical_filter(train_df, 1, 2)
print('# of Binary Varialbes = ',len(category_2_cols))


# In[ ]:


category_15_cols = categorical_filter(train_df, 2, 15)
print('# of Variables less than or equal to 15 categories = ',len(category_15_cols))


# In[ ]:


remainder_cols = categorical_filter(train_df, 15, len(train_df))
print('# of Continuous Variables (with more than 15 categories) = ',len(remainder_cols))


# ### Constant Variable: drop it!
# ***
# - ** there are 256 features, all of which possess single value 0.0 **
# 
# - ** As they do not contain information as to the change in Target, I'd have them truncated **
# 
# 
# *Notice that if you randomly choose a category_1_cols feature from the Test Set, you end up getting a majority of 0.0 (in fact, 99.9% are zeros). So it can be fairly deducted that the non-zero observations are just noise

# In[ ]:


# check if all constants are equal to 0
((train_df[category_1_cols] == 0.0).all()).all()


# In[ ]:


# See how these features cause noise in the Test Set
test_df[category_1_cols[np.random.randint(0,len(category_1_cols))]].value_counts().head(10)


# ### Binary Variable: def one-hot-encoded!
# ***
# - 245 of these.
# 
# - These features all contain only one non-zero value, indicating that they are extremely likely to be one-hot encoded varialbes.

# In[ ]:


n_rows = len(train_df)

train_df_binary = train_df[category_2_cols]


# In[ ]:


count_nonzero = pd.DataFrame(data = np.zeros((2,len(category_2_cols))), index=['zero', 'nonzero'], columns=category_2_cols)
for i in train_df_binary.columns:
    n_zero = train_df_binary[i].value_counts()[0]
    n_nonzero = n_rows - n_zero
    count_nonzero[i].iloc[0] = n_zero
    count_nonzero[i].iloc[1] = n_nonzero
count_nonzero


# In[ ]:


from sklearn.feature_selection import f_regression

f, p_val = f_regression(train_df_binary,target_df_log)


# In[ ]:


f_reg_df = pd.DataFrame(np.array([f, p_val]).T, index = train_df_binary.columns, columns = ['f-statistic', 'p-value'])
binary_stored_features = f_reg_df[f_reg_df['p-value'] < 0.05].sort_values(by = 'f-statistic', ascending = False)
binary_stored_features


# In[ ]:


selected_features_binary = np.array(binary_stored_features.index)

print('Features selected among binary variables: \n', selected_features_binary)


# ### Categorical Variable: 
# 
# > - As prescribed above, I limit the number of unique values of categorical variables to be (2,15]
# > - 15 is an arbitrary cutoff. Later on, a lesser or greater number may make more sense later on
# 
# - ANOVA is used, as a parametric method, to identify if there's 1) linear correlation; 2) difference in variance in each feature's value spectrum
# - Mutual Information is a non-parametric, entropy based method. It identifies **DEPENDENCY** (both linear and non-linear) between Y and each X. 
# - Tree-based method. Random Forest is added here to complement the previous two methods to spot overlaps. 

# In[ ]:


from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


train_df_categorical = train_df[category_15_cols]

# Label encode categories
le=LabelEncoder()

# create a dataframe to store label encoded values
train_df_categorical_le = train_df_categorical.copy()

for i in train_df_categorical.columns:
    le.fit(train_df_categorical[i])
    train_df_categorical_le[i] = le.transform(train_df_categorical[i]).copy()


# #### ANOVA Test
# - Drawback: ANOVA relies on the assumption that distributions of Y will be different across varying values of X. The greater the variance of Y is relative to each categorical value of X, the higher the f-statistic is.
# - It only captures linear correlation, therefore if non-linear relationship exists, we need other tests to complement ANOVA 

# <font color=blue>_ function conducts univariate ANOVA test between each categorical feature and target _</font>

# In[ ]:


def one_way_anova(categorical_data, target_data):
    # create an empty dataframe to store f-statistic and p-value
    stats_df = pd.DataFrame(np.zeros((len(categorical_data.columns), 2)), index = categorical_data.columns, columns = ['f-statistic', 'p-value'])
    
    # merge independent dataframe with target 
    merged_df = categorical_data.merge(pd.DataFrame(target_data, columns = ['target']), left_index=True, right_index=True)
    for i in categorical_data.columns:
        unique_values = categorical_data[i].unique()
        tuple_list = []
        for value in unique_values:
            store_values = merged_df['target'].loc[merged_df[i]==value].values
            tuple_list.append(store_values)
         
        # get stats from f_oneway test
        statistic, pvalue = f_oneway(*tuple_list)
        stats_df.loc[i, 'f-statistic'] = statistic
        stats_df.loc[i, 'p-value'] = pvalue
        
    return stats_df


# In[ ]:


f_test_df = one_way_anova(train_df_categorical_le, target_df_log)
f_top10_features = f_test_df[f_test_df['p-value'] < 0.05].sort_values(by = 'f-statistic', ascending = False).head(10)
f_top10_features


# In[ ]:


sns.heatmap(data = f_top10_features, annot=True )
plt.title('Top 10 Categorical Features -  Correlation w/ Target and log(Target)')


# #### Mutual Information (MI)
# - MI measures the dependence of X to Y. If X and Y are independent, their mutual information is 0; if fully dependent, mutual information is 1. 
# - Unlike ANOVA or f-test, MI does capture non-linear relationship. 

# In[ ]:


mi = mutual_info_regression(train_df_categorical_le, target_df_log, discrete_features = True, 
                             n_neighbors=5, copy=True, random_state=None)
mi_df = pd.DataFrame(mi, index = train_df_categorical.columns, columns = ['mutual_information'])


# In[ ]:


mi_top10_features = mi_df.sort_values(by = 'mutual_information', ascending=False).head(10)
sns.heatmap(data = mi_top10_features, annot=True )
plt.title('Top 10 Categorical Features - Mutual Information Regression - Discrete Features')


# #### Tree-Based Model

# In[ ]:


rf_cat = RandomForestRegressor(n_estimators=100, criterion='mse', max_features='sqrt')
rf_cat.fit(train_df_categorical_le, target_df_log)


# In[ ]:


# Store the top 10 most important features based off rf regressor
rf_cat_feature_importance_df = pd.DataFrame(rf_cat.feature_importances_, train_df_categorical.columns, columns=['Importance_Value'])
rf_cat_top10_features = rf_cat_feature_importance_df.sort_values(by = ['Importance_Value'], ascending=False).head(10)


# In[ ]:


sns.heatmap(data = rf_cat_top10_features, annot=True )
plt.title('Top 10 Categorical Features - Random Forest - Feature Importance Value')


# #### Summary:
# - all 3 methods point to this subset of features: {'0f49e0f05', '7bf58da23', 'c16a7d537'}

# In[ ]:


# Subset of intersection of both f-test and mi-test 

set(f_top10_features.index).intersection(mi_top10_features.index).intersection(rf_cat_top10_features.index)


# In[ ]:


selected_features_categorical = ['0f49e0f05', '7bf58da23', 'c16a7d537']


# In[ ]:


# scatter plot btw target and shortlisted features
index_feature = selected_features_categorical
plt.subplots(3,1,figsize=(5,14))
for i in range(1, 4):
    col = index_feature[i-1]
    plt.subplot(3, 1, i)
    sns.regplot(x=train_df_categorical[col], y = target_df, fit_reg=False)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.title(col)


# ### Continuous Variable: 
# 
# - ** L1 Regularization: Lasso **
# - ** Random Forest **
# - ** Gradient Tree Boosting **

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


# In[ ]:


train_df_continuous = train_df[remainder_cols]

# Standardize X variables
scaler = StandardScaler()
X_train_df_continuous = scaler.fit_transform(train_df_continuous)


# #### Lasso Regression
# - L1 norm tends to produce sparse solutions by forcing weak features to have coefficient values of zero, which is neat for reducting the dimensionality of data
# - To prevent overfitting, we choose a range of alpha to explore the optimal penalty value
# - As alpha increases, the model complexity reduces, namely the degree of overfitting decreases.
# - Overall, only four features stand out to own non-zero coefficients

# In[ ]:


alpha = [0.2, 0.25, 0.275, 0.3, 0.325, 0.35]


# In[ ]:


lasso_feature_coef_df = pd.DataFrame(np.zeros((len(remainder_cols), len(alpha))), index=remainder_cols, columns=alpha)
for a in alpha: 
    lasso = Lasso(alpha=a)
    lasso.fit(X_train_df_continuous, target_df_log)
    
    lasso_feature_coef_df[a] = lasso.coef_


# In[ ]:


lasso_top10_features = lasso_feature_coef_df.reindex(index=lasso_feature_coef_df[0.30].abs().sort_values(ascending = False).index).head(10)

lasso_top10_features


# In[ ]:


selected_features_lasso = lasso_top10_features.index[:4].tolist()
print('Features of significant coefficient include: \n', selected_features_lasso)


# #### Random Forest Regressor

# In[ ]:


rf = RandomForestRegressor(n_estimators=200, criterion='mse', max_features='sqrt')
rf.fit(X_train_df_continuous, target_df_log)


# In[ ]:


# Store the top 30 most important features based off rf regressor
rf_feature_importance_df = pd.DataFrame(rf.feature_importances_, index=remainder_cols, columns=['Importance_Value'])
rf_top30_features = rf_feature_importance_df.sort_values(by = ['Importance_Value'], ascending=False).head(30)


# In[ ]:


ax0 = sns.barplot(x = rf_top30_features.index, y = 'Importance_Value', data=rf_top30_features)
ax0.set_xticklabels(ax0.get_xticklabels(), fontsize = 12, rotation=40, ha="right")
plt.title('Top 30 Features - Random Forest Regression')
plt.show()


# In[ ]:


selected_features_rf = rf_top30_features[rf_top30_features.Importance_Value >= 0.005].index.tolist()

print('Features of high importance value include: \n', selected_features_rf)


# #### Gradient Tree Boosting Regression

# In[ ]:


gbr = GradientBoostingRegressor(loss='ls', n_estimators=200, learning_rate=0.1, 
                                max_depth=8, max_features = 'sqrt',  
                                min_samples_split = 500, random_state=0)
gbr.fit(X_train_df_continuous, target_df_log)


# In[ ]:


gbr_feature_importance_df = pd.DataFrame(gbr.feature_importances_, index=remainder_cols, columns=['Importance_Value'])
gbr_top30_features = gbr_feature_importance_df.sort_values(by = ['Importance_Value'], ascending=False).head(30)


# In[ ]:


ax1 = sns.barplot(x = gbr_top30_features.index, y = 'Importance_Value', data=gbr_top30_features)
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = 12, rotation=40, ha="right")
plt.title('Top 30 Features - Gradient Tree Boosting Regression')
plt.show()


# In[ ]:


selected_features_gbr = gbr_top30_features[gbr_top30_features.Importance_Value >= 0.005].index.tolist()

print('Features of high importance value include: \n', selected_features_gbr)


# In[ ]:


#### Let's merge all features selected from the Continuous section

selected_features_continuous = set(selected_features_rf+selected_features_lasso+selected_features_gbr)

print('Selected features among continuous variables include: \n', selected_features_continuous)


# 

# * * * Work in Process... Stay tuned

# In[ ]:




