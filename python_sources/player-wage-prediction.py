#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re


# # Summary
# In this workook, we aim to make prediction on the players wage based on the features available in the FIFA 2018 dataset.  We take a look at the correlation matrix and will perform dimensionality reduction using PCA. We will also perform cross validation to tune the hyperparameters of the SVR model. The focus will be on utilizing the Support Vector Regression model and how to work with sklearn library to perform parameter tuning.
# 
# The sequence of the content of this workbook is as follows:
#     1. Data Preparation
#     2. Features Correlation Analysis
#     3. Data Preprocessing
#     4. Model Fitting Using All Features (Support Vector Regression)
#     5. SVR Hyperparameter Tuning (Cross Validation)
#     6. Dimensionality Reduction (PCA)
#     7. Modeling Fitting Using Principle Components
#     8. Hyperparameter Tuning
#     9. Conclusion

# In[ ]:


data = pd.read_csv('../input/CompleteDataset.csv', low_memory=False)


# In[ ]:


data = data.drop(data.columns[0], axis = 1)
data.head()


# ## 1. Data Preparation
# Wage (and Value) columns are our dependant variables. They are currently formatted as currency. So I will extract the numerical values using a regular expression. Then I re-arrange the columns to make the dataframe easier to work with. You will see that there are several numeric features that are stored as text. Those will need to be converted to numeric as well. Finally, I bring numerical Wage and Value columns to the front of the dataframe for conveniece.

# In[ ]:


# Extract numeric vales of the wage and value
data['Wage(TEUR)'] = data['Wage'].map(lambda x : re.sub('[^0-9]+', '', x)).astype('float64')
data['Value(MEUR)'] = data['Value'].map(lambda x : re.sub('[^0-9]+', '', x)).astype('float64')


# In[ ]:


reordered_cols = []
personal_cols = []
personal_cols = ['ID', 'Name', 'Photo', 'Club', 'Club Logo', 'Preferred Positions', 'Wage', 'Value',
                 'Nationality', 'Flag']
reordered_cols = personal_cols + [col for col in data if (col not in personal_cols)]
data = data[reordered_cols]


# In[ ]:


country_data = data.iloc[:, 8:].apply(pd.to_numeric, errors='coerce')
price_pred_cols = list(country_data.columns[-2:]) + list(country_data.columns[2:40])
price_pred_data = country_data[price_pred_cols]
price_pred_data.head()


# ## 2. Correlation Analysis
# Let's plot the correlation matrix. To do that. the correlation matrix need to be calculated and that is done but calling the corr() method on the dataframe that we prepared in the previous step.

# In[ ]:


corr = price_pred_data.corr()


# In[ ]:


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(24, 18))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ### Observations
# It is evident that there is high correlation (negative and positive) between several features. For example, one can say that Goal Keeping attributes (features starting with 'GK') are positively correlated to each other and almost negatively correlated to most other features. This high correlation suggest that we could apply dimensionality reduction. We will explore that later in this notebook.

# ## 3. Data Preprocessing
# In this part, we will take a look at the dataset, look for features with missing values and perform feature scaling as some of the features such as 'age' are measured in a different scale.
# 
# When converting the text-numeric features to numeric, we introduced many NaN's to the dataset. We will check for those columns as well as those that originally had missing values.

# In[ ]:


price_pred_data.isnull().any()[price_pred_data.isnull().any()==True]


# ### Note
# You can created a list of column indecis for those features with missing data. Then you could use it when applying the Imputer function. I will apply the Imputer on the entire dataset, which should return the same result (but maybe in longer time).

# In[ ]:


col_name_missing = price_pred_data.isnull().any()[price_pred_data.isnull().any()==True].index
col_index_missing = [price_pred_data.columns.get_loc(x) for x in col_name_missing]
print(col_index_missing)


# Let's create the matrices of independent and dependant variables first.

# In[ ]:


X = price_pred_data.iloc[:, 2:].values
y = price_pred_data.iloc[:, :2].values


# Taking care of missing data

# In[ ]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)


# Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


y_w_train = y_train[:,0]
y_v_train = y_train[:,1]
y_w_test = y_test[:,0]
y_v_test = y_test[:,1]


# ## 4. Model Fitting Using All Features (Support Vector Regression)
# We will use SVR model with radial based kernel function. We will take a look at the R^2 parameter.

# In[ ]:


from sklearn.svm import SVR


# In[ ]:


regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_w_train)
regressor.score(X_test, y_w_test)


# The 0.43 score is a pretty low score. In the next part, we will attempt to improve the regression paramter by tuning some of the hyperparamters.

# ## 5. SVR Hyperparameters Tuning (Cross Validation)
# I will use the gridsearch cross validation feature from sklearn.model_selection library. Due to very long runtime, I will focus only on "rbf" kernel function and will vary eplison and C parameters. One can further tune other parameters to improve model accuracy.
# 
# To do the cross validation, let's first construct our parameters dictionary.

# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = [{'kernel': ['rbf'],
               'epsilon': [0.1, 0.2, 0.5],
               #'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100]
              },
              #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
             ]


# In[ ]:


svr_cv = SVR()
regressor_cv = GridSearchCV(svr_cv, param_grid = parameters)
regressor_cv.fit(X_train, y_w_train)


# This above step takes a relatively long time to finish. And depending on how many paramters you include in the CV, it may vary. After the CV is done, it is nice to take a look at the performance of the model under different settings.
# 
# To do that, we make a datafram out of the cv_results_

# In[ ]:


cv_df = pd.DataFrame(regressor_cv.cv_results_)


# You could compare the time to fit and other information reported in the resutls dataframe. However, for the purpose of this notebook, I will focus on the R^2 (the score). 
# 
# So, let's prepare the data we need to visualize the test and training scores.

# In[ ]:


#'mean_test_score','mean_train_score', 'param_C', 'param_epsilon','rank_test_score'
score_df = cv_df[['rank_test_score','param_C','param_epsilon','mean_test_score','mean_train_score']]
score_df = pd.melt(score_df, id_vars=['rank_test_score','param_C','param_epsilon'],
                   value_vars=['mean_test_score','mean_train_score'],
                   var_name="score_name",
                   value_name="score")


# Now we can look at the best paramaters found:

# In[ ]:


score_df[score_df['rank_test_score']==1]


# In[ ]:


g = sns.FacetGrid(score_df, col="param_C", row="param_epsilon", hue="rank_test_score", margin_titles=True)
g.map(sns.barplot, "score_name", "score")
g.add_legend()
#Rotate x-axis labels
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(10)


# ### Observation
# We can see that increasing both the parameters improved the score, with the C parameter having higher imapct on the score. So out of the paramters tested above, the best model is going to be the one with
# 
#     C=100
#     epsilon=0.5

# ## 6. Dimensionality Reduction
# As discussed earlier, the features are highly correlated. We will now use Principle Component Analysis to see how we could reduce the problem dimension.
# 
# First, let's pass n_components = None to get the full vector for the explained variances.

# In[ ]:


# Principle Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
pca.fit(X_train)
explained_variance = pca.explained_variance_ratio_


# To see how much of overall variance is explained by how many of the principle components, let's plot the cumulative sum of the explained variance by each component.

# In[ ]:


f, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=np.arange(explained_variance.size)+1,
            y=np.cumsum(explained_variance),
            color="b"
)
ax.set(ylabel="Cumulative Explained Variance",
       xlabel="Number of Principal Component")


# About 80% of the variance is explained by 4 Principle Components. That's a pretty good low number of components. Let's go with that and set the n_components equal to 4.

# In[ ]:


pca = PCA(n_components = 4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# ## 7. Model Fitting Using (4) Principle Components
# A similar analysis to part 4 will be performed, this time using only the first 4 principle components.

# In[ ]:


regressor_pca = SVR(kernel='rbf')
regressor_pca.fit(X_train_pca, y_w_train)
regressor_pca.score(X_test_pca, y_w_test)


# Comapring score of the PCA model to the initial model (0.33 vs. 0.43), we see a drop in the accuracy of the model. That's OK, let's see if we can improve the model accuracy with some parameter tuning.

# ## 8. Hyperparameters Tuning (Post PCA)
# We will run the same CV process with an additional C paramter this time, the 1000.

# In[ ]:


parameters = [{'kernel': ['rbf'],
               'epsilon': [0.1, 0.2, 0.5],
               #'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100, 1000]
              },
              #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
             ]


# In[ ]:


svr_pca_cv = SVR()
regressor_pca_cv = GridSearchCV(svr_pca_cv, param_grid = parameters, return_train_score=True)
regressor_pca_cv.fit(X_train_pca, y_w_train)


# Now let's visualize the results of the CV... are you excited already??!!!

# In[ ]:


pca_cv_df = pd.DataFrame(regressor_pca_cv.cv_results_)
pca_score_df = pca_cv_df[['rank_test_score','param_C','param_epsilon','mean_test_score','mean_train_score']]
pca_score_df = pd.melt(pca_score_df, id_vars=['rank_test_score','param_C','param_epsilon'],
                   value_vars=['mean_test_score','mean_train_score'],
                   var_name="score_name",
                   value_name="score")


# Here is a brief summary of the best parameters found with 4 principle components:

# In[ ]:


pca_score_df[pca_score_df['rank_test_score']==1]


# In[ ]:


g = sns.FacetGrid(pca_score_df, col="param_C", row="param_epsilon",
                  hue="rank_test_score", margin_titles=True,
                  palette=(sns.color_palette("coolwarm", 12)))
g.map(sns.barplot, "score_name", "score")
g.add_legend()
#Rotate x-axis labels
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(10)


# It turns out that C=100 and epsilon=0.5 is still the best combination of the paramters. 

# ## 9. Conclustion
# In this notebook, the Support Vector Regression with rbf kernel was utilized in an attempt to make a regression model to predict the Wage of FIFA players based on their age, skills and positions. High correlation among features (some positive and some negative) was observed which led to applying PCA. SVR model was fitted to both full dataset as well as PCA dataset with 4 principle components. In terms of model accuracy, the R^2 value seemed relatively low. Therefore cross validation was performed to tune parameters to improve the model accuracy on both datasets (full and PCA). It was found that the C=100 and epsilon=0.5 are the optimum parameters that will improve the model accuracy. However, the model is still not enough accurate (with score being in the [0.55,0.65] interval).
