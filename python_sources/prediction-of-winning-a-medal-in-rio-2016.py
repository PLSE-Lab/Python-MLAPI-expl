#!/usr/bin/env python
# coding: utf-8

# The aim of this project is to predict winning a medal on 2016 Rio de Janeiro Olympic Games.
# 
# This project was done for "Economics in sport" course on SGH Warsaw School of Economics.

# ## Import packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('fivethirtyeight')
from time import time
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot,plot
init_notebook_mode(connected=True)

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


# ## 0. Get data ready
# 
# Import data and drop NA's, because I don't want to play in imputation of missing data this time.

# In[ ]:


ctr = pd.read_csv(r"../input/olympic-games/countries.csv", encoding='utf-8')
df = pd.read_csv(r"../input/olympic-games/athletes.csv", encoding='utf-8', decimal='.', parse_dates=['dob'], index_col=0).dropna()
df.head()


# ### Get some outlook on the dataset.

# In[ ]:


df.info()


# In[ ]:


df.describe(include='all')


# ## Prepare features
# ### Age
# Transform date of birth to age in years and display histogram.

# In[ ]:


df['Age'] = round(((pd.to_datetime('today') - df['dob']).dt.days) / 365 , 0)
df['Age'].hist()


# In[ ]:


df[df['Age'] < 0].head()


# Looks that in some cases date of birth is wrongly read from csv by pandas (like it was year 2050+). Quick internet research confirms that eg. **Abdelkebir Ouaddar** was born in 1962 instead of 2062. Let's fix this mistake for all athlete's.

# In[ ]:


df['Age'] = np.where(df['Age'] < 0, df['Age'] + 100, df['Age'])
df['Age'].hist(bins=20)


# Looks correct now.

# ### Medal - target variable
# I will add information whether athlete won any medals or not, no matter what colour the medal was. This is going to be my binary target variable.

# In[ ]:


df['medal'] = np.where(df['gold'] + df['silver'] + df['bronze'] > 0, 1, 0)


# ### Country - continent
# There is too many categories in the nationality column, so I will grouby them by continents. To group countries by continents I will need to map them. List of countries with continents: https://gist.github.com/mlisovyi/e8df5c907a8250e14cc1e5933ed53ffd

# In[ ]:


cont = pd.read_excel(r'../input/rio2016/continents.xlsx')
cont.head()


# Join 'continents' and 'countries' data frames. 

# In[ ]:


ctr = ctr.merge(cont[['continent', 'name']],
                left_on='country',
                right_on='name',
                how='left')
ctr.head()


# Let's check if there aren't too many NaN and correct the most significant manually.

# In[ ]:


ctr[ctr['continent'].isna()==True]


# I will correct only few of these countries, as most of them don't win any medals at the Olympics

# In[ ]:


ctr.loc[40, 'continent'] = 'Asia'    # China
ctr.loc[43, 'continent'] = 'Africa'  # Congo
ctr.loc[47, 'continent'] = 'Africa'  # Cote d'Ivoire
ctr.loc[81, 'continent'] = 'Asia'    # Hong Kong
ctr.loc[88, 'continent'] = 'Europe'  # Ireland
ctr.loc[96, 'continent'] = 'Asia'    # South Korea
ctr.loc[129, 'continent'] = 'Europe' # Netherlands


# In[ ]:


df['nationality'].value_counts().head()


# Join information about countries to athlete data frame.

# In[ ]:


df = df.merge(ctr[['code', 'continent', 'population', 'gdp_per_capita']],
              left_on='nationality',
              right_on='code',
              how='left')


# ### IOC joining date
# Import date of joining IOC (International Olimpic Committee) by each country - from Wikipedia.

# In[ ]:


ioc = pd.read_excel(r'../input/rio2016/IOC_joining.xlsx')
ioc.head()


# Calculate seniority of countries in the IOC, which was founded in 1894. 

# In[ ]:


df = df.merge(ioc[['code', 'IOC_year']],
              on='code',
              how='left')
df['IOC_seniority'] = 2019 - df['IOC_year']
df.head()


# ### Preparing final dataset
# Encoding categorical variables with [one-hot encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f).

# In[ ]:


df = pd.concat([df,
                pd.get_dummies(df['sex'], drop_first=False),
                pd.get_dummies(df['continent'], drop_first=False)], axis=1)
df.head()


# Let's drop encoded categorical columns and the rest which is unnecessary.

# In[ ]:


df2 = df.drop(['name', 'dob', 'sex', 'sport', 'continent', 'nationality',
               'code', 'IOC_year', 'gold', 'silver', 'bronze'], axis=1).dropna()
df2.columns


# ## Exploratory Analysis

# ### Distributions

# In[ ]:


plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
sns.scatterplot(x='height', y='weight', data=df, hue='medal')


# In[ ]:


plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
sns.boxplot(x='sex', y='Age', data=df, hue='medal')


# In[ ]:


plt.figure(figsize=(15, 8), facecolor='w', edgecolor='k')
sns.violinplot(x='continent', y='gdp_per_capita', data=df, hue='medal', split=True)


# This beautifully named violin plot shows that most of medals (red peaks) gain countries with high GDP per capita in North America and Oceania, a little lower in Europe and rather low in Africa, Asia and South America. This dependence is not suprising nor informative, as it generally presents GDP per capita distribution among continents. 

# In[ ]:


plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
sns.jointplot(x='IOC_seniority', y='gdp_per_capita', data=df2)


# ### Correlation heatmap.

# In[ ]:


plt.figure(figsize=(6, 6))
corr_matrix2 = round(df2[['medal', 'height', 'weight', 'population', 'gdp_per_capita',
                          'Age', 'IOC_seniority']].corr(), 2)
mask = np.zeros_like(corr_matrix2, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.title('Correlation heatmap')
sns.heatmap(corr_matrix2, center=0, vmin=-1, vmax=1, annot=True, cmap="RdBu_r", mask=mask)


# Correlation heatmap confirms what was presented in the distributions above. Height is clearly correlated with weight. For further usage in modelling one of them should be removed. Also gdp_per_capita is quite correlated with IOC_seniority, but it is less than 70% so I think I will keep them both.

# # 1. Model: Support Vector Classifier

# In order to reach best result I will scale data and resample them using under-sampling to get similar number of 0 and 1 in target 'medal' column. Then I will divide dataset into train and test sets.
# - approach: [Under-sampling](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html) 

# In[ ]:


# 1. Scale data
scaler = StandardScaler()
scaled = scaler.fit_transform(df2.drop(['medal', 'weight'], axis=1))
cols = df2.drop(['medal', 'weight'], axis=1).columns

# 2. Divide into X and y sets
X = pd.DataFrame(scaled, columns=cols)
y = df2['medal']

# 3. Resample data 
RUS = RandomUnderSampler(random_state=765, sampling_strategy=0.75) # because first Olympics are dated to 765 BC
X_res, y_res = RUS.fit_resample(X, y)
X2 = pd.DataFrame(X_res, columns=X.columns)
y2 = y_res

# 4. Divide into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.30, random_state=765)


# Let's find best parameters of SVM and fit model.

# In[ ]:


model = SVC()

param_grid = {'C': [0.1,1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
              'kernel': ['rbf']} 

grid = GridSearchCV(estimator = SVC(), param_grid = param_grid, refit=True, verbose=2,
                    scoring='precision', n_jobs=4, iid=False, cv=5)
grid.fit(X_train, y_train)


# In[ ]:


print(grid.best_params_)
print(grid.best_estimator_)


# - Predict values for test set

# In[ ]:


y_pred = grid.best_estimator_.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification stats:\n", classification_report(y_test, y_pred))


# - 65% accuracy is not bad, but let's try do better with another model

# # 2. Model: Gradient Boosting Classifier
# 
# I will procede on the same train and test datasets as above.

# In[ ]:


GBC = GradientBoostingClassifier(verbose=0)

param_grid_2 = {'n_estimators': range(50, 100, 10),
               'max_depth': [3, 4, 5, 6],
               'min_samples_split': range(30, 60, 10), 
               'min_samples_leaf': range(20, 50, 10)}

grid_2 = GridSearchCV(estimator = GBC, refit=True, verbose=2,
                      param_grid = param_grid_2,
                      scoring='precision', n_jobs=4, iid=False, cv=5)

grid_2.fit(X_train, y_train)


# In[ ]:


print(grid_2.best_params_)
print(grid_2.best_score_)


# In[ ]:


y_pred2 = grid_2.best_estimator_.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred2))
print("\nClassification stats:\n", classification_report(y_test, y_pred2))


# - 70% is better. Now let's inspect what are the most impactful features.

# In[ ]:


def report_top_feat(data_set, features, top = 15):
    indices = np.argsort(features)[::-1]
    for f in range(top):
        print("%d. %s (%f)" % (f + 1, data_set.columns[indices[f]], features[indices[f]]))
    
    indices=indices[:top]
    plt.figure(figsize=[8, 6])
    plt.title("Top features for scoring a medal")
    plt.bar(range(top), features[indices], color="r", align="center")
    plt.xticks(range(top), data_set.columns[indices], fontsize=14, rotation=90)
    plt.xlim([-1, top])
#     plt.savefig("Top %d Feature importances for %s.png" % (top, c))
    plt.show()
    print("Mean Feature Importance %.6f" %np.mean(features), '\n')


features = grid_2.best_estimator_.feature_importances_
report_top_feat(X_test, features, X_test.columns.size)


# Interesting but intuitive - big and rich countries have more medal winners than the poor ones. Out of features related to athlete himself height and Age are quite important, which is also reasonable. 

# # 3. Model: Gradient Boosting Classifier with over-sampling
# 
# Let's try the same model just using different resampling method.

# In[ ]:


sm = SMOTE(random_state=765, sampling_strategy='minority')
X_res, y_res = sm.fit_resample(X, y)
X2 = pd.DataFrame(X_res, columns=X_train.columns)
y2 = y_res

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.30, random_state=765)


# In[ ]:


X2.shape, X_train.shape


# In[ ]:


GBC_2 = GradientBoostingClassifier(verbose=0)

param_grid_3 = {'n_estimators': range(50, 100, 10),
               'max_depth': [3, 4, 5, 6],
               'min_samples_split': range(30, 60, 10), 
               'min_samples_leaf': range(20, 50, 10)}

grid_3 = GridSearchCV(estimator = GBC_2, refit=True, verbose=2,
                      param_grid = param_grid_3,
                      scoring='precision', n_jobs=4, iid=False, cv=5)

grid_3.fit(X_train, y_train)


# In[ ]:


print(grid_3.best_params_)
print(grid_3.best_score_)


# In[ ]:


y_pred3 = grid_3.best_estimator_.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred3))
print("\nClassification stats:\n", classification_report(y_test, y_pred3))


# - 87% is way better! 

# In[ ]:


features = grid_3.best_estimator_.feature_importances_
report_top_feat(X_test, features, X_test.columns.size)


# This model shows Age as the most important feature, which is interesting but I leave it to another analysis to inspect why is that. 
