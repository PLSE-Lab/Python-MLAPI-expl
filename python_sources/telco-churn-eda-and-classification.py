#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data-preview" data-toc-modified-id="Data-preview-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data preview</a></span><ul class="toc-item"><li><span><a href="#Loading-in-the-dataset" data-toc-modified-id="Loading-in-the-dataset-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Loading in the dataset</a></span></li></ul></li><li><span><a href="#Data-Preprocessing" data-toc-modified-id="Data-Preprocessing-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Identify-missing-values" data-toc-modified-id="Identify-missing-values-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Identify missing values</a></span></li><li><span><a href="#Encoding-categorical-variables" data-toc-modified-id="Encoding-categorical-variables-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Encoding categorical variables</a></span></li><li><span><a href="#Upsampling-the-positive-&quot;Churn&quot;-instances-to-even-out-the-target-distribution" data-toc-modified-id="Upsampling-the-positive-&quot;Churn&quot;-instances-to-even-out-the-target-distribution-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Upsampling the positive "Churn" instances to even out the target distribution</a></span></li></ul></li><li><span><a href="#Feature-selection-using-decision-trees" data-toc-modified-id="Feature-selection-using-decision-trees-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Feature selection using decision trees</a></span><ul class="toc-item"><li><span><a href="#Splitting-the-dataset" data-toc-modified-id="Splitting-the-dataset-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Splitting the dataset</a></span></li><li><span><a href="#Important-feature-exploration-using-decision-trees" data-toc-modified-id="Important-feature-exploration-using-decision-trees-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Important feature exploration using decision trees</a></span></li></ul></li><li><span><a href="#Summary-data-visualizations" data-toc-modified-id="Summary-data-visualizations-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Summary data visualizations</a></span><ul class="toc-item"><li><span><a href="#Violinplot-for-continuous-vs-discrete" data-toc-modified-id="Violinplot-for-continuous-vs-discrete-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Violinplot for continuous vs discrete</a></span></li><li><span><a href="#Scatterplot-for-continuous-vs-continuous" data-toc-modified-id="Scatterplot-for-continuous-vs-continuous-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Scatterplot for continuous vs continuous</a></span></li></ul></li><li><span><a href="#Classification-using-machine-learning" data-toc-modified-id="Classification-using-machine-learning-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Classification using machine learning</a></span><ul class="toc-item"><li><span><a href="#Standardizing-the-data" data-toc-modified-id="Standardizing-the-data-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Standardizing the data</a></span></li><li><span><a href="#Defining-a-GridSearchCV-wrapper-class" data-toc-modified-id="Defining-a-GridSearchCV-wrapper-class-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Defining a GridSearchCV wrapper class</a></span></li><li><span><a href="#Displaying-the-GridSearchCV-results-using-plotly" data-toc-modified-id="Displaying-the-GridSearchCV-results-using-plotly-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Displaying the GridSearchCV results using plotly</a></span></li><li><span><a href="#Testing-performance-on-the-holdout-set" data-toc-modified-id="Testing-performance-on-the-holdout-set-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Testing performance on the holdout set</a></span></li></ul></li></ul></div>

# # Telco Churn Dataset
# 
# ## Data preview
# 
# ### Loading in the dataset
# 
# This dataset represents a collection of telecom company customers and whether or not they left the company's services (Churn). It contains information about their account such as how long they've been with the company (tenure), the type of contract they have (Month-to-month, One-year, Two-year), and the amount of money they pay every month on their bill (MonthlyCharges).
# 
# Here we load in the dataset from a CSV file and drop the customerID column, since it's irrelevant to our analysis and unlikely to have any relationship to whether a customer churns or not.

# In[ ]:


import pandas as pd
import numpy as np

pd.set_option('display.width', 100)
pd.set_option('precision', 2)

df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv").drop("customerID", axis=1)


# We start off by looking at the first 20 data points to get a sense of what the data looks like.

# In[ ]:


print(df.head(10))


# We also look at the dimensions and data types of the columns. We can see that there are 7043 customers, as well as 20 different variables that define them. Our target variable is "Churn" and right off the bat we can see that SeniorCitizen is of a numerical type despite being a categorical variable (yes/no). 
# 
# Also strange is the fact that TotalCharges is of type "object" when it should be "float64". We can also see that there are no null objects in any of the columns.

# In[ ]:


print(df.shape)
print(df.info())


# Another thing to look at is the relative proportion of positive versus negative samples with respect to the target variable. As we can see, we have almost a 3:1 ratio of No to Yes, so our dataset is fairly imbalanced. We'll correct for this later on by upsampling the positive examples using bootstrap resampling.

# In[ ]:


print(df.groupby('Churn').size())


# ## Data Preprocessing
# 
# ### Identify missing values
# To get a sense of why TotalCharges is of type `object` despite containing `float64` values, we can iterate over all the columns of the dataframe and a summary of the number of whitespace-only elements in each column.

# In[ ]:


cols =  pd.DataFrame({col_name : sum([int(str(elem).isspace() == True)                                  for elem in col]) for col_name, col in df.iteritems()}, index=[0])
cols = cols.rename(index={cols.index[0]: 'blank rows'})
print(cols)


# As we can see, there are 11 white-space only (i.e. blank) rows in the TotalCharges column, which explains why pandas treats it as type `object` instead of `float64.` We can replace those with np.nan so that `isnull()` correctly identifies those entries as being null.

# In[ ]:


# See null values before and after converstion of whitespace-only chars to np.nan
print(df.isnull().values.sum())
df.replace(r'^\s+$', np.nan, regex=True, inplace=True)
print(df.isnull().values.sum())


# Now we can safely convert TotalCharges to `float64` so our models will correctly treat it as a continuous variable instead of a categorical one. Afterwards, we can run `describe()` on the dataframe to see some summary statistics of the numerical columns. Note that SeniorCitizen is still considered numeric since we have not addressed it yet.

# In[ ]:


df['TotalCharges'] = pd.to_numeric(df.TotalCharges)
print(df.describe())


# We can also take a look at the null rows to get an idea of why TotalCharges might be missing, and therefore get a sense of what to do with the data (impute or discard). 

# In[ ]:


null_rows = df[df['TotalCharges'].isnull()]
print(null_rows[['MonthlyCharges', 'tenure', 'TotalCharges', 'Churn']])


# In particular if we look at MonthlyCharges, tenure, TotalCharges, and Churn, we can see that all of these individuals had a tenure of 0, meaning they have not yet been with the company for a whole month, and therefore have not yet been charged any  money. This explains why TotalCharges is missing, and also why none of them have churned yet. At this point we can safely remove these from our model.

# In[ ]:


# Look at the shape before and after to be sure they were removed
print(df.shape)
df = df[df['TotalCharges'].notnull()]
print(df.shape)


# ### Encoding categorical variables
# Now we can move on to encoding the categorical variables for use with machine learning techniques, which require that variables be converted to a from that can be mapped in feature space for techniques like kNN and logistic regression.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df:
    if df[col].dtype == 'object':
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
        elif df[col].nunique() > 2:
            df[col] = df[col].astype('category').cat.codes


# ### Upsampling the positive "Churn" instances to even out the target distribution
# To even out the target distribution and avoid falling victim to the accuracy paradox, we use bootstrap resampling to upsample the minority class to get an even target distribution between "Churn" and "No churn" groups. To do this, we split the dataset according to the target and then resample the minority class to be the same size as the majority class, then combine the two to form a new, balanced dataset.

# In[ ]:


df_maj = df[df['Churn'] == 0]
df_min = df[df['Churn'] == 1]
from sklearn.utils import resample
df_min_ups = resample(df_min, replace=True, n_samples=5163)
print(df_min.shape)
df_ups = pd.concat([df_min_ups, df_maj])
print(df_ups.Churn.value_counts())


# ## Feature selection using decision trees
# 
# ### Splitting the dataset
# Now we split the data into features (X) and target (y), as well as training and test sets for downstream analysis.

# In[ ]:


X = df_ups.iloc[:, :-1].values
y = df_ups.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# We can also write a function to print out performance metrics for the different classifiers we'll try out to save us from repeating code later on with every classifier.

# In[ ]:


from sklearn import metrics

def print_metrics(y_pred, y_test):
    '''
    Prints accuracy, confusion matrix, and classification report for a given set
    of true and predicted values.
    '''
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


# ### Important feature exploration using decision trees
# At this point we can run a decision tree classifier to get a first sense of which attributes might be important, and what sort of accuracy we can expect.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz

dt_clf = DecisionTreeClassifier(max_depth=3)
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print_metrics(dt_pred, y_test)


# We can visualize the resulting tree to get a sense of which are the optimal splits, and therefore which features seem to be essential to good separation of the classes.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as img

col_names = df.columns.values.tolist()[:-1]
dot_data = export_graphviz(dt_clf,
                                feature_names=col_names,
                                out_file=None,
                                filled=True,
                                rounded=True)

graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
graph.write_png('telco_tree.png')
plt.figure(figsize=(20,20))
plt.imshow(img.imread(fname='telco_tree.png'))
plt.show()


# ## Summary data visualizations
# 
# ### Violinplot for continuous vs discrete
# A good plot for visualizing the relationship between a continuous and discrete variable is the violinplot. It combines the summary statistics of a boxplot with a kernel density estimation to give a better sense of where the most data points are concentrated. This help us to see which combinations of features might be important for distinguishing customers who churn from those who don't. 

# In[ ]:


import plotly.figure_factory as ff

data = []
for i in range(3):
    trace = {
        "type": 'violin',
        "x": df['Contract'][df['Churn'] == i],
        "y": df['MonthlyCharges'][df['Churn'] == i],
        "name": 'Churn' if i == 1 else 'No Churn',
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }
    data.append(trace)

fig1 = {
    "data": data,
    "layout": {
        "title": "Violin Plot of Contract vs Monthly Charges",
        "yaxis": {
            "zeroline": False,
            "title": "Monthly Charges ($)",
        },
        "xaxis": {
            "title": "Contract",
            "tickvals": [0, 1, 2],
            "ticktext": ['Month-to-month', 'One-year', 'Two-year']
        }
    }
}

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
iplot(fig1, filename='violin_contract_monchar')


# ### Scatterplot for continuous vs continuous
# We can use a scatterplot to visualize the relative distribution of two continuous variables. This can be further faceted by the target variable to give us an idea of which data ranges best correspond to churning customers. We can see that customers who churn had a shorter tenure and higher monthly charges. It makes sense intuitively that customers who left had a shorter tenure due to leaving, but interestingly those who stayed with the company longer had a higher monthly bill.

# In[ ]:


fig2 = ff.create_facet_grid(
    df,
    x='tenure',
    y='MonthlyCharges',
    facet_col='Churn',
)

iplot(fig2, filename='scatter_tenure_monchar')


# ## Classification using machine learning
# 
# ### Standardizing the data
# Before we can use this data with classifiers such as logistic regression and k-nearest neighbors, we need to standardize it to ensure that variables on a larger scale such as TotalCharges do not dominate the feature space. We do this using the `StandardScaler()` from sklearn.

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_scaled, y, test_size=0.25)


# ### Defining a GridSearchCV wrapper class
# In order to compare different classifiers we can use 5-fold cross-validation in combination with a grid search over several different parameters to simultaneously tune and evaluate several classifiers to get a sense of which one might perform best for this problem, and what performance we can expect. 
# 
# One way to do this is to define a wrapper class that takes in a dictionary of classifiers and a dictionary of parameters, with a `fit()` method that does a grid search for each classifier in turn.

# In[ ]:


from sklearn.model_selection import GridSearchCV

class ClassifierEvaluator:
    '''
    Wrapper class for applying GridSearchCV to multiple classifiers,
    each with their own set of parameters to search through and evaluate.
    '''
    def __init__(self, classifiers, params):
        self.classifiers = classifiers
        self.params = params
        self.grid_search_results = {}

    def fit(self, X, y, cv=5, n_jobs=5, scoring=None, refit=True, return_train_score=False):
        for clf in self.classifiers.keys():
            print("Running grid search for %s." % clf)
            model = self.classifiers[clf]
            params = self.params[clf]
            grid_search = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, scoring=scoring,
                                       refit=refit, return_train_score=return_train_score)
            grid_search.fit(X, y)
            self.grid_search_results[clf] = grid_search
        print("Grid search complete.")

    def gs_cv_results(self):
        def iter_results(clf_name, scores, params):
            stat_dict = {
                'classifier' : clf_name,
                'mean_score': np.mean(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'std_score': np.std(scores),
            }
            return pd.Series({**params, **stat_dict})

        rows = []
        for clf_name, gs_run in self.grid_search_results.items():
            params = gs_run.cv_results_['params']
            scores = []
            for i in range(gs_run.cv):
                key = "split{}_test_score".format(i)
                result = gs_run.cv_results_[key]
                scores.append(result.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for param, score in zip(params, all_scores):
                rows.append((iter_results(clf_name, score, param)))

        df = pd.concat(rows, axis=1, sort=False).T.sort_values(['mean_score'], ascending=False)

        columns = ['classifier', 'mean_score', 'min_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


# With the class defined, we can now import the classifiers we need and create the `models` and `params` dictionaries we'll need for our GridSearchCV. Then we can use the class we've created to perform the grid search using the `fit()` method, then get the results using `gs_cv_results()`.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'KNeighborsClassifier': KNeighborsClassifier(),
    'RandomForestClassifier' : RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
}

params = {
    'KNeighborsClassifier': {'n_neighbors': [1, 3, 5, 10]},
    'RandomForestClassifier': {
                  'n_estimators': [100, 200, 400],
                  'max_depth': [None, 3, 5, 10],
                  'min_samples_split': [2, 5, 10],
                  'max_features': ['sqrt']
    },
    'LogisticRegression': {'C': [1, 10, 100]}
}

evaluator = ClassifierEvaluator(models, params)
evaluator.fit(Xs_train, ys_train)
results = evaluator.gs_cv_results()


# ### Displaying the GridSearchCV results using plotly
# The `ClassifierEvaluator` class has a method called `gs_cv_results()` that returns a dataframe with a summary of each iteration of the grid search, as well as the mean score across the 5-fold cross-validation. The higher the score, the better the classifier performs in terms of accuracy. The `scoring` parameter can be modified to use different scoring metrics instead of the default.
# 
# We can view the resulting dataframe by using plotly to create a table summarizing the results. From the table we can see that the `RandomForestClassifier` seems to perform best.

# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go

trace = go.Table(
    header=dict(values=['classifier', 'mean_score', 'C', 'max_depth', 'max_features',
                        'min_samples_split', 'n_estimators', 'n_neighbors'],
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[results.classifier, results.mean_score,
                       results.C, results.max_depth, results.max_features, results.min_samples_split,
                       results.n_estimators, results.n_neighbors],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))

data = [trace]
iplot(data, filename = 'telco_gridsearchcv_results')


# ### Testing performance on the holdout set
# As a final step, we can create a `RandomForestClassifier()` with the optimal parameters we have selected and then test it out on the holdout test set to see if we still get a good performance. This is to try to counteract overfitting that so often occurs with all classifiers.

# In[ ]:


rf = RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_split=2, n_estimators=100)
rf.fit(Xs_train, ys_train)
y_pred = rf.predict(Xs_test)
print_metrics(y_pred, ys_test)

