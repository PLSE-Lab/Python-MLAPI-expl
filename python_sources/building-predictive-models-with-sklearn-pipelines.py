#!/usr/bin/env python
# coding: utf-8

# # Building End to End predictive models with sklearn pipelines
# 
# Sklearn pipelines have been around for a while now. I always thought pipelines as a nifty feature that can take out the back forth scrolling in kernel notebooks between data cleaning, feature engineering and prediction steps, and have a single object/pipe that can take in data and give out predictions. I also found that this was a promiseland, it was not a easy task to create a practical/useful pipeline, that took in a DataFrame and gave out predictions.
# 
# But now, after a revisit to the land of pipelines, I could bet you that pipelines are back in action. No, I'm serious, I bet that it's possible to acheive the most organized and concise code for a problem by adapting pipelines. I'm willing to bet my upvote on it ;)

# I've explained every step along the way, So it should be easy to follow. 
# 
# **What can you expect in this notebook**
# 
# **Beginner** - I've never heard of sklearn pipelines - You are in for a treat. You I'll be introduced to a new/clean/easy coding model to build predictive models
# 
# **Intermediate** - I've dabbled with pipes, haven't used them end to end - I'll introduce you to new techniques that can help you use pipelines better.
# 
# **Advanced** - I've wrote complex pipelines in production - I've made some cool tweaks and wrote a helper library with common transformation which you can evaluate. Let's dicuss best practices

# ## Let's Start afresh...what are pipelines.
# 
# On the outlook solution to any supervised learning problem, is to learn a set of transformations to apply to the training data so that it matches the results closely, and applying the same transformation to the future test data, to get closely matching result. This entire process is the premise of pipelines. Pipelines are objects that help to record a chain of transformer steps applied to training data, and apply the same to the test data.
# 
# 
# ![Imgur](https://i.imgur.com/tDMLxup.png)
# 
# ### What are the benefits of this approach.
# 
# - You are in complete control of the transformations that goes in to your final model. 
# - The whole model right from data preprocessing to regression/classification can be pickled into a single object. No more is the trouble of remembering what features went into my best model.
# - When used right, reduces the lines of code drastically, this is more apparent when you start developing reusable transformers. (Exactly what I've done here)
# - It's data leakage proof. Here is an [interesting read](https://machinelearningmastery.com/data-leakage-machine-learning/) on this topic. This is illustrated in detail in the coming sections.

# ## Now to the problem at hand.
# 
# There are pretty awesome kernels out there that does an amazing job at feature engineering. So without going in depth on that, let's do some basic analysis of our data.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"] = [8,8]


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


plt.plot(train_data['GrLivArea'], train_data['SalePrice'], 'ro')
plt.show()


# ### Filter out the outliers

# In[ ]:


train_data = train_data[train_data['GrLivArea']<4000]


# ### Splitting our train data into 70% train dataset and 30% test dataset

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                        train_data.drop('SalePrice',axis=1),
                                        train_data['SalePrice'],
                                        test_size=0.3,
                                        random_state=0
                                    )


# ## Let's examine the attributes

# In[ ]:


pd.DataFrame(X_train.dtypes.values, columns=['dtype']).reset_index().groupby('dtype').count().plot(kind='bar')


# ### Our data set has a mix of numerical and categorical data. our pipeline should handle both

# ## Let's construct pipelines
# 
# Given below is the overview of our pipeline, We'll have a seperate pipelines to handle numerical data, categorical data, data that can transformed to scores and a seperate one to handle any engineered feature, we might come up with. 

# ![pflow](https://i.imgur.com/fnBNz6O.jpg)

# ## Libraries required
# 
# The base library of our pipelines is [sklearn.pipeline](http://scikit-learn.org/stable/modules/pipeline.html), in addition to that we'll be requiring a few helpers to make life easier.
# 
# ### [sklearn_pandas](https://github.com/scikit-learn-contrib/sklearn-pandas)
# 
# #### Installation 
# 
# `pip install sklearn_pandas`
# 
# sklearn pipelines natively have poor support for pandas dataframe, sklearn_pandas is an effort to bridge the gap between pandas and sklearn. They have an amazingly easy documentation to their api, it took me just an hour to adapt it in my flow, I would highly encourage everyone to give it a try.
#     
#    * **sklearn_pandas.DataFrameMapper** - A Pipeline util that can be used to apply specific set of transformations to column/columns of a DataFrame
#     
#    * **sklearn_pandas.gen_features** - An utility which can be used to map the same transformer steps to multiple columns in the DataFrame
#    
# ### [sklearn_pipeline_utils](https://github.com/gautham20/sklearn_pipeline_utils)
# 
# #### Installation
# 
# `pip install sklearn_pipeline_utils`
# 
# sklearn_pipeline_utils is a library I hacked together. It's set of pipeline transformer that I felt will be reusable across all the pipelines I'll be building in the future. It has transformers that are lacking in the sklearn.preprocessing libraries
# 
# 
# 
#    * **sklearn_pipeline_utils.CustomImputer** - sklearn.preprocessing.Imputer does not work on categorical data, so this is a wrapper to handle any data formats 
#    
#    * **sklearn_pipeline_utils.CustomMapper** - This is a transformer to map the values of a column based on the dictionary we provide. 
# 
# 

# Bonus - Significant parts of these helper libraries are currently under development to be included in sklearn library. For the curious minds - [Heterogenous Feature Union](https://github.com/scikit-learn/scikit-learn/issues/2034), [Categorical Encoder](https://github.com/scikit-learn/scikit-learn/pull/9151), [ColumnTransformer](https://github.com/scikit-learn/scikit-learn/pull/3886). Hoping that these make it to the scikit-learn0.20 stable release, things are going to get a lot better for pipelines

# In[ ]:


#I Can't pip install a library in kernel, hence embedding source of the package
#This can be assumed to be lib code
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas import DataFrameMapper

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', filler='NA'):
        self.strategy = strategy
        self.fill = filler

    def fit(self, X, y=None):
        if self.strategy in ['mean', 'median']:
            if not all([dtype in [np.number, np.int] for dtype in X.dtypes]):
                raise ValueError('dtypes mismatch np.number dtype is required for ' + self.strategy)
        if self.strategy == 'mean':
            self.fill = X.mean()
        elif self.strategy == 'median':
            self.fill = X.median()
        elif self.strategy == 'mode':
            self.fill = X.mode().iloc[0]
        elif self.strategy == 'fill':
            if type(self.fill) is list and type(X) is pd.DataFrame:
                self.fill = dict([(cname, v) for cname, v in zip(X.columns, self.fill)])
        return self

    def transform(self, X, y=None):
        if self.fill is None:
            self.fill = 'NA'
        return X.fillna(self.fill)
    
def CustomMapper(result_column='mapped_col', value_map={}, default=np.nan):
    def mapper(X, result_column, value_map, default):
        def colmapper(col):
            return col.apply(lambda x: value_map.get(x, default))
        mapped_col = X.apply(colmapper).values
        mapped_col_names = [result_column + '_' + str(i) for i in range(mapped_col.shape[1])]
        return pd.DataFrame(mapped_col, columns=[mapped_col_names])
    return FunctionTransformer(
        mapper,
        validate=False,
        kw_args={'result_column': result_column, 'value_map': value_map, 'default': default}
    )


# ## Numerical Features pipeline

# In[ ]:


#numerical features
X_train.select_dtypes([int, float]).columns


# In[ ]:


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import FeatureUnion, make_union
from sklearn_pandas import DataFrameMapper, gen_features
#import sklearn_pipeline_utils as skutils

# Using dataFrameMapper to map the chosen imputer to the columns
# we are imputing the columns that doesn't have missing values also, this is generally 
# a good practice because we are not making any assumptions on the test data

numerical_data_pipeline = DataFrameMapper(
        [
            (['LotFrontage',
              'LotArea',
              'OverallQual',
              'OverallCond', 
              'YearBuilt',
              'YearRemodAdd'],CustomImputer(strategy='median'), {'alias': 'num_data1'}
            ),
            (['BsmtFinSF1',
              'BsmtFinSF2',
              'BsmtUnfSF',
              'GrLivArea',
              '1stFlrSF',
              '2ndFlrSF',
              'BedroomAbvGr',
              'TotRmsAbvGrd',
              'Fireplaces',
              'GarageCars',
              'GarageArea',
              'WoodDeckSF'], CustomImputer(strategy='fill', filler=0), {'alias': 'num_data2'}
            )
        ],input_df=True ,df_out=True)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaled_numerical_pipeline = make_pipeline(
    numerical_data_pipeline,
    StandardScaler(),
    MinMaxScaler()
)


# ### Feeling a little lost?
# 
# Let's look back on what we've done, We've imputed all our numerical features in the **numerical_data_pipeline**
# and reused the same in **scaled_numerical_pipeline** to apply StandardScaler and MinMaxScaler. 
# 
# ### Step by Step review.
# 
# * **numerical_data_pipeline** - A DataFrameMapper was used to apply imputation based on median to a set of columns and impute zero to missing values in some columns, the result of which can be observed below

# In[ ]:


numerical_data_pipeline.fit_transform(X_train).head()


# * This numerical_data_pipeline is the first step in our **scaled_numerical_pipeline**, this means that given the train_data to scaled_numerical_data, the transformations in numberical_data_pipeline is applied to train_data, and result is passed to the next steps of the pipeline, which take care of scaling. the result can be observed below

# In[ ]:


scaled_numerical_pipeline.fit_transform(X_train)[0:2]


# ## Categorical Data Pipeline

# In[ ]:


train_data.select_dtypes('object').columns


# according to the [data](https://ww2.amstat.org/publications/jse/v19n3/decock/datadocumentation.txt) here, most of these attibutes can be converted to scores. The other attributes can be transformed into one hot vectors.

# In[ ]:


from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn_pandas import gen_features

impute_mode_cols = gen_features(
    columns=['MSSubClass', 'MSZoning', 'LotShape', 'LandContour',
             'LotConfig', 'LandSlope', 'Foundation', 'Condition1',
             'Condition2', 'BldgType', 'HouseStyle'],
    classes=[
        {'class':CustomImputer,'strategy':'mode'},
        {'class':LabelBinarizer}
    ]
)

impute_NA_cols = gen_features(
    columns=['Neighborhood', 'SaleType', 'SaleCondition', 'RoofStyle', 'GarageType'],
    classes=[
        {'class':CustomImputer, 'strategy':'fill', 'filler':'NA'},
        {'class':LabelBinarizer}
    ]
)

categorical_data_pipeline = make_union(
    DataFrameMapper(impute_mode_cols, input_df=True, df_out=True),
    DataFrameMapper(impute_NA_cols, input_df=True, df_out=True)
)


# ### what happened?
# 
# * using **gen_features** we mapped a unique CustomImputer and LabelBinarizer to each column in the columns feed, this can be fed into DataFrameMapper, the output of gen_features can be seen below.

# In[ ]:


# we wrote this manually in case of numerical data pipeline, we have used gen_features to genrate this here
# printing the first two, similar transformers mapping is generated for all columns
impute_mode_cols[0:2]


# * the ouput of both impute_mode_cols and impute_NA_cols and concatenated by using **make_union**, this is helper function to the [**FeatureUnion**]() in sklearn, which can be used to concatenate outputs from multiple pipelines/transformers. Let's check out the shape of categorical_data_pipeline

# In[ ]:


categorical_data_pipeline.fit_transform(X_train).shape


# ### Engineered Data Pipeline

# ## Let's assign scores to our categorical data

# In[ ]:


score_map = {
    'Ex' : 5.0, 'Gd' : 4.0,
    'TA' : 3.0,'Av' : 3.0,
    'Fa' : 2.0, 'Po' : 1.0,
    'NA' : 0, 'No' : 1.0,
    'GLQ': 6,'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA':0,
    'Fin' : 3, 'RFn' : 2,
    'Typ' : 6 ,'Min2': 5,
    'Min1': 4, 'Mod' : 3,
    'Maj1': 2, 'Maj2': 1,
    'Sev' : 0, 'Mn' : 2.0,
}

score_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond',
             'FireplaceQu', 'GarageFinish', 'Functional'
            ]

score_data_pipeline = DataFrameMapper([
    (score_cols, [CustomImputer(strategy='fill', filler='NA'),
                  CustomMapper(value_map=score_map, default=0)], {'alias': 'score_col'})
], input_df=True, df_out=True)


# * ### How did we generate scores?
# 
# * **score_data_pipeline** also uses the DataFrameMapper, to all the score_cols we have identified, we are appling **skutils.CustomMapper**, with the **score_map** as values. Let's review our score_data_pipeline output

# In[ ]:


score_data_pipeline.fit_transform(X_train).head()


# ## Let's engineer new features

# ### Is a remodelled house more valuable than a house that has never been remodelled?

# In[ ]:


tdf = train_data.copy()


# In[ ]:


tdf['remod'] = tdf['YearBuilt']!=tdf['YearRemodAdd']
tdf.boxplot(column=['SalePrice'], by=['remod'])


# #### there isn't much difference in the lower ranges, I can see that remodelled house have little edge in upper ranges, so I'll take it

# ### How hot is a recently remodelled house?

# In[ ]:


tdf['recent_remod'] = tdf['YrSold'] == tdf['YearRemodAdd']
tdf.boxplot(column=['SalePrice'], by=['recent_remod'])


# #### Significantly hot :)

# ### Does building a garage after few years make the house more valuable?

# In[ ]:


tdf['garage_remod'] = tdf['YearBuilt'] != tdf['GarageYrBlt']
tdf.boxplot(column=['SalePrice'], by=['garage_remod'])


# #### This is unexpected, why does building a garage after the house is built reduce it's price, I know that this assumption is a stretch, but let me know if there a reason behind it.

# In[ ]:


### How are recently built house treated in the market

tdf['recentbuilt'] = tdf['YrSold']==tdf['YearBuilt']
tdf.boxplot(column=['SalePrice'], by=['recentbuilt'])


# ### As expected, we are using this feature in our model

# ## Building Custom Transformer for Pipelines

# All our engineered features involves comparing if two columns equal or unequal to each other. A single transformer that does this is all we need to engineer these features to our pipeline.
# 
# Let's see how to develop a custom transformer to be used in pipeline.
# 
# A Transformer by definition should support the **fit()/transform()** interface. There are two ways to achieve this,
# 
# * creating a class which inherits from **sklean.base.TransformerMixin** and override fit() and transform()
# * **sklearn.preprocessing.FunctionTransformer** can be used to transform any functions to a transformer
# 
# We I'll be using the **FunctionTransformer** as it's simpler and adequete for our needs. You can check out the other method [here](https://github.com/gautham20/sklearn_pipeline_utils/blob/master/sklearn_pipeline_utils/skutils.py) in defenition of **CustomImputer**

# In[ ]:


from sklearn.preprocessing import FunctionTransformer

def ColumnsEqualityChecker(result_column='equality_col', inverse=False):
    def equalityChecker(X, result_column, inverse=False):
        def roweq(row):
            eq = all(row.values == row.values[0])
            return eq
        eq = X.apply(roweq, axis=1)
        if inverse:
            eq = eq.apply(np.invert)
        return pd.DataFrame(eq.values.astype(int), columns=[result_column])
    return FunctionTransformer(
        equalityChecker,
        validate=False,
        kw_args={'result_column': result_column, 'inverse': inverse}
    )


# **ColumnsEqualityChecker** returns a FunctionTransformer that wraps the function **equalityChecker**. Any arguments to equalityChecker can be passed in kw_args of FunctionTransformer

# In[ ]:


engineered_feature_pipeline = DataFrameMapper([
    (['YearBuilt','YearRemodAdd'], ColumnsEqualityChecker(inverse=True)),
    (['YearRemodAdd', 'YrSold'], ColumnsEqualityChecker()),
    (['YearBuilt', 'GarageYrBlt'], ColumnsEqualityChecker(inverse=True)),
    (['YearBuilt', 'YrSold'], ColumnsEqualityChecker(inverse=True)),
], input_df=True, df_out=True)


# #### Let's have a look the result

# In[ ]:


engineered_feature_pipeline.fit_transform(X_train).head()


# ## Features Pipeline

# All our feature transformations steps now live inside pipelines, namely **scaled_numerical_pipeline**, **categorical_data_pipeline**, **score_data_pipeline** and **engineered_feature_pipeline**. Combining all the features using a FeatureUnion will give the data our model will train on

# In[ ]:


features_pipeline = make_union(scaled_numerical_pipeline, 
                      categorical_data_pipeline, 
                      score_data_pipeline,
                      engineered_feature_pipeline)


# In[ ]:


features_pipeline.fit_transform(X_train).shape


# I'm not making any further changes to features for simplicity, but other Feature selection and Decompositions steps can take place at this point.

# ## Regression Pipeline and Cross Validation

# We are at the final step of applying Regressors to predict the SalePrice with our pipeline. Let's spot check regressors to pick the best one.

# In[ ]:


import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor 

regressors = [
    SVR(),
    SGDRegressor(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    ExtraTreeRegressor(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    xgb.XGBRegressor()
]


# In[ ]:


Regression_pipeline = Pipeline([
    ('features', features_pipeline),
    ('regressor', regressors[0])
])


# In[ ]:


from sklearn.model_selection import cross_validate
from pprint import pprint

for reg in regressors:
    Regression_pipeline.set_params(regressor=reg)
    scores = cross_validate(Regression_pipeline, X_train, y_train, scoring='neg_mean_squared_log_error', cv=10)
    print('----------------------')
    print(str(reg))
    print('----------------------')
    pprint('Leaderboard score - mean log rmse train '+str((-scores['train_score'].mean())**0.5))
    pprint('Leaderboard score - mean log rmse test '+str((-scores['test_score'].mean())**0.5))


# ## Pipelines prevents data leakage
# 
# Let's take this code,
# 
# ``train_data['GrlivArea'].fillna(train_data['GrlivArea'].mean())``
# 
# And train_data is used is 3 fold cross validation, this is a classic example of **Data Leakage**. 
# 
# **why?** - because the while training your model with 2 folds in CV, `GrLivArea` has it's missing values imputed with `mean()` of all GrlivArea values, including values in the 3rd hold out set. Essentially data leaked from your test set to train set.
# 
# **This doesn't happen in pipelines** because all transformations happen as a part of the pipeline. And while CV it gets only the 2 folds data to train, with no knowledge of the test_data.

# #### I'm skipping tuning hyper parameter to keep it consise. That being said hyper parameter tuning is the secret sauce, fork kernel and grid search for the best xgboost params.  

# ## Applying pipeline to Validation data

# We now train the pipeline with 70% of train_data, and predict the prices of validation dataset-30% of the train data we set aside

# In[ ]:


Regression_pipeline.fit(X_train, y_train)


# In[ ]:


y_validation_predict = Regression_pipeline.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_log_error

score = (mean_squared_log_error(y_validation_predict, y_test))**0.5


# In[ ]:


print('Validation Score '+str(score))


# #### Notice how easy it was to apply all the transformations to a new dataset? :D
# 
# The benifits we get by using pipelines are,
# 
# 
# * The original dataset X_train and X_test remains unchanged
# * No intermediate dataframes, I don't have to keep track of multiple dataframes, each with it's own transformations
# * All the transformer objects that has been fitted with the train data, all applied to my test data too. No place for confusions.

# ## Predicting test data

# Now we train the pipeline with entire test set and predict the result for the test dataset. Since we can apply all transformations to test dataset with a single line of code, this step becomes a piece of cake :)

# In[ ]:


X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']


# In[ ]:


Regression_pipeline.fit(X,y)


# In[ ]:


result = Regression_pipeline.predict(test_data)


# In[ ]:


def generate_submission(filename, y_predict):
    df = pd.DataFrame({'Id': range(1461,2920), 'SalePrice': y_predict})
    df.to_csv(filename, index=False)


# In[ ]:


#generate_submission('improved_pipe2.csv', result)


# ## And there you go!! A Practical solution using pipelines from end to end as promised  :)    
# 
# This scratches the surface on what's possible with pipelines, fork the notebook and try to add new features, try writing custom transformations. If you end up writing a transformer that might be useful for the community let's curate that in [sklearn_pipeline_utils](https://github.com/gautham20/sklearn_pipeline_utils), feel free to issue a pull request.
# 
# I'd love to hear your comments on pipelines and any ways to improve this is very much welcome. 
# 
# If you've learned and enjoyed this kernel, **support me with an upvote**
# 
# **Thank You**
