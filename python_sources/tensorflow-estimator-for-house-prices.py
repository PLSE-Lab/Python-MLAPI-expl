#!/usr/bin/env python
# coding: utf-8

# # Using TensorFlow Estimator for House Prices prediction
# 
# ## lucasmoura
# 
# **April 2018**

# In this notebook, I would like to show how we can use TensorFlow [Estimator](https://www.tensorflow.org/programmers_guide/estimators) API to perform predictions for the House Prices competition.  One of the main benefits of this API is the simplification it provides when dealing with a machine learning pipeline and how TensorFlow deals with this pipeline. First, the Estimator handles the session management internally, meaning that we don't need to manually create a session to run our model when using an Estimator model. Second, the Estimator encapsulates the **training, evaluation and prediction** of a machine lerning pipeline. Usually, we need to code each of this parts separately when we use a model. However, by using a Pre-made Estimator, we don't need to code any of these parts.

# ## Data Pre-processing

# Since the focus of this notebook is on using the Estimator API, I will not discuss any techniques about data pre-processing. However, I do like to point that this is a really important task for almost any machine learning task and is a crucial task for this competition.
# 
# With that said, the pre-processing I have applied is based on some excellent notebooks that I have read on Kaggle, such as:
# 
# * [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
# * [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
# * [A study on Regression applied to the Ames dataset](https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset)
# 
# If you want to better understand why I am doing the pre-processing steps in this notebook or even better understand data pre-processing in general, take a look on the notebooks I have listed and look at other Kaggle notebooks as well. There are a lot of great examples to be found on that area :)

# Finally, let's start to code our model. First, let's import the required libraries that I will use for this example.

# In[1]:


import os

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.special import boxcox1p
from scipy.stats import skew

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[2]:


# Let's load the house prices dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print('Train data shape: {}'.format(train.shape))
print('Test data shape: {}'.format(test.shape))


# Now, we can remove common outliers in this dataset, houses that have a large area but were sold too cheaply.

# In[3]:


train = train.drop(
    train[
        (train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)
    ].index
)

print('New Train data shape: {}'.format(train.shape))


# In[4]:


# Let's save some variables that we will use in the future

test_id = test['Id']
ntrain = train.shape[0]
y_train = train.SalePrice.values


# Now, let's handle the missing values that appear in this dataset. In order to deal with that problem, we will concat the train and test dataset and perform this for them both at the same time.

# In[5]:


dataset = pd.concat((train, test)).reset_index(drop=True)

fill_zero = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
             'GarageYrBlt', 'GarageArea', 'GarageCars']

for column in fill_zero:
    dataset[column] = dataset[column].fillna(0)

fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'MasVnrType', 'MSSubClass', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for column in fill_none:
    dataset[column] = dataset[column].fillna('None')

fill_mode = ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd',
             'SaleType', 'MSZoning']

for column in fill_mode:
    dataset[column] = dataset[column].fillna(
        dataset[column].mode()[0])

dataset["Functional"] = dataset["Functional"].fillna("Typ")

dataset['LotFrontage'] = dataset.groupby(
    'Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))


# Now, let's convert some numeric columns into strings, allowing us to treat these columns as categorical features.

# In[6]:


to_str_columns = ['MSSubClass', 'OverallQual', 'OverallCond', 'YrSold', 'MoSold']

for column in to_str_columns:
    dataset[column] = dataset[column].apply(str)


# We also drop some columns from our dataset.

# In[7]:


dataset = dataset.drop(['Utilities'], axis=1)
dataset = dataset.drop(['SalePrice'], axis=1)
dataset = dataset.drop(['Id'], axis=1)


# Now, lets convert skewed numeric features using the [Box Cox](http://www.statisticshowto.com/box-cox-transformation/) transformation

# In[8]:


numeric_features = dataset.dtypes[dataset.dtypes != "object"].index
skewed_features = dataset[numeric_features].apply(
    lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew': skewed_features})
skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
lam = 0.15
        
for feature in skewed_features:
    dataset[feature] = boxcox1p(dataset[feature], lam)


# Now, let's split our dataset back into train and test datasets and create the train labels:

# In[9]:


train = dataset[:ntrain]
test = dataset[ntrain:]
targets = pd.DataFrame(np.log(y_train))

print('Train data shape after pre-processing: {}'.format(train.shape))
print('Test data shape after pre-processing: {}'.format(test.shape))


# Finally, we can normalize our numeric features, allowing them to have zero mean and unit variance.

# In[10]:


numerical_features = train.select_dtypes(exclude=["object"]).columns

normalizer = StandardScaler()

train.loc[:, numerical_features] = normalizer.fit_transform(
    train.loc[:, numerical_features])
test.loc[:, numerical_features] = normalizer.transform(
    test.loc[:, numerical_features])


# ## Creating our model

# Now that we have pre-processed our dataset, we can finally create our Estimator model. We are going to user the [LinearRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor) Estimator.  In order to create this estimator we need to feed it [feature columns](https://www.tensorflow.org/get_started/feature_columns). As the documentation states, a feature column is an entermediate abstractation between the raw data and the estimator. Therefore, it converts a raw data feature into a format that an Estimator can read and act upon.
# 
# For our model, we are going to use two feature columns:
# 
# * [numeric_column](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column): For continuos numerical features in our dataset.
# * [categorical_column_with_vocabulary_list](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list): For the categorical features in our dataset.
# 
# There are several different types of feature columns that can be used, we can even create feature_columns that combine other feature column, called [crossed_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/crossed_column).
# For all the available feature columns in TensorFlow, take a look at the [official documentation](https://www.tensorflow.org/versions/master/get_started/feature_columns).

# First, lets create a the numeric column.  To create one, we must pass a **key** parameter. Since we are using a pandas dataframe, the key value we will pass is the name of the column. TensorFlow will them be able to find the related column in the DataFrame and convert it to a numeric column. 
# 
# We will create a different numeric column for each numeric feature we have in our dataset. 

# In[11]:


numeric_columns = []

numeric_features = train.select_dtypes(exclude=[np.object])

for feature in numeric_features:
    numeric_columns.append(
        tf.feature_column.numeric_column(
            key=feature
        )
    )

print('Number of numeric features: {}'.format(len(numeric_columns)))


# For the categorical features, we need two separate values, the **key** and the **vocabulary_list**. The **key** variable is again the name of the column in our pandas DataFrame. The **vocabulary_list** is a list containing all the possible values that the **key** variable can have.
# 
# We can create the categorical columns as follows:

# In[12]:


categorical_columns = []
categorical_dict = {}

categorical_features = train.select_dtypes(exclude=[np.number])

for feature in categorical_features:
    categorical_dict[feature] = dataset[feature].unique()

for key, unique_values in categorical_dict.items():
    categorical_columns.append(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key=key,
            vocabulary_list=unique_values
        )
    )

print('Number of categorical features: {}'.format(len(categorical_columns)))

# Verify if we have used all the available features
assert len(categorical_columns) + len(numeric_columns) == train.shape[1]


# Now we can create our Estimator model by combining both numeric and categorical columns. Also, we can add some regularizers in our model. In the example below, I have added a L2 regularization to the model's weights.

# In[13]:


def create_model():
    linear_estimator = tf.estimator.LinearRegressor(
        feature_columns=numeric_columns + categorical_columns,
        optimizer=tf.train.FtrlOptimizer(
                        learning_rate=0.1,
                        l1_regularization_strength=0.0,
                        l2_regularization_strength=3.0)
    )
    
    return linear_estimator

linear_estimator = create_model()


# Once we have created our Estimator, we can now **train, evaluate and predict** using it. To run any of these steps we need to feed the Estimator an input function. This input function must return at maximum two variables, **features**  and **labels**.  **Features**  is dictionary containing a mapping from the feature name to the feature value and **labels** are just the targets used for training the model. For the **predict** function, our input function should not return the **labels** variable.
# 
# Furthermore, the input function should also be responsible for other dataset related tasks, such as batching and shuffling. 
# 
# However, TensorFlow has default input functions for some python data abstractions, such as pandas DataFrames, called [pandas_input_fn](https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/pandas_input_fn). This function has the following initializer:
# 
# ```python
# tf.estimator.inputs.pandas_input_fn(
#     x,
#     y=None,
#     batch_size=128,
#     num_epochs=1,
#     shuffle=None,
#     queue_capacity=1000,
#     num_threads=1,
#     target_column='target'
# )
# ```
# 
# Where **x** is the pandas DataFrame containing the data, **y** the labels (Not necessary for prediction). We have also some really useful control variable as well, such as **batch_size**,  **shuffle** and **num_threads**. We can use this variables to better optimize our input pipeline, which is also an important step for machine learning usage.
# 
# With that said, to train our model, we can the run the following command:

# In[14]:


batch_size = 64

def train_model(linear_estimator, x, y, should_shuffle):
    num_epochs = 25
    
    linear_estimator.train(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=x,
            y=y,
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle=should_shuffle
        )
    )

train_model(linear_estimator, train, targets, should_shuffle=True)


# See how simple it was to train our model !!
# 
# It as simple to evaluate the model as well. First, we must define a metric to use. Since the competition uses the RMSE metrics, this will be our default choice. The metric must return a dict containing a name to identify the metric and how to compute it:

# In[15]:


def rmse(labels, predictions):
    # Casting is used to guarantee that both labels and predictions have the same types.
    return {'rmse': tf.metrics.root_mean_squared_error(
        tf.cast(labels, tf.float32), tf.cast(predictions['predictions'], tf.float32))}


# Now, we must add this metric to our Estimator:

# In[16]:


linear_estimator = tf.contrib.estimator.add_metrics(linear_estimator, rmse)


# Now, we can use the **evaluate** method from our Estimator. Again, we need to pass to it an input function which is similar to the one we used to train the model:

# In[17]:


def evaluate_model(linear_estimator, validation_data, validation_targets, should_shuffle):
    evaluate_dict = linear_estimator.evaluate(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=validation_data,
            y=validation_targets,
            batch_size=batch_size,
            shuffle=should_shuffle
        )
    )
    
    """
    The dict returns values such as the average loss as well, which you can use as well.
    However, for our example, we only need the value of the rmse metric
    """
    return evaluate_dict['rmse']

# We are using the train dataset to validate our model here just for an example
rmse_value = evaluate_model(linear_estimator, train[:100], targets[:100], should_shuffle=False)
print('Rmse metric: {}'.format(rmse_value))


# With that setup complete, it is now easy to use techniques such as k-fold cross validation in our model:

# In[18]:


#Disable TensorFlow logs for running k-fold
tf.logging.set_verbosity(tf.logging.ERROR)

k_fold = KFold(n_splits=5)
all_rmse = []

for index, (train_index, validation_index) in enumerate(k_fold.split(train)):
    print('Running fold {}'.format(index + 1))

    train_data = train.loc[train_index, :]
    train_targets = targets.loc[train_index, :]

    validation_data = train.loc[validation_index, :]
    validation_targets = targets.loc[validation_index, :]

    linear_estimator = create_model()
    linear_estimator = tf.contrib.estimator.add_metrics(linear_estimator, rmse)
    train_model(linear_estimator, train_data, train_targets, should_shuffle=True)
    rmse_value = evaluate_model(linear_estimator, validation_data, validation_targets, should_shuffle=False)
    all_rmse.append(rmse_value)

final_rmse= sum(all_rmse) / len(all_rmse)        
print('K-fold rmse: {}'.format(final_rmse))


# And to generate our final predictions, we can use the **predict** method. Similar to both **train** and **evaluate**, we need to pass an input function to it. It will return a dict similar to the return by the **evaluate** method, but without the metrics values.
# 
# However, before generating our predictions, it is better to train our model with the whole available training data and then generate the predictions.

# In[19]:


#Enable TensorFlow logging
tf.logging.set_verbosity(tf.logging.INFO)

linear_estimator = create_model()
train_model(linear_estimator, train, targets, should_shuffle=True)


# Now we can generate our predictions:

# In[20]:


def model_predict(linear_estimator, data):
    predictions = linear_estimator.predict(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=data,
            batch_size=batch_size,
            shuffle=False
        )
    )
    
    # all of the predictions are returned as a numpy array. We simple transform this into a list
    pred = [prediction['predictions'].item(0) for prediction in predictions]
    return pred

estimator_predictions = model_predict(linear_estimator, test)
print('Number of predictions: {}'.format(len(estimator_predictions)))


# Now we can combine our predictions with the test dataset **Id** to create our submission DataFrame.

# In[21]:


final_predictions = np.exp(estimator_predictions)

submission = pd.DataFrame()
submission['Id'] = test_id
submission['SalePrice'] = final_predictions

print('Submission shape: {}'.format(submission.shape))


# And that ends our guide on how we can use TensorFlow Estimator.  Although using an Estimator for this competition may not be the best approach, given the dataset size, this API is still really powerfull. If we have a large number of data points, that would be different, since we would have the benefits of TensorFlow(GPU computation) with a simple model to implement. Also, Estimators can also be used for [deployment](https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators), which is a great feature to have if you want to use our model for other tasks, and not only for a competition.
# 
# Furthermore, you can also define your own Estimators, which make it easier for you to share your models with other people, (As an example, here is a [project](https://github.com/lucasmoura/movie_critic_stars) I have developed that contains custom Estimators)
# 
# I hope this notebook was useful for you :) If you do like to learn more, try using a Neural Networks, such as the [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor). It will be necessary to adapt the categorical columns for it, but after that you can follow the same setup I have presented here.
# 
# If you want to see this code in a better project structure, please take a look at this [github](https://github.com/lucasmoura/kaggle_competitions) page. Also, if you have any suggestions or doubts for this notebook, let me now in the commentaries :)
