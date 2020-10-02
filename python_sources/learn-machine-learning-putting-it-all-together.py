#!/usr/bin/env python
# coding: utf-8

# # Learn Machine Learning - Putting it all Together, Part One
# 
# This is based off of Dan Becker's [Learn Machine Learning](https://www.kaggle.com/dansbecker/learn-machine-learning/) course. Dan covered many useful topics, including:
# 
# 1. Fitting basic ML models, including the popular XGBoost model
# 2. Avoiding overfitting
# 3. Handling missing values (imputing)
# 4. Dealing with categorical or text data (one-hotting)
# 5. Data and model plotting
# 6. Model validation
# 7. Pipelines
# 
# In this kernel, I simply put together everything from that tutorial, without adding unnecessary extras, to create a basic but serious analysis of the house prices data. It may be useful to a few people, if they are bumping into the same issues that I did.
# 
# This kernel however **does not use pipelines** - while pipelines is important, there are also a few issues there that make the implementation not entirely straightforward at first. So for tutorial purposes, this first Kernel skips the pipelines, and I re-factor the code to use pipelines in this next Kernel - [part two](https://www.kaggle.com/euanrichard/learn-machine-learning-putting-it-all-together-2/), and in the final [part three](https://www.kaggle.com/euanrichard/learn-machine-learning-putting-it-all-together-3/).
# 
# *This is my first published attempt at a ML analysis, so I would appreciate kindly any comments or corrections - thanks!*

#  ## Prepare the Data
# 
# First, read in the data, and separate the predictors (X) and the targets (y).

# In[ ]:


import pandas as pd
import numpy as np

# read training data
data = pd.read_csv('../input/train.csv')
# drop values without target saleprice
# (in this case, there are none, but good practice nonetheless)
data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# split targets and predictors
y = data.SalePrice
X = data.drop(['Id','SalePrice'], axis=1)


# ## Visualize the Data
# 
# At this stage, the next key step should be to run some visualizations on our data - to plot as many things as we can, to understand the shape of the data, and to get a feel for it before we throw it through the ML machine. This is also an important step in cutting false values - almost no data set is perfectly behaved, e.g. what if a house was accidentally recorded as costing USD 1,000,000,000,000 instead of USD 100,000 - this would clearly bias our model hugely during training.
# 
# However, since the "house prices" data set has been carefully pruned and prepared for us, and Dan didn't emphasize this aspect during the tutorial, we'll skip it here.

# ## Preprocess the Data
# 
# We learned how to deal with both missing numerical values, and categorical (text) data. To do both, we can simply split the data, process the numerical and text parts separately, then recombine it.
# 
# Let's start with imputing the data (while remembering to make a record of entries with missing values, which may be a useful piece of information in itself).

# In[ ]:


# split numeric and text
X_numeric = X.select_dtypes(exclude=['object'])
X_text = X.select_dtypes(include=['object'])

##### Prepare to impute the numerical data
from sklearn.preprocessing import Imputer
# make a copy of the numerical data
X_imputed = X_numeric.copy()
# make a record of any NaNs by creating extra predictor columns
cols_with_missing = [col for col in X_imputed.columns if X_imputed[col].isnull().any() ]
for col in cols_with_missing:
    X_imputed[col + '_was_missing'] = X_imputed[col].isnull()


# One small issue here is that the SKLearn Imputer doesn't deal correctly with Pandas' DataFrames yet, and simply returns a NumPy array - this throws away the column labels, which will cause problems when we recombine with the text data. A tidy way to fix this is to define a small function that can call the imputer, but re-wrap the results back into a DataFrame.

# In[ ]:


def impute_DataFrame(DF):
    """
    Calls SKLearn's Imputer, but casts the imputed object back
    as a DataFrame rather than a NumPy array
    """
    my_imputer = Imputer()
    columns = DF.columns
    index = DF.index
    DF_imputed = pd.DataFrame(my_imputer.fit_transform(DF))
    DF_imputed.columns = DF.columns
    DF_imputed.index = DF.index
    return DF_imputed

X_imputed = impute_DataFrame(X_imputed)


# Okay, we've imputed the numerical data, now let's one-hot the text data. I left a few print statements in here, so we can peek into some aspects of our data as we run the analysis - these kind of statements are very useful, especially as you're in the process of developing your analysis.

# In[ ]:


##### one hot the text data
# first let's check the cardinalities
cardinalities = [X_text[col].nunique() for col in X_text.columns]
print("One-hot cardinalities per variable:", cardinalities)

# drop high cardinality columns
max_cardinality = 10
high_cardinality_columns = [col for col in X_text.columns
                            if X_text[col].nunique() > max_cardinality]
X_text = X_text.drop(high_cardinality_columns, axis=1)
print("Dropped text columns with more than", max_cardinality, "unique labels:")
print(high_cardinality_columns)

# do the one-hotting (is that a verb?)
X_onehot = pd.get_dummies(X_text, dummy_na=True)


# Rather than throwing the high-cardinality variables away, we could do something slightly more sophisticated - for example, keep them, but reduce their cardinality by combining particularly rare labels into a new "others" label. This could be a useful technique, depending on the dataset, but we skip it for now.
# 
# All we have to do now is to recombine the imputed numerical and one-hotted text data.

# In[ ]:


# recombine the imputed and one-hotted data
X_train = pd.concat([X_imputed, X_onehot], axis=1)
print("Final predictors have shape:", X_train.shape)


# ## Prepare our Model
# 
# As in Dan's tutorial, we'll use the SKLearn wrapper for the XGBoost library.

# In[ ]:


# Define XGBoost model
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=500, learning_rate=0.05)


# We should be careful of not over-fitting our model. As we learned, the easy way to do this with XGBoost is using the *n_estimators* variable.
# 
# However, first we should keep in mind the **metric** our model will be evaluated on for this competition - the *RMSLE*, or Root-Mean-Square-Log-Error. This makes sense as a metric choice, as it means our model is penalized the same for a 10% error on an expensive house as it is for a cheap house (the log part), but is penalized more and more heavily the further its predictions stray from reality as that percentage increases (the square part). It's therefore a lot better than the simple *MAE* metric.
# 
# XGBoost doesn't have this metric built in, so we have to define it in our own function, which can give our model a score by comparing its predictions with the true values. We have to be a bit careful here, as XGBoost provides our function with its predictions as a NumPy array, but the true values as its own internal datatype *DMatrix*, from which we need to extract the numerical values using the strangely-named *get_label()* command! This is the kind of minor headache that can only be solved by looking up the documentation. Luckily, the SKLearn library contains a similar metric *MSLE*, so we can use that as a base and our function is quite simple.

# In[ ]:


# Define error metric function, as required by XGBoost
from sklearn.metrics import mean_squared_log_error
from math import sqrt,log
def metric(a,b):
    b = b.get_label()
    RMSLE = sqrt(mean_squared_log_error(a, b))
    return ("RMSLE", RMSLE)


# Using this metric, we can now rely on our model's self-evaluations, so let's find the sweet spot between fitting and overfitting. Again, we can define a function to do this, to keep our code tidy.

# In[ ]:


# Fit the number of estimators
from sklearn.model_selection import train_test_split
def get_n_estimators(X,y):
    """
    Split the traning data further, into a traning and evaluation set
    Then run XGBoost until we stop seeing an improvement
    and return the optimum number of iterations
    """
    X_1, X_2, y_1, y_2 = train_test_split(X, y, random_state=0)
    my_model.fit(X_1, y_1, eval_set=[(X_2,y_2)],
                            eval_metric=metric,
                            verbose=False, early_stopping_rounds=10)
    return my_model.best_iteration

# Set n_estimators on our model
my_model.n_estimators = int(get_n_estimators(X_train,y))
print("We set n_estimators as",my_model.n_estimators)


# ## Validate and Train our Model
# 
# OK, we now have an XGBoost model which is ready to fit (but not overfit) on our traning data. We can use the useful *cross_val_score* command, which does a heck of a lot of work for one line of code, [as explained by Dan here](https://www.kaggle.com/dansbecker/cross-validation). By the way, we can use SKLearn's built-in *SLE* metric here, which is close enough (alternatively we could have improved our "metric" function that we defined earlier, so that it can also be passed to SKLearn here).

# In[ ]:


# Cross valildate using the metric RMSLE
# this command does a lot in one line! fits and scores 5 folds of the data
from sklearn.model_selection import cross_val_score
print("Cross-training on all of the training data...")
SLE = cross_val_score(my_model, X_train, y, scoring='neg_mean_squared_log_error', cv=5 )
RMSLE = sqrt( - SLE.mean() )
print("Final model score as RMSLE:", RMSLE)


# Since the *cross_val_score* function relies on k-folds of our data, the final fit would not have used all the training data available. In this case, 4/5 of the data are used to fit the model, and 1/5 are used for evauation at each validation step. So it's important to re-fit the model one more time as a last step, to avoid wasting training data.

# In[ ]:


# re-fit on 100% of training data
my_model.fit(X_train, y)


# ## Making Real Predictions
# 
# All that's left is to apply our trained model to the test data, and make our final predictions. Remember that we have to make sure that the test data has the same shape as our training data, so that the model can be applied. First, we have to process it to impute the numerical data, and create one-hot variables, just as we did with our training data.

# In[ ]:


# read final test data
test_data = pd.read_csv('../input/test.csv')

##### Process the test data, exactly as we did with our training data
# split
test_numeric = test_data.select_dtypes(exclude=['object'])
test_text = test_data.select_dtypes(include=['object'])
# impute
test_imputed = test_numeric.copy()
cols_with_missing = [col for col in test_imputed.columns if test_imputed[col].isnull().any() ]
for col in cols_with_missing:
    test_imputed[col + '_was_missing'] = test_imputed[col].isnull()
test_imputed = impute_DataFrame(test_imputed)
# one-hot
high_cardinality_columns = [col for col in test_text.columns
                            if test_text[col].nunique() > max_cardinality]
test_text.drop(high_cardinality_columns, axis=1)
test_onehot = pd.get_dummies(test_text)
# recombine
test_data = pd.concat([test_imputed, test_onehot], axis=1)


# Then, we have to compare our resulting training and test datasets using the "left join" procedure. Any extra variables in the test set are thrown away, and any missing variables are by default *filled with NaNs*.
# 
# **A side note** - We can realize here that XGBoost can deal directly with missing values - so hold on a second... we didn't actually need to impute the data in the first place! In this particular case however, XGBoost's internal methods for traning with NaN data don't seem to do any better than our mean-value imputation (if you like, you can go back and remove the imputer to check this), so I left the imputer code in place. In any case, it would be useful if you wanted to change to a different model for the final step, as unlike XGBoost many models don't work well with NaN data.
# 
# We then proceed to make and save predictions as usual (making sure that our *Id* column is saved as an integer, as it was converted to a float during imputation).

# In[ ]:


# Left-join the test data with the training data
# so that our trained model can be applied
final_train, final_test = X_train.align(test_data, join='left', axis=1)

predictions =  my_model.predict(final_test)

my_submission = pd.DataFrame({'Id': test_data.Id.apply(int), 'SalePrice': predictions})
my_submission.to_csv('submission.csv', index=False)


# ## Final Score
# 
# After submitting to the leaderboard, we get an RMSE score of 0.131, which is only slightly worse than our predicted score of 0.127. This is in the top 50% of models, but keep in mind that this is a tutorial dataset, with many people submitting very simple models. We see that without extra techniques, like feature optimization or parameter tuning, we can't really compete with the more serious models.
# 
# ## Discussion and Improvements
# 
# ### Visualization
# 
# We should have taken some time to visualize our output - for example, check the spread of our results, see what price range our model is most accurate in, check our partial dependence plots, and so on. I can't stress enough the importance of this kind of analysis - without it, it would really be valid to say that all we have done is apply a "black box" model, by hacking together a few function calls! In reality, our job as data scientists is to understand the data, and present our findings to others clearly, accurately, and even beautifully. These days this can be quite easily accomplished with the *Matplotlib* and *seaborn* libraries.
# 
# ### Pipelines
# 
# As Dan mentioned in his tutorial, it's very difficult to do cross-validation correctly without pipelines, and mistakes tend to creep in. We also had to write extra code to re-process the test data just as we processed the training data, which is uneccessary, and again leaves room for mistakes.
# 
# In this model, there are two mistakes that I noticed, but left in on purpose to illustrate some issues.
# 
# 1. We fit the *n_estimators* variable on a different training set, compared to the training sets we used for cross-validation.
# 2. We re-fit the imputer on the test data, rather than using the values that were fitted on the training data.
# 
# In this particular case, probably neither of these mistakes cause major problems, as both the fitted value of *n_estimators* and the mean values calculated by the imputer probably don't change much when they are re-fitted. However, it's possible that these types of errors (not cleanly separating our training & validation data, or not correctly applying the same transformations to the training and test data) could cause major problems. We could end up with a model that gets a great score when self-validating, but will fail completely when applied to test data.
# 
# ### Where to Next
# 
# To avoid these kind of mistakes, we should tidy up our code into a pipeline, [which I do here](https://www.kaggle.com/euanrichard/learn-machine-learning-putting-it-all-together-2/).
# 
# *Thanks for reading - comments and corrections welcome!*
