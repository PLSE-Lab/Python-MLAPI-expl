#!/usr/bin/env python
# coding: utf-8

# # Introduction
# - This is a short and concise notebook explaining what is Cross Validation and how to use it using SkLearn.
# - We are using <a href='https://www.kaggle.com/dansbecker/melbourne-housing-snapshot'>Melbourne Housing Snapshot</a> which is a snapshot of a dataset created by <a href='https://www.kaggle.com/anthonypino/melbourne-housing-market'>Tony Pino</a>.
# 
# 

# ## The Cross-Validation Procedure
# 
# In cross-validation, we run our modeling process on different subsets of the data to get multiple measures of model quality. For example, we can  have 5 **folds** or experiments.  We divide the data into 5 pieces, each being 20% of the full dataset.  
# 
# ![cross-validation-graphic](https://i.stack.imgur.com/1fXzJ.png)
# 
# 
# We run an experiment called experiment 1 which uses the first fold as a holdout set, and everything else as training data. This gives us a measure of model quality based on a 20% holdout set, much as we got from using the simple train-test split.  
# We then run a second experiment, where we hold out data from the second fold (using everything except the 2nd fold for training the model.) This gives us a second estimate of model quality.
# We repeat this process, using every fold once as the holdout.  Putting this together, 100% of the data is used as a holdout at some point.  
# 
# Returning to our example above from train-test split, if we have 5000 rows of data, we end up with a measure of model quality based on 5000 rows of holdout (even if we don't use all 5000 rows simultaneously.
# 
# ## Trade-offs Between Cross-Validation and Train-Test Split
# Cross-validation gives a more accurate measure of model quality, which is  important if you are making a lot of modeling decisions.  However, it can take more time to run, because it estimates models once for each fold.  So it does more total work.
# Given these tradeoffs, when should you use each approach
# On small datasets, the extra computational burden of running cross-validation isn't a big deal.  These are also the problems where model quality scores would be least reliable with train-test split.  So, if your dataset is smaller, you should run cross-validation.
# 
# For the same reasons, a simple train-test split is sufficient for larger datasets.  It will run faster, and you may have enough data and need to re-use some of it for holdout.
# 
# There's no simple threshold for what constitutes a large vs small dataset.  If your model takes a couple minute or less to run, it's probably worth switching to cross-validation.  If your model takes much longer to run, cross-validation may slow down your workflow more than it's worth.
# 
# Alternatively, you can run cross-validation and see if the scores for each experiment seem close. If each experiment gives the same results, train-test split is probably sufficient.

# # Example

# First we read the data

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


import pandas as pd
data = pd.read_csv('../input/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price


# Then specify a pipeline of our modeling steps (It can be very difficult to do cross-validation properly if you arent't using [pipelines](https://www.kaggle.com/dansbecker/pipelines))

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())


# Finally get the cross-validation scores:

# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)


# You may notice that we specified an argument for *scoring*.  This specifies to what measure of model quality we need to report.  The docs for scikit-learn show a [list of options](http://scikit-learn.org/stable/modules/model_evaluation.html).  
# 
# It is a little surprising that we specify *negative* mean absolute error in this case. Scikit-learn has a convention where all metrics are defined so a high number is better.  Using negatives here allows them to be consistent with that convention, though negative MAE is almost unheard of elsewhere.
# 
# You typically want a single measure of model quality to compare between models.  So we take the average across experiments.

# In[ ]:


print('Mean Absolute Error %2f' %(-1 * scores.mean()))


# # Conclusion
# 
# Using cross-validation gave us much better measures of model quality, with the added benefit of cleaning up our code (no longer need to keep track of separate train and test sets.  So, it's a good win.
# THANK YOU
