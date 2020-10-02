#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import statistics


# In[ ]:


# settitng the notebook parameters

plt.rcParams["figure.figsize"] = (12, 12)
sns.set_style("darkgrid")
warnings.filterwarnings("ignore")
print(os.listdir("../input"))
pd.set_option("display.max_rows", 100)


# In[ ]:


X_train = pd.read_csv('../input/X_train.csv') 
X_train.head()


# In[ ]:


Y_train = pd.read_csv('../input/y_train.csv')
Y_train.head()


# In[ ]:


X_test = pd.read_csv('../input/X_test.csv')
X_test.head()


# ### Predictors
# - **row_id** - The ID for this row.
# - **series_id** - ID number for the measurement series. Foreign key to y_train/sample_submission.
# - **measurement_number** - Measurement number within the series.
# - **orientation** - The orientation channels encode the current angles how the robot is oriented as a quaternion.
# - **angular_velocity** - Angular velocity describes the angle and speed of motion.
# - **linear_acceleration** - Linear acceleration components describe how the speed is changing at different times.
# 
# ### Labels
# - **series_id**: ID number for the measurement series.
# - **group_id**: ID number for all of the measurements taken in a recording session. Provided for the training set only, to enable more cross validation strategies.
# - **surface**: the target for this competition.

# In[ ]:


# creating a dataset, info_df, to check the unique values and null values in every column of the dataset.

info_df = pd.DataFrame({
    "Unique_Count": X_train.nunique(),
    "Null_Count": X_train.isnull().sum()
})

info_df.T


# Each element in `series_id` contains 128 elements, hence it has a uniform distribution. Similarly, `measurement_number` also has uniform distribution. For every one series, there are 128 measurements.

# ## Descriptive Statistical Summary of the Predictors

# In[ ]:


col = X_train.columns
col_drop = ["row_id", "series_id", "measurement_number"]
col_plot = [i for i in col if i not in col_drop]

stats_df = pd.DataFrame({
    "Mean": X_train[col_plot].mean(),
    "Median": X_train[col_plot].median(),
    "Std Dev": X_train[col_plot].std(),
    "Variance": X_train[col_plot].var()
})

stats_df


# In[ ]:


# checking the distribution of the Predictors

fig = plt.figure(figsize = (12, 12))
for i in range(0, len(col_plot)):
    ax = fig.add_subplot(5, 2, i + 1, xticks = [], yticks = [])
    col = col_plot[i]
    sns.distplot(X_train[col])


# In[ ]:


# checking the outliers in predictors

fig = plt.figure(figsize = (12, 12))

for i in range(0, len(col_plot)):
    ax = fig.add_subplot(5, 2, i+1, xticks = [], yticks = [])
    col = col_plot[i]
    sns.boxplot(X_train[col])


# ## Random Error
# Here, we will sample the dataset repeatedly and check if the Predictors in our dataset abides to Central Limit Theorem. By establishing that, we can say that, there won't be any random error (error due to random sampling). 

# In[ ]:


class Central_limit_theorem(object):
    def __init__(self, sample, xlim = 100):
        self.sample = sample
        self.n = len(sample)
        self.xlim = xlim
        
    def resample(self):
        new_sample = np.random.choice(self.sample, self.n, replace = True)
        return new_sample
    
    def sample_stat(self, sample):
        return sample.mean()
    
    def compute_sampling_distribution(self, iteration = 1000):
        stats = [self.sample_stat(self.resample()) for i in range(iteration)]
        return np.array(stats)
    
    def plot_sampling_distribution(self):
        sample_stats = self.compute_sampling_distribution()
        se = sample_stats.std()
        ci = np.percentile(sample_stats, [5, 95])
        
        sns.distplot(sample_stats, color = "red")
        plt.xlabel("sample statistics")
        plt.xlim(self.xlim)
        
        se_str = "SE = " + str(se)
        ci_str = "CI = " + str(ci)
        
        ax = plt.gca()
        plt.text(0.3, 0.95, s = se_str, horizontalalignment = "center", verticalalignment = "center", transform = ax.transAxes)
        plt.text(0.7, 0.95, s = ci_str, horizontalalignment = "center", verticalalignment = "center", transform = ax.transAxes)
        
        plt.show()


# In[ ]:


def random_error(x ,n, xlim):
    x = x.values
    sample = np.random.choice(x, n)
    resampler = Central_limit_theorem(sample, xlim = xlim)
    resampler.plot_sampling_distribution()


# In[ ]:


# range of xlim can be selected from the stats_df above. Generally, value of sample statistic should be included in your range.
plt.figure(figsize = (12, 8))
random_error(X_train["orientation_X"], 100, xlim = [-1, 1])


# In[ ]:


plt.figure(figsize = (12, 8))
random_error(X_train["orientation_Y"], 100, xlim = [-1, 1])


# In[ ]:


plt.figure(figsize = (12, 8))
random_error(X_train["orientation_Z"], 100, xlim = [-1, 1])


# In[ ]:


plt.figure(figsize = (12, 8))
random_error(X_train["orientation_W"], 100, xlim = [-1, 1])


# In[ ]:


plt.figure(figsize = (12, 8))
random_error(X_train["angular_velocity_X"], 100, xlim = [-1, 1])


# In[ ]:


plt.figure(figsize = (12, 8))
random_error(X_train["angular_velocity_Y"], 100, xlim = [-1, 1])


# In[ ]:


plt.figure(figsize = (12, 8))
random_error(X_train["angular_velocity_Z"], 100, xlim = [-1, 1])


# In[ ]:


plt.figure(figsize = (12, 8))
random_error(X_train["linear_acceleration_X"], 100, xlim = [-1, 1])


# In[ ]:


plt.figure(figsize = (12, 8))
random_error(X_train["linear_acceleration_Y"], 100, xlim = [0, 5])


# In[ ]:


plt.figure(figsize = (12, 8))
random_error(X_train["linear_acceleration_Z"], 100, xlim = [-20, 0])


# # Effect Size
# As all are variables, more or less, abided to CLT, we will now check Effect Size of each variable with respect to the target variable. For that, we have to first concatenate the X_train and Y_train.
# 
# The Effect Size, is considered better than p-value, when checking the statistical significance of the variables. Effect size measures either measure the sizes of associations or the sizes of differences. The Effect size of 0.2 is considered to be "smal"l effect. Meaning, effect is trivial in nature and is random in nature. Similarly, effect size of around 0.5 is considered "medium" and effect size larger that 0.75 is considered "large".
# 
# Here, we will be using Cohen's Effect size (d).
# 
# ## Cohen's Effect Size
# 
# There is one other common way to express the difference between distributions. Cohen's $d$ is the difference in means, standardized by dividing by the standard deviation. Here's the math notation:
# 
# $ d = \frac{\bar{x}_1 - \bar{x}_2} s $
# 
# where s is pooled std_dev

# In[ ]:


print("Shape of X_Train = {}".format(X_train.shape))
print("Shape of Y_Train = {}".format(Y_train.shape))


# The shapes of X_train and Y_train differs because data is normalized. Here, `series_id` act as the foreign key to Y_train. 

# In[ ]:


df = pd.merge(X_train, Y_train, on = "series_id", how = "inner")
print("Shape of df = {}".format(df.shape))


# In[ ]:


df.head()


# In[ ]:


surface_df = df.surface.value_counts()
surface_df = pd.DataFrame(surface_df)
surface_df = surface_df.reset_index()
surface_df.columns = ["surface", "value_counts"]

sns.barplot(surface_df.surface, surface_df.value_counts, alpha=0.8)
plt.show()


# There is class imbalance problem here. When training model, it would be better if we go for Stratified KFold.

# In[ ]:


def cohen_effect_size(group1, group2):
    diff = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_var = ((n1*var1) + (n2*var2))/(n1 + n2)
    d = diff/np.sqrt(pooled_var)
    return d


# In[ ]:


def surface_effect_size(predictor, cutoff):
    surface_1 = []
    surface_2 = []
    es = []
    surface_list = list(surface_df.surface)
    for surface in surface_list:
        temp_list = surface_list
        temp_list.remove(surface)
        for surface_ in temp_list:
            d = cohen_effect_size(df[predictor][df.surface == surface], df[predictor][df.surface == surface_])
            if abs(d) > cutoff:
               surface_1.append(surface)
               surface_2.append(surface_)
               es.append(abs(d)) 
    effect_df = pd.DataFrame({
        "surface_01": surface_1,
        "surface_02": surface_2,
        "effect_size": es
    })
    return effect_df


# In[ ]:


orientation_X_effect_df = surface_effect_size("orientation_X", cutoff = 0.75)
orientation_X_effect_df


# In[ ]:


orientation_Y_effect_df = surface_effect_size("orientation_Y", cutoff = 0.75)
orientation_Y_effect_df


# In[ ]:


orientation_Z_effect_df = surface_effect_size("orientation_Z", cutoff = 0.75)
orientation_Z_effect_df


# In[ ]:


angular_velocity_X_effect_df = surface_effect_size("angular_velocity_X", cutoff = 0.25)
angular_velocity_X_effect_df


# In[ ]:


angular_velocity_Y_effect_df = surface_effect_size("angular_velocity_Y", cutoff = 0.5)
angular_velocity_Y_effect_df


# In[ ]:


angular_velocity_Z_effect_df = surface_effect_size("angular_velocity_Z", cutoff = 0.5)
angular_velocity_Z_effect_df


# In[ ]:


linear_acceleration_X_effect_df = surface_effect_size("linear_acceleration_X", cutoff = 0.25)
linear_acceleration_X_effect_df


# In[ ]:


linear_acceleration_Y_effect_df = surface_effect_size("linear_acceleration_Y", cutoff = 0.25)
linear_acceleration_Y_effect_df


# In[ ]:


linear_acceleration_Z_effect_df = surface_effect_size("linear_acceleration_Y", cutoff = 0.25)
linear_acceleration_Z_effect_df


# The Orientation parameters have significant effect size between various surface types. They can be very helpful for model. But, problem comes in the Angular Velocity and Linear Acceleration Parameters. They don't show any significant effect size. We can't drop these variables because they are important from domain perspective. These variable, if to be used in final prediction model, must be feature engineered so we can extract valuable information out of these variables.

# ---
# # Model

# In[ ]:


encoder = LabelEncoder()
df.surface = encoder.fit_transform(df.surface)


# In[ ]:


drops = ["row_id", "series_id", "measurement_number", "group_id"]
use_cols = [c for c in df.columns if c not in drops]

features = list(df[use_cols].columns)
model_df = df[use_cols]

model_df.head()


# In[ ]:


x = model_df.drop(columns = ["surface"])
y = model_df["surface"]

train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[ ]:


xgboost = xgb.XGBClassifier(
    max_depth = 5,
    learning_rate = 0.05,
    n_estimators = 100,
    gamma = 0,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    reg_alpha = 0.005
)


# In[ ]:


xgboost.fit(train_X, train_Y)


# In[ ]:


preds = xgboost.predict(test_X)
accuracy = (preds == test_Y).sum().astype(float) / len(preds)*100
print("XGBoost's prediction accuracy WITH optimal hyperparameters is: %3.2f" % (accuracy))

