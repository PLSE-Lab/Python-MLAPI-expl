#!/usr/bin/env python
# coding: utf-8

# # What Causes an Accident to Involve Multiple Fatalities?

# # Contents
# 
# 1. [Introduction ](#section1)
# 2. [The Data](#section2)
# 3. [The Model](#section3)
# 4. [The Explanation](#section4)
# 5. [Conclusion](#section5)

# <a id='section1'></a>

# # Introduction
# 
# ### Computer says what?
# 
# With the rise of any new technology comes the inevitable rise of problems and criticisms. With machine-learning, it's become an almost mandatory retort in some circles to decry *"yes, but it's a black box!"*. Or, as I once saw a salesman say when talking about a new medical device that uses deep learning, *"no-one knows how it works!".*
# 
# The idea of **machine-learning explainability** is not new, but in my experience, you had to previously plug examples in, see what you got out, and try to infer what was going on. For example, creating a Titanic prediction model, running through a fictional passenger, and seeing the probability of survival that came out.
# 
# Today, a raft of tools and techniques are available to help and (crucially) visualize this process. I won't get into the details, as the [course by Dan Becker on Kaggle Learn](https://www.kaggle.com/learn/machine-learning-explainability) does that infinitely better that I could. Having made my way through that course, I've applied some of these ideas to the Kaggle [US Traffic Fatality Records](https://www.kaggle.com/usdot/nhtsa-traffic-fatalities) dataset. I also found the official shap library GitHub page useful [here](https://github.com/slundberg/shap).
# 
# ### The Dataset
# 
# This dataset (accessible via BigQuery) contains multiple tables, including data on the passengers, drivers, visibility at the time, damage done to the vehicle, etc. I'm going to stick to the main tables of the accident details, which give information on crash characteristics and environmental conditions at the time of the crash. There is one record per crash.
# 
# 
# ### The Question
# 
# At first, I considered putting together a model to see what factors led to a fatal accident, before realising that *all* of the accidents were fatal (the clue is in the title, I guess). So, next I wondered if there were factors that could distinguish accidents involving a single fatality and those that involved multiple fatalities. The emphasis isn't so much on creating an amazing model, but I wanted to see if I could get some predictive power, which I could then explore future with such tools as permutation importance, partial plots and SHAP values.
# 

# <a id='section2'></a>

# # The Data
# 
# First, let's load the relavent libraries, including those for BigQuery,

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery #For BigQuery
from bq_helper import BigQueryHelper #For BigQuery
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values
from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproducibility

pd.options.mode.chained_assignment = None  #hide any pandas warnings


# Let's take a look at the data,

# In[ ]:


us_traffic = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")
us_traffic.head("accident_2015")


# OK, there is a lot of data there, both in terms of the number of accidents and the number of columns. Let's limit it. I'll pick out variables that I think may have an influence on the severity of the accident (which is, in essence, the question I'm asking), and limit to 5000 rows each from the two years available (2015 and 2016). Note that there is a large imbalance between accidents involving 1 fatality and those involving multiple fatalities Therefore, I'm getting 10,000 single-fatality accidents and *all* the multiple-fatality accidents. I've also set some limits on the latitude and longitude, as I noticed some odd values,

# In[ ]:


accidents_query_2015 = """SELECT month_of_crash,
                                 day_of_week,
                                 hour_of_crash,
                                 manner_of_collision_name,
                                 light_condition_name,
                                 land_use_name,
                                 latitude,
                                 longitude,
                                 atmospheric_conditions_1_name,
                                 number_of_drunk_drivers,
                                 number_of_fatalities
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                          WHERE number_of_fatalities = 1
                          AND longitude < 0
                          AND longitude > -140
                          LIMIT 5000
                      """ 

accidents_query_2016 = """SELECT month_of_crash,
                                 day_of_week,
                                 hour_of_crash,
                                 manner_of_collision_name,
                                 light_condition_name,
                                 land_use_name,
                                 latitude,
                                 longitude,
                                 atmospheric_conditions_1_name,
                                 number_of_drunk_drivers,
                                 number_of_fatalities
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                          WHERE number_of_fatalities = 1
                          AND longitude < 0
                          AND longitude > -140
                          LIMIT 5000
                      """ 

accidents_query_multiple_2015 = """SELECT month_of_crash,
                                          day_of_week,
                                          hour_of_crash,
                                          manner_of_collision_name,
                                          light_condition_name,
                                          land_use_name,
                                          latitude,
                                          longitude,
                                          atmospheric_conditions_1_name,
                                          number_of_drunk_drivers,
                                          number_of_fatalities
                                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                                    WHERE number_of_fatalities > 1
                                    AND longitude < 0
                                    AND longitude > -140
                      """ 

accidents_query_multiple_2016 = """SELECT month_of_crash,
                                          day_of_week,
                                          hour_of_crash,
                                          manner_of_collision_name,
                                          light_condition_name,
                                          land_use_name,
                                          latitude,
                                          longitude,
                                          atmospheric_conditions_1_name,
                                          number_of_drunk_drivers,
                                          number_of_fatalities
                                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                                    WHERE number_of_fatalities > 1
                                    AND longitude < 0
                                    AND longitude > -140
                      """ 


# Now, let's merge them into a single dataframe,

# In[ ]:


accidents_2015 = us_traffic.query_to_pandas(accidents_query_2015)
accidents_2015_multiple = us_traffic.query_to_pandas(accidents_query_multiple_2015)

accidents_2016 = us_traffic.query_to_pandas(accidents_query_2016)
accidents_2016_multiple = us_traffic.query_to_pandas(accidents_query_multiple_2016)

frames = [accidents_2015, accidents_2015_multiple, accidents_2016, accidents_2016_multiple]
accidents_all = pd.concat(frames)


# Let's take a quick look and the distribution,

# In[ ]:


accidents_all['number_of_fatalities'].hist()


# I'm going to create a new variable called **'multiple_fatalities'** that is 0 for single fatalities and 1 for anything greater than 1. I'm also going to change the 'number_of_drunk_drivers' into a binary category (any drunk drivers involved or not?), because the number of drunk drivers could give the model a clue as to how many people are involved in the accident, which might give it an unfair advantage,

# In[ ]:


accidents_all['drunk_driver_involved'] = 0
accidents_all['drunk_driver_involved'][accidents_all['number_of_drunk_drivers'] > 1] = 1
accidents_all = accidents_all.drop('number_of_drunk_drivers', 1)

accidents_all['Multiple_fatalities'] = 0
accidents_all['Multiple_fatalities'][accidents_all['number_of_fatalities'] > 1] = 1
accidents_all = accidents_all.drop('number_of_fatalities', 1)


# There are a few variables with unknown values. For example,

# In[ ]:


accidents_all.groupby(['land_use_name']).size()


# I'm going to remove them,

# In[ ]:


accidents_all = accidents_all[accidents_all['hour_of_crash'] != 99]
accidents_all = accidents_all[accidents_all['manner_of_collision_name'] != 'Unknown']
accidents_all = accidents_all[accidents_all['light_condition_name'] != 'Unknown']
accidents_all = accidents_all[accidents_all['atmospheric_conditions_1_name'] != 'Unknown']
accidents_all = accidents_all[accidents_all['land_use_name'] != 'Unknown']


# I'm also going to remove a handful of rows with 'Trafficway Not in State Inventory' as the land use,

# In[ ]:


accidents_all = accidents_all[accidents_all['land_use_name'] != 'Trafficway Not in State Inventory']


# Double-check that the longitude and latitude values are sensible,

# In[ ]:


x = accidents_all['longitude']
y = accidents_all['latitude']

plt.plot(x, y)


# That looks about right.
# 
# The training data now looks like this,

# In[ ]:


accidents_all.head(10)


# Sort out any data types into categorical,

# In[ ]:


accidents_all['month_of_crash'] = accidents_all['month_of_crash'].astype('category')
accidents_all['day_of_week'] = accidents_all['day_of_week'].astype('category')
accidents_all['hour_of_crash'] = accidents_all['hour_of_crash'].astype('category')
accidents_all['drunk_driver_involved'] = accidents_all['drunk_driver_involved'].astype('category')

accidents_all.dtypes


# One-hot encode the data,

# In[ ]:


accidents_all = pd.get_dummies(accidents_all, drop_first=True) #from the reduntant dummy categories


# Now the training data looks like this,

# In[ ]:


accidents_all.head()


# Quick check on the shape,

# In[ ]:


accidents_all.shape


# <a id='section3'></a>

# # The Model
# 
# I'm going to create a fairly basic model (random forest), to see if there is any predictive ability,

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(accidents_all.drop('Multiple_fatalities', 1), accidents_all['Multiple_fatalities'], test_size = .3, random_state=25)


# In[ ]:


model = RandomForestClassifier()
model.fit(X_train, y_train)


# Get the predictions of the test set,

# In[ ]:


y_predict = model.predict(X_test)
y_pred_quant = model.predict_proba(X_test)[:, 1]
y_pred_bin = model.predict(X_test)


# Review with a [confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/),

# In[ ]:


confusion_matrix = confusion_matrix(y_test, y_pred_bin)
confusion_matrix


# Let's also look at the sensitivity and specificity (recall and precision are also often used to evaluate binary classification problems),

# In[ ]:


total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
print('Specificity : ', specificity)


# Finally, let's create a [ROC plot](https://en.wikipedia.org/wiki/Receiver_operating_characteristic),

# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# And check the area-under-the-curve (AUC),

# In[ ]:


auc(fpr, tpr)


# As a rule of thumb, an AUC can be classed as follows,
# 
# - 0.90 - 1.00 = excellent
# - 0.80 - 0.90 = good
# - 0.70 - 0.80 = fair
# - 0.60 - 0.70 = poor
# - 0.50 - 0.60 = fail
# 

# OK, so something is working. The model seems to be able to distinguish between single and multiple-fatality accidents. Now, let's see if we can figure out how it's working.

# <a id='section4'></a>

# # The Explanation

# ## Permutation Importance

# **Permutation importance** is the first tool for understanding a machine-learning model, and involves shuffling individual variables in the validation data (after a model has been fit), and seeing the effect on accuracy. Learn more [here](https://www.kaggle.com/dansbecker/permutation-importance).
# 
# Let's take a look,

# In[ ]:


perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# So, it looks like variables relating to the manner of the accident are the most important. That seems to make sense.
# 
# ## Partial Dependence Plots
# 
# Next, I'm going to use a **Partial Dependence Plot** (learn more [here](https://www.kaggle.com/dansbecker/partial-plots)). These plots vary a single variable in a single row across a range of values and see what effect it has on the outcome. It does this for several rows and plots the average effect. Let's take a look at the front-to-front collision variable, which was near the top of the permutation importance list,

# In[ ]:


X_test_sample = X_test.iloc[:200]

base_features = accidents_all.columns.values.tolist()
base_features.remove('Multiple_fatalities')

feat_name = 'manner_of_collision_name_Front-to-Front'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test_sample, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# We only have two possibilities, 0 and 1 (not front-to-front and front-to-front, respectively). As you can see, if an accident manner is front-to-front, the outcome increases by ~25%
# 
# We can also create 2D plots by selecting two variables. Let's see how the front-to-front variable interacts with Urban land use,

# In[ ]:


inter1  =  pdp.pdp_interact(model=model, dataset=X_test_sample, model_features=base_features, features=['land_use_name_Urban', 'manner_of_collision_name_Front-to-Front'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['land_use_name_Urban', 'manner_of_collision_name_Front-to-Front'], plot_type='contour')
plt.show()


# It looks like the lowest probability is when the accident occurs on urban land and isn't a front-to-front accident (bottom-right corner).

# ## SHAP Values

# These work by showing the influence of the values of every variable in a single row, compared to their baseline values (learn more [here](https://www.kaggle.com/dansbecker/shap-values)). Let's take a look at the overall values,

# In[ ]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)

shap.summary_plot(shap_values[1], X_test_sample, plot_type="bar")


# These looks similar to the permutation importance list.
# 
# One of my favourite plots from all of the explainability tools is the summary plot, which shows the effect from the individual rows,

# In[ ]:


shap.summary_plot(shap_values[1], X_test_sample)


# Some things really jump out here. For example, 
# 
# - The top feature, 'not collision with a motor vehicle', shows that when this is true (red), the probability of multiple fatalities is reduced. That makes a lot of sense
# - Front-to-front collision again. All the cases in blue (corresponding to 'not front-to-front') reduce the impact, and all the cases in red (corresponding to front-to-front) increase the impact
# - The land use, urban vs rural (or *is urban* vs *not urban*), are directly opposite each other, which makes sense
# - Longitude and latitude are jumbled together and don't seem to have a great deal of impact
# - Days of the week 7 is interesting. According to the data dictionary, this is Sunday, and it's showing that accidents on this day tend to increase the impact. Is this people out drinking at the weekend? (perhaps early Sunday morning, i.e. after a Saturday night out?)
# 
# Below is a function (written in one of the ML explainability exercises) that creates another type of plot. This is for an individual accident, and show the factors that increase (red) and decrease (blue) the impact compared to a baseline prediction,

# In[ ]:


def accident_risk_factors(model, accident):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(accident)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], accident)


# Let's look at the first accident,

# In[ ]:


data_for_prediction = X_test.iloc[1,:].astype(float)
accident_risk_factors(model, data_for_prediction)


# Here we see an output of 0.7 compared to a baseline of 0.3221. It looks like the biggest factor increasing the probability is the front-to-front collision.
# 
# Here is another,

# In[ ]:


data_for_prediction = X_test.iloc[5,:].astype(float)
accident_risk_factors(model, data_for_prediction)


# Much lower than baseline. The accident occuring on urban land is the largest factor in reducing the probability.

# We can look at the SHAP values of two variables in the sort of plot below. These are a little limited with this dataset as it's mainly made up of categorical variables. Below shows latitude and front-the-front collisions,

# In[ ]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)

shap.dependence_plot('latitude', shap_values[1], X_test_sample, interaction_index="manner_of_collision_name_Front-to-Front")


# Arguably higher overall SHAP values at the lower and upper latitudes?
# 
# Finally, let's take an ensemble of the individual row-plots and plot them all together (imagine rotating the plots and stacking them horizontally),

# In[ ]:


shap_values = explainer.shap_values(X_train.iloc[:100])
shap.force_plot(explainer.expected_value[1], shap_values[1], X_train.iloc[:100])


# Very cool. Hover over to see the influence of the different variables.

# <a id='section5'></a>

# # Conclusion
# 

# No doubt many people reading this could do a better job at creating a machine-learning model to predict accident severity. However, it seems to be working on some level, and the various explainability tools have highlighted some interesting points.
# 
# Imagine if a hospital could predict the severity of an accident, and consequently plan better, by knowing a few features of the crash scene? An interesting thought!
