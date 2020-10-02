#!/usr/bin/env python
# coding: utf-8

# # Random Forest
# Learned from [here](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0)
# ## Data Acquisition
# This is the weather data for Seattle, WA from 2016 using NOAA Climate Data Online tool.

# In[ ]:


import pandas as pd
features = pd.read_csv("../input/temperature-data-seattle/temps.csv")


# In[ ]:


features.head(5)


# Data description
# year: 2016 for all
# month: number for month of the year
# day: number for day of the year
# week: day of thwe week as a character string
# temp_2: max temperature 2 days prior
# temp_1: max temperature 1 day prior
# average: historical average max temperature
# actual: max temperature measurement
# friend: your friend's prediction, a random number between 20 below the average and 20 above the average

# ## Identify Anomalies/Missing Data
# There are 348 rows in the data (not 366 days in 2016),so there are several missing days.

# In[ ]:


print(features.shape)


# Summarize statistics to identify anomalies

# In[ ]:


# Descriptive statistics for each column
features.describe()


# There are not any data points that immediately appear as anomalous and no zeros in any of the measurement columns. Another method to veryfing the quality of the data is make basic plots. Often it is easier to spot anomalies in a graph than in numbers.

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


with plt.style.context('ggplot'):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
    axes[0][0].plot(features['actual'])
    axes[0][0].set_title('Max Temp')
    axes[0][1].plot(features['temp_1'])
    axes[0][1].set_title('Previous Max Temp')
    axes[1][0].plot(features['temp_2'])
    axes[1][0].set_title('Prior Two Days Max Temp')
    axes[1][1].plot(features['friend'])
    axes[1][1].set_title('Friend Estimate')
    plt.plot()


# # Data Preparation
# ## One-Hot Encoding
# This step takes categorical variables (e.g., days of the week) and converts it to numerical representation without an arbitrary ordering. Days of the week are intuitive to us because we use them all the time.
# One solution is to put numbers 1-7 for the days, but this might lead to the algorithm placing more importance on Sunday because it has a higher numerical value. Instead, we change the single column of weekdays into seven columns of binary data. this is best illustrated pictorially. Contvering
# <img src='https://miro.medium.com/max/431/1*lw3v5DrfjwlAUXJb-P06IA.png'/>
# 
# To
# <img src='https://miro.medium.com/max/994/1*dYu_qkF2OKwnyS2YPr18iA.png'/>

# In[ ]:


# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)


# In[ ]:


features.head()


# ## Features and Targets and Convert Data to Arrays

# In[ ]:


import numpy as np
# Access the targets
labels = np.array(features['actual'])
# remove targets from the features
# axis 1 refers to the columns
features=features.drop('actual', axis=1)


# In[ ]:


# saving feature names for later use
feature_list = list(features.columns)


# In[ ]:


# Convert to numpy array
features = np.array(features)


# ## Training and Testing Sets

# In[ ]:


# Suing Scikit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training adn testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[ ]:


print(f'Training Features Shape: {train_features.shape}')
print(f'Testing Features Shape: {test_features.shape}')
print(f'Training Labels Shape: {train_labels.shape}')
print(f'Testing Labels Shape: {test_labels.shape}')


# # Establish Baseline
# Before we can make and evaluate predictions, we need to establish a baseline, a sensible measure that we hope to beat with our model. If our model cannot improve upon the baseline, then it will be a failure and we should try a different mode, or admit that machine learning is not right for our problem.
# 
# One simple baseline prediction for our case can be the historical max temperature averages.

# In[ ]:


# The baseline predictions ar eth historical averages
baseline_preds = test_features[:, feature_list.index('average')]
# Baseline errors, and display average baseline error
baseline_errors = abs(test_labels - baseline_preds)

print(f'Average baseline error (MAE): {round(np.mean(baseline_errors), 2)}')


# # Train Model
# We import the random forest regression model from scikit learn, instantitate the model, and fit (scikit-learns name for training) the model on the trasining data. 

# In[ ]:


# Improt the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels)


# # Make Predictions on the Test Set

# In[ ]:


# Use the forests predict method on the test data
predictions = rf.predict(test_features)

# Calcuate the absoulte errors
errors = abs(test_labels - predictions)

# Print out the mean absolute error (MAE)
print(f'Mean Absolute Error: {round(np.mean(errors), 2)}')


# Though it is not very good result, it is 25% better than the baseline

# # Determine Performance Metrics
# To put our predictions in perspective, we can calculate an accuracy using the mean average percentage error substracted from 100%.

# In[ ]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors/test_labels)
# Calcualte and display accuracy
accuracy = 100 - np.mean(mape)
print(f'Accuracy: {round(accuracy, 2)}%.')


# # Improve Model if Necessary
# In the usual machine learning workflow, this would be when start hyperparameter tuning. This is a complicated phrase that means "adjust the settings to improve performance" (The settings are known as hyperparameters to distinguish them from model parameters learned during training). The most common way to do this is simply make a bunch of model with different settings, evaluate them all on the same validations et, and see which one does best. Of cours,e this would be a tedious process to do by hand, and there are automated methods to do this process in Scikit learn. Hyperparameter tuning is often more engineering than theory-based.
# 
# # Interpret Model and Report Results
# At this point, we know our model is good, but it's pretty much a black box. We feed in some Numpy arrays for training, ask it to make a prediction, evaluate the predictions, and see that they are reasonable. The question is: how does this model arrive at the values? There are two approaches to get under the hood of the random forest: first, we can look at a single tree in the forest, and second, we can look at the feature importances of our explanatory variables.
# 
# # Visualizing a Single Decision Tree
# One of the coolest parts of Random Forest implementation in Scikit learn is we can actually examine any of the trees in the forest. We will select one tree, and save the whole tree as an image.
# 
# The following code takes one tree form the forest and saves it as an image.

# In[ ]:


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names=feature_list, rounded=True, precision = 1)


# In[ ]:


# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png')


# In[ ]:


from IPython.display import Image
Image(filename='tree.png')


# We may limit the depth of the trees in the forest to preoduce an understandable image

# In[ ]:


# Limit the depth of the tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]

# Svae thre tree as a png image
export_graphviz(tree_small, out_file='small_tree.dot', feature_names=feature_list, rounded=True, precision=1)

(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png("small_tree.png")


# In[ ]:


Image(filename='small_tree.png')


# Based solely on this tree, we can make a prediction for any new data point. Let's take an example of making a prediction for Wednesday, December 27, 2017. The (actual) variables are: temp_2 = 39, temp_1 = 35, average = 44, a friend = 30.
# 
# We start at the root node and the first answer is True because $temp\_1 \leq 59.5$ so on and so forth. Finally, we come to a leave node with a value of 41.0 as the value of the leaf node.
# 
# An interesting observation is that in the root node, there are only 162 samples, despite there being 261 training data points. This is because each tree in the forest is trained on a random subset of the data points with replacement (called bagging, short for bootstrap aggregating). (We can turn off the sampling with replacement and use all the data points by setting bootstrap = False when making the forest). Random sampling of **data points**, combined with random sampling of a **subset of the features** at each node fo the tree, is why the model is called a `random` forest.
# 
# Furthermore, notice that in our tree, there are only 3 variables we actually used to make a prediction! According to this particular decision tree, the rest of the features are not important for making a prediction. Month of the year, day of the month, and our friends prediction are utterly useless for predicting the maximum temperature tomorrow! The only important information according to our simple tree is the temperature 1 day priior and the historical average, and one other. 

# # Variable Importances
# In order to quantify the usefulness of all the variables in the entire random forest, we can look at the relative importances of the variables. The importances returned in Scikit-learn represent how much including a particular variable improves the prediction. The actual calculation of the importance is beyond the scope of this post, but we can use the numbers to make relative comparisons between variables.
# 
# The code here takes advantage of a number of tricks in the Python language, namely list comprehensive, zip, sorting, and argument unpacking. It's not that important to understand these at the moment, but if you want to become skilled at Python, these are tools you should have in your arsenal!

# In[ ]:


# Get the numerical fature importances
importances = list(rf.feature_importances_)


# In[ ]:


list(zip(feature_list, importances))


# In[ ]:


# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]


# In[ ]:


feature_importances


# In[ ]:


# Sort hte feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)


# In[ ]:


feature_importances


# In[ ]:


# Print out
_ = [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# At the top of the list is temp_1, the max temperature of the day before. This tells us the best predictor of the max temperature for a day is the max temperature of the day before, a rather intuitive finding. The second most important factor is the historical aveage max tempterature, also not that surprising. Your friend turns out to not be very helpful, along with the day of the week, the year, the month, and the temperature 2 days prior. These importances all make sense as we would nto expect the day of the week to be a predictor of maximum temperatures as it has nothing to do with weathre. Moreover, the year is the same for all data points and hence provides us with no infomration for predicting the max temperature.
# 
# In the future implementation of the model, we can remove those variables that have no importance and the performance will not suffer. <font color = red>Additionally, if we are using a different model, say a support vector machine, we could use random forest feature importances as kind of feature selection method. </font> Let's quickly make a randomf orest with only the two most important variables, the max temperature 1 day prior and the historical averager and see how the performance compares.

# In[ ]:


# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators = 1000, random_state=42)


# In[ ]:


# Extract the two most important features
important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

# Train the random forest
rf_most_important.fit(train_important, train_labels)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)

errors = abs(predictions - test_labels)

# Display the performance metrics
print('Mean Absolute Error: ', round(np.mean(errors), 2), ' degrees.')

mape = np.mean(100*(errors/test_labels))
accuracy = 100 - mape

print('Accuracy: ', round(accuracy, 2), '%.')


# This tells us that we actually do not need all the data we collected to make accurate predictions! If we were continue using this model, we could only collect the two variables and achieve nearly the same performance. In a production setting, we would need to weigh the decrease in accuracy versus extra time required to obtain more information. Knowing how to find the right balance between performance and cost is an essential skill for a machine learning engineer and will ultimately depend on the problem!

# # Visualizations
# The first chart I'll make a simple bar plot of the feature importances to illustrate the disparities in the relative significance of the variables. 

# In[ ]:


# Set the style
plt.style.use('fivethirtyeight')
# List of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation='vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')


# Next, we can plot the entire dataset with predictions highlighted. This requires a little data manipulation, but it is not too difficult. We can use this plot to determine if there are any outliers in either the data or our predictions.

# In[ ]:


# Use datetime for creating data objects for plotting
import datetime

# Dates of training values
months = features[:, feature_list.index('month')]
days =features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]


# In[ ]:


true_data = pd.DataFrame(data={'date': dates, 'actual': labels})


# In[ ]:


# Dates of predictions
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})


# In[ ]:


# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='60')
plt.legend()
# Graph labels
plt.xlabel('Date')
plt.ylabel('Maximum temperature (F)')
plt.title('Actual and Predicted Values')


# In[ ]:


# Make the data accessible for plotting
true_data['temp_1'] = features[:, feature_list.index('temp_1')]
true_data['average'] = features[:, feature_list.index('average')]
true_data['friend'] = features[:, feature_list.index('friend')]

# Plot all the data as lines
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual', alpha = 1.0)
plt.plot(true_data['date'], true_data['temp_1'], 'y-', label='temp_1', alpha=1.0)
plt.plot(true_data['date'], true_data['average'], 'y-', label='average', alpha = 0.8)
plt.plot(true_data['date'], true_data['friend'], 'r-', label='friend', alpha =0.3)
plt.legend()
plt.xticks(rotation='60')
plt.xlabel('Date')
plt.ylabel('Maximum Tewmperature (F)')
plt.title('Actual Max Temp and Variables')


# In[ ]:




