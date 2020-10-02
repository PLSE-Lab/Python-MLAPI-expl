# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np                                      # linear algebra
import pandas as pd                                     # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt                         # plotting
from sklearn.linear_model import LinearRegression       # linear regression
from sklearn.model_selection import train_test_split    # can use this for splitting the data into training and testing sets
from sklearn.metrics import mean_squared_error          # mean squared error metric

# Any results you write to the current directory are saved as output.

# Read in the train & test data csv:
train_and_test = pd.read_csv('/kaggle/input/relex-beer-challenge/cerveja_train_and_test_2.csv', parse_dates = ['Data'])
# Divide the data into separate train and test sets:
training_data, testing_data = train_test_split(train_and_test, test_size=0.2)

# TRAINING BEGINS
# Take average temperatures from the training set:
x = training_data['Temperatura Media (C)']
# Take the beer consumption (aka response) from the training set:
y = training_data['Consumo de cerveja (litros)']
x, y = np.array(x).reshape((-1, 1)), np.array(y)

# Fit a linear regression model with one regressor (average temperature):
model = LinearRegression().fit(x, y)
# Now you have a model that you can use to predict beer consumption!

# TESTING BEGINS
# Take average temperatures from the test set:
x_test = testing_data['Temperatura Media (C)']
x_test = np.array(x_test).reshape((-1, 1))
# Predict beer consumption using the model you trained above:
y_test_predicted = model.predict(x_test)
# Take the true, observed values of beer consumption from the test data:
y_test_ground_truth = testing_data['Consumo de cerveja (litros)']

# If you have a type of notebook that supports visualization, 
# you can use this to plot the correct answer (blue) and your prediction of the consumption (red) in the same plot:
plt.plot(range(0, y_test_ground_truth.size), y_test_ground_truth, 'b-', range(0, y_test_predicted.size), y_test_predicted, 'r-')

# Calculate the Mean Squared Error of your prediction:
mse = mean_squared_error(y_test_ground_truth, y_test_predicted)
print('Got this MSE on the test set:')
print(mse)

# Now you have trained and tested one model on one test data set. 
# You might want to train several models and select the one that seems to "perform best" 
# (for example, the one with lowest MSE, or the one whose line plot looks more promising - the question "which model is better" is not simple).
# You might want to test the models on different combinations of training and testing data, which is called cross validation.
# The models you trained have learned something from the training set, but they might have learned too little (under-fitting), or too much (over-fitting).
# Cross-validation helps to select a model that has learned some underlying connections behind the data, 
# but not something that only happens to be present in the small training set, and is not universally true.

# PREDICTING THE FUTURE BEGINS
# Read in the November & December weather and weekend data:
future_data = pd.read_csv('/kaggle/input/relex-beer-challenge/cerveja_predict_input_data.csv', parse_dates = ['Data'])
# Take average temperatures from the input (future) data set:
x_future = future_data['Temperatura Media (C)']
x_future = np.array(x_future).reshape((-1, 1))
# Predict the beer consumption in the future, meaning the November and December:
y_to_submit = model.predict(x_future)

# WRITING PREDICTIONS INTO A CSV FILE BEGINS
# Take dates from the future data set:
future_dates = np.array(future_data['Data'])
# Make a data frame from the dates and predicted beer consumption:
answer_df = pd.DataFrame(y_to_submit, future_dates)
# Make a csv file from the data frame:
answer_df.to_csv('my_predictions.csv', index_label = 'date', header = ['prediction']) 

# Now if you commit your notebook and run it, and go to the latest version of it, you should find "output" tab with the my_predictions.csv file written.
# Download that file and make sure it has predictions for last 2 months of the year, 61 predictions in total.
# Then submit that file to the competition and see how your solution rates!
# If your solution gives good predictions, be prepared to tell other competitors how you got that solution, and what you think is the reason it performs well in this task.
