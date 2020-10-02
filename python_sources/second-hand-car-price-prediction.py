# Load libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# Read the data
car = pd.read_csv('../input/second-hand-used-cars-data-set-linear-regression/train.csv')

# Read the head.()
car.head()

# Check the shape of the data 
car.shape

# Check the info() of the data
car.info()

# Check initial statistics of the data 
car.describe()

# Exploratory Data Analysis
sns.pairplot(car)

# Compute the correlations 
car.corr()

# Plot a heatmap
sns.heatmap(car.corr(), cmap = 'viridis')

# More km less price
sns.jointplot(x='km', y='current price', data=car)

sns.jointplot(x='km', y='current price', data=car, kind='hex')

# Based on the pairplot and the correlation data the 'km' and 'current price' look to be the most correlated features

# Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership. 
sns.lmplot(x='km', y='current price', data=car)

# Training and Testing Data
X = car[['v.id', 'on road old', 'on road now', 'years', 'km', 'rating',
       'condition', 'economy', 'top speed', 'hp', 'torque']]
y = car['current price']

# Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

# Training the model
from sklearn.linear_model import LinearRegression

# Create an instance of a LinearRegression() model named lm.
lm = LinearRegression()

# Train/fit lm on the training data
lm.fit(X_train, y_train)


# Print out the coefficients of the model
# The coefficients
print('Coefficients: \n', lm.coef_)

# Predict Test Data
predictions = lm.predict(X_test)

# Create a scatterplot of the real test values versus the predicted values
plt.scatter(y_test, predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')

# Evaluating the model
# Compute the metrics
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Residuals
sns.distplot((y_test-predictions),bins=50)

# Conclusion
coeffecients = pd.DataFrame(lm.coef_,X.columns, columns = ['Coeffecient'])
coeffecients