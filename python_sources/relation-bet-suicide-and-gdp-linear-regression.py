#Used libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

#Reading the csv file using pandas
df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

#Groupby data to get the sum of suicides/100k pop per country each year
df2 = df.groupby(['country','year', 'gdp_per_capita ($)']).sum().reset_index()

print (df2.head(9))

#printing titles of the columns
print(df2.columns)


#Converting dataframe to Numpy array so it can be used with sklearn
cdf =df2[['country', 'year', 'gdp_per_capita ($)', 'suicides/100k pop']]
print(cdf.head(9))

#Plotting data for visual investigation
cdf.plot.scatter('gdp_per_capita ($)' , 'suicides/100k pop', color='blue')
plt.xlabel = 'gdp_per_capita ($)'
plt_ylabel = 'suicides/100k pop'
plt.show()

#spliting data into train group(80%) and test group(20%)
msk=np.random.rand(len(df2))<.8
train = cdf[msk]
test = cdf[~msk]

#Plotting the train group
train.plot.scatter('gdp_per_capita ($)','suicides/100k pop', color='blue')
plt.xlabel = 'gdp_per_capita ($)'
plt.ylabel = 'suicides/100k pop'
plt.show()

#Generate arrays for training sets
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['gdp_per_capita ($)']])
train_y = np.asanyarray(train[['suicides/100k pop']])
regr.fit(train_x, train_y)


print('Coeffiecient :  ', regr.coef_)
print('Intercept : ', regr.intercept_)

#Plotting the regression line using linear regression equation
train.plot.scatter('gdp_per_capita ($)','suicides/100k pop', color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x+regr.intercept_[0], '-r')
plt.xlabel = 'gdp_per_capita ($)'
plt.ylabel = 'suicides/100k pop'
plt.show()

#Generating arrays for test sets
test_x = np.asanyarray(test[['gdp_per_capita ($)']])
test_y = np.asanyarray(test[['suicides/100k pop']])

#Predicion of test set
test_y_hat = regr.predict(test_x)

#Model Evaluation
print('Mean absolute error : %0.2f' % np.mean(np.absolute(test_y_hat - test_y)))


print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )