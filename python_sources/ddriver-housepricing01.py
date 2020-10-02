# Multiple linear regression model for predicting house prices using data from:
# Kaggle, House Prices Competition: Advanced Regression Techniques
    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques

    # Submissions: 1
    # Kaggle.com Prediction Score: 0.12994
    # Ranking - 08/16/18: 1525 of 4512 (Top 33.8%)
    
# Required modules
    # https://github.com/damani-14/Kaggle/blob/master/HousePrice/encoder.py
    # encoder.py

# Required data sets
    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
    # train.csv, test.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

def main():
#-----------------
# Data Exploration
#-----------------

    # Importing Data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Set plot parameters
    plt.style.use(style='ggplot')
    plt.rcParams['figure.figsize'] = (7, 5)

    # Investigating response distribution
    plt.hist(train.SalePrice, color='green')
    plt.show()

        # NOTE: Response variable is skewed
    print(train.SalePrice.skew(),'\n')

        # Adjust for Skew
    response = np.log(train.SalePrice)

        # Check
    print(train.SalePrice.skew(),'\n')

#--------------------
# Feature Engineering
#--------------------

    # Handling Numeric Variables
    #---------------------------

    quant_feat = train.select_dtypes(include = (np.number))
    corr = quant_feat.corr()


        # Investigating Correlations

    print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
    print(corr['SalePrice'].sort_values(ascending=False)[-5:], '\n')


        # Visualizing Positive Correlations

    print("Overall Quality: \n", train.OverallQual.unique(), "\n")
    print("Above Ground Living Area (ft-sq): \n", train.GrLivArea.unique(), "\n")
    print("No. of Cars in Garage: \n", train.GarageCars.unique(), "\n")
    print("Garage Area (sq-ft): \n", train.GarageArea.unique(), "\n")

    quality_pivot = train.pivot_table(index='OverallQual',
                                   values='SalePrice',aggfunc=np.median)
    quality_pivot.plot(kind='bar', color='green')
    plt.xlabel('Overall Quality')
    plt.ylabel('Median Sale Price')
    plt.show()
    # NOTE: Outliers @ 4000+

    livArea = plt.scatter(x=train['GrLivArea'],y=response)
    plt.xlabel('Above Ground Living Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.show()

    cars_pivot = train.pivot_table(index='GarageCars',
                                   values='SalePrice',aggfunc=np.median)
    cars_pivot.plot(kind='bar', color='green')
    plt.xlabel('Overall Quality')
    plt.ylabel('Median Sale Price')
    plt.show()

    garageArea = plt.scatter(x=train['GarageArea'],y=response)
    plt.xlabel('Garage Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.show()
    # NOTE: Outliers @ 1200+


        # Visualizing Negative Correlations

    print('Year Sold: \n', train.YrSold.unique(), '\n')
    print('Overall Condition: \n', train.OverallCond.unique(), '\n')
    print('Building Class: \n', train.MSSubClass.unique(), '\n')
    print('Enclosed Porch: \n', train.EnclosedPorch.unique(), '\n')
    print('Above Ground Kitchen: \n', train.KitchenAbvGr.unique(), '\n')

    year_pivot = train.pivot_table(index='YrSold',
                                    values='SalePrice',aggfunc=np.median)
    year_pivot.plot(kind='bar', color='green')
    plt.xlabel('Year Sold')
    plt.ylabel('Median Sale Price')
    plt.show()

    cond_pivot = train.pivot_table(index='OverallCond',
                                    values='SalePrice',aggfunc=np.median)
    cond_pivot.plot(kind='bar', color='green')
    plt.xlabel('Overall Cond')
    plt.ylabel('Median Sale Price')
    plt.show()

    bldg_pivot = train.pivot_table(index='MSSubClass',
                                    values='SalePrice',aggfunc=np.median)
    bldg_pivot.plot(kind='bar', color='green')
    plt.xlabel('Building Class')
    plt.ylabel('Median Sale Price')
    plt.show()

    porch_plot = plt.scatter(x=train['EnclosedPorch'],y=response)
    plt.xlabel('Enclosed Porch Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.show()
    # NOTE: Outliers @ 400+

    ktch_pivot = train.pivot_table(index='KitchenAbvGr',
                                    values='SalePrice',aggfunc=np.median)
    ktch_pivot.plot(kind='bar', color='green')
    plt.xlabel('Kitchen Above Ground(?)')
    plt.ylabel('Median Sale Price')
    plt.show()

        # Removing Outliers

    livArea = plt.scatter(x=train['GrLivArea'],y=response)
    plt.xlabel('Above Ground Living Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('LIVING AREA')
    plt.show()

    train = train[train['GrLivArea'] < 4000]
    response = np.log(train.SalePrice)
    livArea = plt.scatter(x=train['GrLivArea'],y=response)
    plt.xlabel('Above Ground Living Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('Outliers Removed')
    plt.show()

    garageArea = plt.scatter(x=train['GarageArea'],y=response)
    plt.xlabel('Garage Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('GARAGE AREA')
    plt.show()

    train = train[train['GarageArea'] < 1200]
    response = np.log(train.SalePrice)
    garageArea = plt.scatter(x=train['GarageArea'],y=response)
    plt.xlabel('Garage Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('Outliers Removed')
    plt.show()

    porch_plot = plt.scatter(x=train['EnclosedPorch'],y=response)
    plt.xlabel('Enclosed Porch Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('ENCLOSED PORCH')
    plt.show()

    train = train[train['EnclosedPorch'] < 400]
    response = np.log(train.SalePrice)
    porch_plot = plt.scatter(x=train['EnclosedPorch'],y=response)
    plt.xlabel('Enclosed Porch Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('Outliers Removed')
    plt.show()


    # Handling Non-Numeric Variables
    #-------------------------------

    qual_feat = train.select_dtypes(exclude=[np.number])
    quals = qual_feat.columns.values[np.newaxis]
    print('Qualitative Variables: \n',quals,'\n')

        # Feature Encoding 
    import encoder
    train, test = encode(train, test)

    # Handling Null Values
    # ---------------------

        # Visualizing

    nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
    nulls.columns = ['Null Count']
    nulls.index.name = 'PREDICTOR'
    print(nulls)

        # Interpolation

    data = train.select_dtypes(include=[np.number]).interpolate().dropna()
    print('\n Interp_NewNulls: \n', sum(data.isnull().sum() != 0))

#---------------
# Model Building
#---------------

    y = np.log(train.SalePrice)
    x = data.drop(['SalePrice', 'Id'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42, test_size=.33
    )

    lr = linear_model.LinearRegression()
    linReg = lr.fit(x_train, y_train)
    print('\n\n R-Squared: ', linReg.score(x_test, y_test))

    predictions = linReg.predict(x_test)
    print('\n\n RMSE: ', mean_squared_error(y_test, predictions))

    actual = y_test
    plt.scatter(predictions, actual, alpha=.75, color='black')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Linear Regression Model')
    overlay = 'R-Squared: {}\nRMSE: {}'.format(
        linReg.score(x_test, y_test),
        mean_squared_error(y_test, predictions))
    plt.annotate(s=overlay, xy=(11.7, 10.6))
    plt.show()

main()