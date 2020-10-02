#101077205 Nolan Honey
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from scipy.stats import skew

from scipy.special import boxcox1p

from sklearn.feature_selection import RFECV

from sklearn.linear_model import Lasso

from sklearn.model_selection import cross_val_score



training_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



# remove outliers

training_data = training_data[~((training_data['GrLivArea'] > 4000) & (training_data['SalePrice'] < 300000))] #filter out potentially uniquely huge houses that don't fit the general trend as told by the documentation



combined_data = pd.concat((training_data.loc[:,'MSSubClass':'SaleCondition'], #load the CSV's into an object

                      test_data.loc[:,'MSSubClass':'SaleCondition']))




combined_data.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'], axis=1, inplace=True) #drops unimportant columns



training_data["SalePrice"] = np.log1p(training_data["SalePrice"]) #logarithmically convert the sale price of the test data. This creates an easier to predict value that leaves less room for prediction errors (for the training data)



num_carcts = combined_data.dtypes[combined_data.dtypes != "object"].index #find the non object data in the csv's



skewed_carcts = training_data[num_carcts].apply(lambda x: skew(x.dropna())) #identify data skew in the training data

skewed_carcts = skewed_carcts[skewed_carcts > 0.65] 

skewed_carcts = skewed_carcts.index


                                                                #: https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Box-Cox_Transformation.pdf

combined_data[skewed_carcts] = boxcox1p(combined_data[skewed_carcts], 0.15) #learn about box-cox at the link above, we are using what we learned from the training data, and applying it to the test data as well


combined_data = pd.get_dummies(combined_data)



combined_data = combined_data.fillna(combined_data.mean()) #fills blank data, with average data. Clever way of potentially smoothing our the curve to gain minor percent points. If aiming for 100% accuracy this is not something you want to do.



X_training_data = combined_data[:training_data.shape[0]]

X_test_data = combined_data[training_data.shape[0]:]

y = training_data.SalePrice



lasso = Lasso(alpha=0.0004) #learn about lasso here, and why I used it: https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b
                            #Pretty much, we are attempting to remove the strict overfitting done above and return a more accurate result.
model = lasso



model.fit(X_training_data, y)



preds = np.expm1(model.predict(X_test_data)) #undoing the logarithmic function we started with, re-expanding the data

solution = pd.DataFrame({"id":test_data.Id, "SalePrice":preds})

solution.to_csv("COMP3122Assignment3_NolanHoney.csv", index = False)