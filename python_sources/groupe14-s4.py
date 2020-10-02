# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib 
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (18,5)
import subprocess
import sys
import re
import seaborn as sns

print(os.listdir("../input"))

sales= pd.read_csv('../input/train3.csv')

categories = pd.read_csv('../input/item_categories.csv')
items = pd.read_csv('../input/items.csv')
sales = pd.read_csv('../input/sales_train.csv')
submission = pd.read_csv('../input/sample_submission.csv')
shops = pd.read_csv('../input/shops.csv')
test = pd.read_csv('../input/test.csv')

# Traitement de l'ensemble de train 
# Création des variables temporelles 
sales['date'] = pd.to_datetime(sales.date)
sales['year'] = sales['date'].dt.year
sales['month'] = sales['date'].dt.month
sales["weekday"] = sales["date"].dt.day_name()

# Une vue globale des données 
sales.describe()

# Nettoyage des données & préparation de la dataset
# Enveler les prix négatifs 
sales = sales.query('item_price > 0')

# Enveler pour toutes les ventes négatives (correspondant aux retours ou aux données aberrantes)
sales = sales.query('item_cnt_day > 0')

#En faisant une traduction, ce produit correspond au service de livraison : "Livraison au point de livraison (Boxberry)"
sales_train = sales.query('item_id != 11373')

#La traduction de l'intitulé du item est : "Paquet de marque T-shirt 1C White Interest (34*42)" Nous avons les dimentions donc ça correspond à un packet, nous allons également le sortir de notre dataset 
sales_train = sales_train.query('item_id != 20949')

#Nous allons vérifier les vents 
sales_train.sales.max()
sales_train.sales.median()
sns.barplot(x='date_block_num', y= 'sales', data= sales)

#Nous observons qu'il y a des outiers/des valeurs de ventes très élévées et rares, nous allons les supprimer
sales_train = sales_train.query('sales >200 ') 

# Nous allons analyser des magasins 

# A l'aide de ce code nous visualisons tous les magasins 
g = sns.catplot(x="date_block_num", y="sales",
                col="shop_id",
                data=sales_train, kind="bar",
                height=4, aspect=.7);

# Nous remarquons que certains magasins n'ont plus de données, nous allons des supprimer

closed_shops = [0, 1, 8, 11, 13, 17, 23, 27, 29, 30, 32, 33, 36, 40, 43, 54]
sales_non_closed_shop = sales_train[~sales_train.shop_id.isin(closed_shops)]

# Nous analysons les prix 
sns.lmplot(x="item_price", y="sales", data=sales_non_closed_shop);

# Nous avons une outlier, nous allons la supprimer en limitant le prix à 100000
sales_non_closed_shop = sales_non_closed_shop.query('item_price < 100000')


# Nous vérifions les valeurs de ventes et prix 
sales_non_closed_shop.sales.max()
sales_non_closed_shop.item_price.max()

# Nous vérifions la structure les données 
sales_non_closed_shop.head()

#renommer la base de donnée

sales0 = sales_non_closed_shop

# Réalisation du modèle de régression linéiare : 

from sklearn.linear_model import LinearRegression

#création un objet reg lin
modeleReg=LinearRegression()

#création y et X
list_var=sales0.columns.drop("sales")
y=sales0.sales
X=sales0[list_var]

modeleReg.fit(X,y)

#calcul du R²
modeleReg.score(X,y)

# Le modèle à un R² très faible 

# Nous allons tester le Random forest 

# la variable à expliquer 
labels = np.array(sales0['sales'])

# l'ensemble des variables explicatives 
features= sales0.drop('sales', axis = 1)

feature_list = list(features.columns)

# Convertion to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('date_block_num')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 50 decision trees
rf = RandomForestRegressor(n_estimators = 50, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 50, random_state=42)

# Extract the two most important features
important_indices = [feature_list.index('month'),feature_list.index('item_id'), feature_list.index('shop_id')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

# Train the random forest
rf_most_important.fit(train_important, train_labels)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))

accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

# Use datetime for creating date objects for plotting
import datetime

# Dates of training values
months = features[:, feature_list.index('month')]
years = features[:, feature_list.index('year')]

# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) for year, month in zip(years, months)]
dates = [datetime.datetime.strptime(date, '%Y-%m') for date in dates]

# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'sales': labels})

# Dates of predictions
months = test_features[:, feature_list.index('month')]
years = test_features[:, feature_list.index('year')]

# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) for year, month in zip(years, months)]

# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m') for date in test_dates]

# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})

# plot of real data and prediction 
sns.barplot(x='date', y= 'prediction', data= predictions_data)
sns.barplot(x='date', y= 'sales', data= true_data)

# second plot of real data and prediction of sales
plt.figure(figsize=(16, 6))
plt.scatter(x='date', y= 'sales', data= true_data)
plt.scatter(x='date', y= 'prediction', data= predictions_data)

# Convertion of the table test with numpy 
print('Testing Labels Shape:', test.shape)
test_labels = np.array(test)

# Novembre 2015 sales prediction 
test_predictions = rf_most_important.predict(test)

# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (test_labels[0], test_predictions[0]))









