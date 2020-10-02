import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn import metrics, preprocessing 
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

decades = ['60', '70', '80', '90', '00', '10']

for decade in decades:

	filename = '../input/the-spotify-hit-predictor-dataset/dataset-of-' + decade + 's.csv'
	data = pd.read_csv(filename, sep=',')

	data = data.iloc[1:, 3:]
	X, y = data.iloc[:, :-1], data.iloc[:, -1]

	X = preprocessing.scale(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=5)

	model = XGBClassifier() 
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	# predictions = [round(value) for value in y_pred]

	accuracy = round(100*float(metrics.accuracy_score(y_test, y_pred)),2)
	print(decade + "s Accuracy: ", accuracy)
