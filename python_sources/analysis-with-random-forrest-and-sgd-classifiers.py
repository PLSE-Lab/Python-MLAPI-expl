# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.simplefilter("ignore")
# Any results you write to the current directory are saved as output.
#Setting dataframe max limit of columns in output to 15
pd.set_option('display.max_columns', 15)

#Importing data
data = pd.read_csv("../input/Admission_Predict.csv")

#Displaying the columns of the data
print(data.columns)

#Renaming chance of admit to chance_of_admit
data.rename(columns={'Chance of Admit ':'Chance_of_Admit'}, inplace=True)
print(data.columns)


#Displaying the shape and description of the data
print(data.shape)
print(data.describe())

#Displaying histogram of Chance_of_Admit column.
plt.hist(data["Chance_of_Admit"])
plt.show()

#Calculating and displaying mean value of Admittance
print("\n")
meanCOF = data["Chance_of_Admit"].mean()
print("Mean Percentage of Admittance:", meanCOF)
print("\n")

#Transforming the chance of admittance to 1 and 0
#1: Accepted
#0: Declined
data["Chance_of_Admit"] = np.where(data["Chance_of_Admit"] >= meanCOF, 1, 0)
print("Displaying the transformed column data:\n",data["Chance_of_Admit"].head(5))


#Correlation matrix & Heatmap - Finding correlation among the parameters
corrmat = data.corr()
fig = plt.figure()
sns.heatmap(corrmat, annot=True, fmt='.1f', vmin=0, vmax=1, square=True);
plt.show()


# Get all the columns from the dataframe.
columns = data.columns.tolist()
# Filter the columns to remove ones we don't want which in our case is LOR & Reserach as their correlation is less than 50%.
columns = [c for c in columns if c not in ["Serial No.", "Chance_of_Admit", "LOR", "Research"]]
# Store the variable we'll be predicting on.
target = "Chance_of_Admit"


#Defining features and labels
X = data[columns] #Features
y = data[target] #Labels

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print("\n")

#Printing the shapes of both sets.
print("Training FeaturesSet:", X_train.shape)
print("Training Labels:", y_train.shape)
print("Testing FeaturesSet:", X_test.shape)
print("Testing Labels:", y_test.shape)
print("\n")

'''Using linear classifier model(stochastic gradient descent (SGD))'''
#Initializing the model class.
model = SGDClassifier(max_iter = 100)
#Fitting the model to the training data.
model.fit(X_train, y_train)
#Generating our predictions for the test set.
predictions = model.predict(X_test)
#Computing the Model Accuracy
print("SGD Accuracy:",metrics.accuracy_score(y_test, predictions))
#Computing the error.
print("Mean Absolute Error:", mean_absolute_error(predictions, y_test))
#Computing classification Report
print("Classification Report:\n", classification_report(y_test, predictions))
#Plotting confusion matrix
print("Confusion Matrix:")
df = pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0', '1']])
print(df)
print("\n")

'''Using random forrest Model'''
#Initializing the model with some parameters.
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)
#Fitting the model to the data.
model.fit(X_train, y_train)
#Generating our predictions for the test set.
predictions = model.predict(X_test)
#Computing the Model Accuracy
print("Random Forrest Accuracy:",metrics.accuracy_score(y_test, predictions))
#Computing the error.
print("Mean Absoulte Error:", mean_absolute_error(predictions, y_test))
#Computing classification Report
print("Classification Report:\n", classification_report(y_test, predictions))
#Plotting confusion matrix
print("Confusion Matrix:")
df = pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0', '1']])
print(df)