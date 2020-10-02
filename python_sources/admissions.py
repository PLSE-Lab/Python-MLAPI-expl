import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression


my_data = pd.read_csv('../input//Admission_Predict_Ver1.1.csv')
my_data.drop('Serial No.', 1, inplace=True)
my_data.dropna(inplace=True)
'''
FEATURES
GRE Score, OEFL Score, University Rating, SOP, LOR, CGPA, Research
LABEL
Chance of Admit'''
# first we pre process
X = my_data.iloc[:, :-1]  # grabs every col except chance of admit
y = my_data.iloc[:, -1:]  # grabs chance of admit (label)
X = preprocessing.scale(X)  # scales all values between -1 - 1

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
# fits a line to the data
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Line of best fit found with accuracy of: ", accuracy*100, "%")
