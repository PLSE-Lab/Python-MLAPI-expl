# Import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics

#list files
import os
print(os.listdir("../input"))

# read files in
df = pd.read_csv("../input/creditcard.csv")
df.columns
df.head(10)
df.dtypes
df.describe().T
df.drop(["Time","Amount"],axis=1 ,inplace=True)

# Any missing variables?
df.isnull().sum()

# Find distribution of class
df['Class'].value_counts()
df.hist(figsize = (20,20))

# Standard Scaler for variables 
sc_X = StandardScaler()
columns = list(df.loc[:,df.columns != 'Class'].columns)
features = df[columns]

ct = ColumnTransformer([
        ('Transform', StandardScaler(), columns)
    ], remainder='passthrough')

X = pd.DataFrame(ct.fit_transform(features),columns = columns)

# Split labels and features
X.head()
Y= df['Class']

#Split data into training and test sets
test_size = 0.30
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
print (X_train.shape, Y_train.shape)
print (X_test.shape, Y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(Y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(Y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, Y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

X_train_res_df = pd.DataFrame(X_train_res,columns = X_train.columns)
X_train_res_df.shape
X_train_res_df.head()
y_train_res

#Evaluate different models

seed = 7
scoring = 'precision'
# Evaluate training accuracy
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RANDOM FOREST', RandomForestClassifier()))
#models.append(('SVM', SVC()))
models.append(('Gradient Boosting', GradientBoostingClassifier()))
#models.append(('Naive Baiyes', GaussianNB()))

results = []
names = []
for name, model in tqdm(models):
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train_res_df, y_train_res, cv=kfold, scoring=scoring, n_jobs= -1 ,verbose = 10)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
# Compare Algorithms
fig = plt.figure(figsize=(10,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Fit and Evaluate with the best model

model =  RandomForestClassifier()
model.fit(X_train_res_df, y_train_res)

feat_imp = pd.Series(model.feature_importances_, X_train_res_df.columns).sort_values(ascending=False)
fig = plt.figure(figsize=(15,10))
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

best_features= feat_imp[feat_imp > 0]
best_features

predictions = model.predict(X_test)
predictions_probs = model.predict_proba(X_test)
preds = np.where(predictions_probs[:,1] >= 0.5 , 1, 0)
print(accuracy_score(Y_test, preds))
print(confusion_matrix(Y_test, preds))
print(classification_report(Y_test, preds))
score_test = metrics.f1_score(Y_test, preds,
                          pos_label=list(set(Y_test)), average = None)
print(score_test)
