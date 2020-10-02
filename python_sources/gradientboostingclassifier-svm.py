#!/usr/bin/env python
# coding: utf-8

# **Predict Diabetes From Medical Records**

# The [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) can be used to train machine learning models to predict if a given patient has diabetes.

# *Step 1: Import Python Packages*

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import graphviz 
from sklearn.preprocessing import Imputer
from sklearn import preprocessing  #hy
from sklearn.preprocessing import StandardScaler #hy
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.impute import SimpleImputer
from pandas import read_csv
get_ipython().run_line_magic('matplotlib', 'inline')


# *Step 2: Define Helper Functions*

# In[ ]:


def plot_decision_tree(a,b):
    """
    http://scikit-learn.org/stable/modules/tree.html
    """
    dot_data = tree.export_graphviz(a, out_file=None, feature_names=b,class_names=['Healthy','Diabetes'],filled=False, rounded=True,special_characters=False)  
    graph = graphviz.Source(dot_data)  
    return graph 

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("DT",DecisionTreeClassifier()))
models.append(("SVM",SVC()))


# *Step 3: Inspect and Clean Data*

# In[ ]:


dataset = read_csv('../input/train.csv')
dataset=dataset[['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']]
dataset.head(10)


# In[ ]:


dataset2 = dataset.iloc[:, :-1]
print("# of Rows, # of Columns: ",dataset2.shape)
print("\nColumn Name           # of Null Values\n")
print((dataset2[:] == 0).sum())


# In[ ]:


trainingData = read_csv('../input/train.csv') 
trainingData=trainingData[['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']]
testingData = read_csv('../input/test.csv')
testingData=testingData[['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
trainingFeatures = trainingData.iloc[:, :-1]
trainingLabels = trainingData.iloc[:, -1]
imputer = SimpleImputer(missing_values=0,strategy='median')
trainingFeatures = imputer.fit_transform(trainingFeatures)
trainingFeatures = pd.DataFrame(trainingFeatures)
trainingFeatures.columns=['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
#further feature engineering
#trainingData['Glucose'] = ((trainingData['Glucose'] >= 160)|(trainingData['Glucose'] <= 100)).astype(int)  #hy
#trainingData['Pregnancies'] = (trainingData['Pregnancies'] >= 5).astype(int)
#trainingData['Insulin'] = (trainingData['Insulin'] >= 200).astype(int)
#trainingData['DiabetesPedigreeFunction'] = (trainingData['DiabetesPedigreeFunction'] >= 0.5).astype(int)#hy
#print(trainingData[:])

testingData = imputer.transform(testingData)
testingData = pd.DataFrame(testingData)                  
#testingData.columns=['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
#testingData['Glucose'] = ((testingData['Glucose'] >= 160)|(testingData['Glucose'] <= 100)).astype(int)  #hy
#testingData['Pregnancies'] = (testingData['Pregnancies'] >= 5).astype(int)
#testingData['Insulin'] = (testingData['Insulin'] >= 200).astype(int)
#testingData['DiabetesPedigreeFunction'] = (testingData['DiabetesPedigreeFunction'] >= 0.5).astype(int) 

print("# of Rows, # of Columns: ",trainingFeatures.shape)
print("\nColumn Name           # of Null Values\n")
print((trainingFeatures[:] == 0).sum())


# *Step 4: Feature Engineering and Feature Selection*

# In[ ]:


g = sns.heatmap(trainingFeatures.corr(),cmap="Blues",annot=False)


# In[ ]:


trainingFeatures.corr()


# In[ ]:


#trainingFeatures2 = trainingFeatures.drop(['Pregnancies','BloodPressure','DiabetesPedigreeFunction', 'Age','SkinThickness','Insulin','Id'], axis=1)
trainingFeatures2 = trainingFeatures.drop(['Id'], axis=1)


# In[ ]:


g = sns.heatmap(trainingFeatures2.corr(),cmap="Blues",annot=False)
print(trainingFeatures2.corr())


# *Step 5: Evaluate Model*

# In[ ]:



#model = DecisionTreeClassifier(max_depth=8,min_samples_leaf=2)
#0.70-no norm . 0.76--w/ normalization
"""
model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                           C=1.0, fit_intercept=True, intercept_scaling=1, 
                           class_weight=None, random_state=10, solver='liblinear', 
                           max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)  
"""
#model =GaussianNB()   #74.17
#model= RandomForestClassifier(max_depth=6, random_state=0)
#""" %77.00
model= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=1,
            random_state=10, verbose=0, warm_start=False)
#"""
#model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6), n_estimators=150, random_state=10) #74

"""70.07
model = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=10, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
"""
#model = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage='auto', solver='eigen', store_covariance=False, tol=0.0001)
#model = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,  store_covariance=False, store_covariances=None, tol=0.0001)

#SVC: rbf(64) linear(76) poly-3
"""
model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=2, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
"""
#gradientBoostingClassifier . 0.77
#params = {'max_depth':9, 'subsample':0.5, 'learning_rate':0.01, 'min_samples_leaf':1, 'random_state':0}
#model = GradientBoostingClassifier(n_estimators=290,**params)

#model = GradientBoostingClassifier()

X_train, X_test, y_train, y_test = train_test_split(trainingFeatures2, trainingLabels, test_size=0.1, random_state=10)


#original
model.fit(X_train, y_train)

#scaler = StandardScaler()
#X_train_scaler = scaler.fit_transform(X_train)
#model.fit(X_train_scaler, y_train)

columns = trainingFeatures2.columns
feature_names = trainingFeatures2.columns.values

#coefficients = model.feature_importances_.reshape(trainingFeatures2.columns.shape[0], 1)
#absCoefficients = abs(coefficients)
#fullList = pd.concat((pd.DataFrame(columns, columns = ['Feature']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
#print('\nFeature Importance:\n\n',fullList,'\n')
#plot_decision_tree(model,feature_names)


# In[ ]:


kfold = KFold(n_splits=10, random_state=10)
results = cross_val_score(model, trainingFeatures2, trainingLabels, cv=kfold)
#print("DecisionTreeClassifier:\n\nCross_Val_Score: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("Logistic Regression Classifier:\n\nCross_Val_Score: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#origina
prediction = model.predict(X_test)

#X_test_scaler = scaler.fit_transform(X_test)
#prediction = model.predict(X_test_scaler)

cnf_matrix = confusion_matrix(y_test, prediction)
dict_characters = {0: 'Healthy', 1: 'Diabetes'}
plot_confusion_matrix(cnf_matrix, classes=dict_characters,title='Confusion matrix')


# *Step 6: Prepare submission file*

# In[ ]:


test = testingData
test = pd.DataFrame(test)
test.columns=['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
#test2 = test.drop(['Id','Pregnancies','BloodPressure','DiabetesPedigreeFunction', 'Age','SkinThickness','Insulin'], axis=1)
test2 = test.drop(['Id' ], axis=1)
#test2_scaler = scaler.fit_transform( test2)
my_predictions = model.predict(test2)
#my_predictions = model.predict(test2_scaler)
Identifier = test.Id.astype(int)
my_submission = pd.DataFrame({'Id': Identifier, 'Outcome': my_predictions})
my_submission.to_csv('my_submission.csv', index=False)
my_submission.head(10)


# *Step 7: Submit Results*

# 1. Click on the "Commit & Run" button in the top right corner of the kernel editor
# 1. Wait for the kernel to finish running then click on "View Snapshot"
# 1. Wait for the kernel viewer page to load then click on the "Output" tab
# 1. Find your submission file in the "Output" tab
# 1. Click "Submit to Competition" 
