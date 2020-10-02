## Import Modules
import numpy as np
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix


## Load Data (Python conform)
df = pd.read_csv("../input/EEG_data.csv") 
df = df.rename(columns={'user-definedlabeln': 'userdefinedlabeln'})

print("DATA OVERVIEW")
print("\nActive file full dimensions (rows, columns): ", df.shape)

## Define Input/Output
X = df[["SubjectID", "VideoID", "Attention", "Mediation", "Raw","Delta","Theta","Alpha1","Alpha2","Beta1","Beta2","Gamma1","Gamma2","predefinedlabel"]]
y = df.userdefinedlabeln

print("Chosen Input: ", X.columns.values)
print("Chosen Outpout:", y.name)

## Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)

## Scaling(seperate for train and test set)

scaler = preprocessing.StandardScaler() 
scaled_values_train = scaler.fit_transform(X_train) 
scaled_values_test = scaler.fit_transform(X_test) 

X_scaled = scaled_values_train
columns = X.columns
X_test_scaled = pd.DataFrame(data=scaled_values_test,columns=columns)


##Over-sampling training data using SMOTE
os = SMOTE(random_state=0)
columns = X.columns
os_X_train,os_y_train = os.fit_sample(X_scaled, y_train)
os_data_X = pd.DataFrame(data=os_X_train,columns=columns)
os_data_y = pd.DataFrame(data=os_y_train,columns=['y'])


# Oversampling report
print("\nBalancing data with synthetic data..")
print("\nLength of synthetic training data:",(len(os_data_X) - len(X_train)))
print("Length of original training data:",len(X_train))
print("Length of oversampled training data:",len(os_data_X))
print("Proportion of negative examples in original data:", round(len(y_train[y_train==0])/len(y_train), 2))
print("Proportion of negative examples in oversampled data:",len(os_data_y[os_data_y['y']==0])/len(os_data_X))



# Set the parameters by cross-validation

tuned_parameters = {'kernel': ['rbf'],
                    'gamma': [1e-3, 1e-4,'scale'],
                    'C': [1, 10, 100, 1000]}

##Add different measures for optimisation

scores = ['accuracy']

###Fitting Models

for score in scores:
    print()
    print("Tuning number of features for %s ..." % score)
    
    # Recursive feature elimination with Cross validation.
    estimator = SVC(kernel = "linear", C = 10 )
    RFECV = RFECV(estimator, step = 1, cv = 10, n_jobs = -1, scoring = score)
    
    selector = RFECV.fit(os_data_X, np.ravel(os_data_y))
    X_selected = selector.transform(os_data_X)
    
    
    print()
    print("RFECV Grid scores on development set (not on test set):")
    print()
    index = 0
    for number in selector.grid_scores_:
        index += 1
        print(index,"features =", number)
    print()
    print("Optimal number of features according to gridscore: %d" % selector.n_features_)
    print()
    print("Ranking the features ...")
    print()
    index = -1
    for number in selector.ranking_:
        index += 1
        print(X_train.columns.values[index], number)
    
    #Building DF with columns of selected features
    X_selected_df = pd.DataFrame(X_selected, columns=[X_train.columns[i] for i in range(len(X_train.columns)) if selector.get_support()[i]])
    print()
    print("Selected features for train set:") 
    print(X_selected_df.columns.values)
    X_test_scaled_selected = pd.DataFrame(X_test_scaled.loc[:,X_selected_df.columns.values])
    print()
    print("Selected features for test set:")
    print(X_test_scaled_selected.columns.values)
    print()
    print('Accuracy with of optimal features on test set: {:.2f}%'.format(selector.score(X_test_scaled, y_test) * 100))
    print()
    print("Tuning hyper-parameters for %s ..." % score)
    print()
    clf = GridSearchCV(SVC(probability=True),tuned_parameters, cv=10,n_jobs=-1, scoring = score)
    grid = clf.fit(X_selected, np.ravel(os_data_y))
    print("GridSearchCV Grid scores on development set (not on test set):")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print()
    print("ACCURACY REPORT")
    print()  
    print('Accuracy of optimised SVC on test set: {:.2f}%'.format(grid.score(X_test_scaled_selected, y_test) * 100))
    print('Old SVM Accuracy for reference : 67.2%')
    print()  
    print("Detailed report for Test set measures:")
    print()  
    y_true, y_pred = y_test, grid.predict(X_test_scaled_selected)
    print(classification_report(y_true, y_pred))
    
###ROC Curve


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

probs = grid.predict_proba(X_test_scaled_selected)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print()
print("AUC ROC REPORT")
print()
print('AUC score: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr, tpr)

