import pandas as pd
import numpy  as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold


train = pd.read_csv(Load the processed train dataset)
test  = pd.read_csv(Load the processed test dataset)
PassengerId = test["PassengerId"]
test  = test.drop("PassengerId", axis = 1)

# Some useful parameters which will come in handy later on

ntrain = train.shape[0]
ntest  = test.shape[0]
seed   = 0
nfolds = 5
kf = KFold(ntrain,nfolds , random_state = seed)


class SklearnHelper(object):
    def __init__(self,clf, seed = 0 , params = None):
        params["random_state"] = seed
        self.clf = clf(**params)
    def train(self,x_train, y_train):
        self.clf.fit(x_train,y_train)
    def predict(self,x):
        return self.clf.predict(x)
    def fit(self,x,y):
        return self.clf.fit(x,y)
    def feature_importance(self,x,y):
        print self.clf.fit(x,y).feature_importances_

def get_oof(clf,x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test  = np.zeros((ntest,))
    oof_test_skf = np.empty((nfolds, ntest))
    
    for i ,(train_index,test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i,:] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}
# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}



# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }
    
rf = SklearnHelper(clf=RandomForestClassifier, seed=seed, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=seed, params=et_params)
svc = SklearnHelper(clf=SVC, seed=seed, params=svc_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=seed, params=ada_params)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data


et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 


base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'SVM' : svc_oof_train.ravel()
    })

x_train = np.concatenate(( et_oof_train, rf_oof_train, svc_oof_train,ada_oof_train), axis=1)
x_test  = np.concatenate(( et_oof_test, rf_oof_test,svc_oof_test,ada_oof_test), axis=1)

# Gradient Boosting parameters
gbm = GradientBoostingClassifier(
 n_estimators = 500,
 max_depth =  5,
 min_samples_leaf = 2,
 verbose =  0
)

gbm.fit(x_train, y_train)
predictions = gbm.predict(x_test)

# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })

StackingSubmission.to_csv(save the prediction, index=False)

