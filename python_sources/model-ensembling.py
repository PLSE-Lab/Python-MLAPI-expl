# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from sklearn.base import TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!pip install mlens

from mlens.ensemble import SuperLearner
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression,Ridge
#from sklearn.svm import SVC
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#from mlens.ensemble import SequentialEnsemble




#Train test target prepared
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
test = test.drop('ID_code',axis= 1)
X= train.drop(['ID_code','target'],axis=1)
y = train.target


def feature_engineering(X,test):	
	
	ss = StandardScaler()
	X = ss.fit_transform(x)
	test = ss.transform(test)

	qt = QuantileTransformer(output_distribution='normal')
	X = qt.fit_transform(X)
	test = qt.transform(test)
	return X,test 

#Models and parameters 
lgb_param = {
	'bagging_freq': 5,
	'bagging_fraction': 0.335,
	'boost_from_average':'false',
	'boost': 'gbdt',
	'feature_fraction': 0.041,
	'learning_rate': 0.0083,
	'max_depth': -1,
	'metric':'auc',
	'min_data_in_leaf': 80,
	'min_sum_hessian_in_leaf': 10.0,
	'num_leaves': 13,
	'num_threads': 8,
	'tree_learner': 'serial',
	'objective': 'binary', 
	'verbosity': -1
}
spw = float(len(y[y == 1])) / float(len(y[y == 0]))
params_xgb = {
	'eta': 0.02,
	'max_depth': 1,
	'subsample': 0.29,
	'colsample_bytree': 0.04,
	'lambda': 0.57,
	'alpha': 0.08,
	'min_child_weight': 5.45,
	'max_delta_step': 1.53,
	'scale_pos_weight': spw,
	'tree_method': 'gpu_hist',
	'objective': 'binary:logistic',
	'eval_metric': 'auc',
	'n_gpus': 1,
	'verbosity': 0,
	'silent': True
}
#rf = RandomForestClassifier (n_estimators=100, max_depth=7 ,max_features=0.7)
#et  = ExtraTreesClassifier (n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1)
#gbc = GradientBoostingClassifier(n_estimators=999999, learning_rate=0.02, max_depth=5, max_features=0.5, random_state=1)
xgb = XGBClassifier(params=params_xgb)
lgb = LGBMClassifier(params=lgb_param)
lr = LogisticRegression(random_state=1)
#rf_2 = RandomForestClassifier (n_estimators=200, criterion="entropy", max_depth=5, max_features=0.5, random_state=1)



#Build ensemble model
def build_ensemble(incl_meta, proba, propagate_features=[0,1]):
	"""Return an ensemble."""
	if propagate_features:
		n = len(propagate_features)
		propagate_features_1 = propagate_features
		propagate_features_2 = [i for i in range(n)]
	else:
		propagate_features_1 = propagate_features_2 = None
		
		#change here
	estimators_layer1 = [xgb]
	estimators_layer2 = [lgb]
#	estimators_layer3 = [rf,et ...........]


	ensemble = SuperLearner()
	ensemble.add(estimators_layer1, proba=proba, propagate_features=propagate_features)
	ensemble.add(estimators_layer2, proba=proba, propagate_features=propagate_features)

	if incl_meta:
		ensemble.add_meta(lr )
		
		return ensemble





#function to run & evaluate the ensembled model
def run_model(X,y,test,model):
	model.fit(X,y)
	preds_1 = model.predict(test)
	preds_0 = model.predict(X)
	print("\nAccuracy:\n%r" % accuracy_score(preds_0, y))
	return preds_1




# GO
X,test  = feature_engineering(X, test)
model = build_ensemble(True,True)
preds_1 = run_model(X, y, test, model)



# Create_submission file
submission = pd.DataFrame({"ID_code": test.ID_code.values})
submission["target"] = preds_1
submission.to_csv("submission_spl_1.csv", index=False)

#from google.colab import files
#files.download("submission_spl_1.csv")