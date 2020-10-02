


import pandas as pd
import seaborn as sns
import numpy as np
import pickle

data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')

##Data travelling 
test_data.info()
data.head(10)
nul=data.isnull().sum()
nul/data.shape[0]*100


sns.heatmap(data.corr())

label=data['Label']

#percentage of outcomes
(label.value_counts())/data.shape[0]*100
label.value_counts().plot(kind='bar')
timestamp=data['Timestamp']
train=data.drop(['Label','Timestamp'],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.as_matrix(), label.as_matrix(), test_size=0.2)


def evaluate(y_true,y_pred):
    e={}
    e.update({'confusion_matrix': confusion_matrix(y_true, y_pred)})
    e.update({'acc': accuracy_score(y_true, y_pred)})
    e.update({'f1_score': f1_score(y_true, y_pred, average='weighted')})
    e.update({'precision': precision_score(y_true, y_pred, average='weighted')})
    e.update({'recall': recall_score(y_true, y_pred, average='weighted')})
    return e
    


import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.01).fit(X_train, y_train)
pred2 = gbm.predict(X_test)

eval=evaluate(y_test,pred2)

df_cm = pd.DataFrame(eval['confusion_matrix'], index = [i for i in "ABCDE"],columns = [i for i in "ABCDE"])
sns.heatmap(df_cm, annot=True)



#saving the model
filename = 'xgboost_model.sav'
pickle.dump(gbm, open(filename, 'wb'))