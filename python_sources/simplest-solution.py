import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
import numpy as np

def get_macc(real_label, predict_label):
    n1 = len(np.nonzero(real_label == 1)[0])
    n2 = len(np.nonzero(real_label == 0)[0])
    tp_temp = sum(predict_label[np.nonzero(real_label == 1)[0]] == 1)
    tn_temp = sum(predict_label[np.nonzero(real_label == 0)[0]] == 0)
    tp = tp_temp / n1
    tn = tn_temp / n2
    m_acc = (tp + tn) / 2
    return tp,tn,m_acc

data=pd.read_csv('../input/creditcard.csv')
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
y=data['Class'].values
data=data.drop(['Class'],axis=1)
X=data.values

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
folds.get_n_splits(X,y)
params={
    'C':[100]
}
params=list(ParameterGrid(params))
for param in params:
    print(param)
    clf=LogisticRegression(C=param['C'],class_weight='balanced')
    for i,(train_index, validate_index) in enumerate(folds.split(X, y)):
        print('fold:'+str(i+1))
        X_train_fold=X[train_index,:]
        y_train_fold=y[train_index]
        X_validate_fold=X[validate_index,:]
        y_validate_fold=y[validate_index]
        clf.fit(X_train_fold,y_train_fold)
        fold_preds=clf.predict_proba(X_validate_fold)
        fold_preds=np.reshape(fold_preds[:,-1],(1,-1))[0]
        fold_preds[fold_preds>=0.5]=1
        fold_preds[fold_preds<0.5]=0
        tp, tn, m_acc=get_macc(y_validate_fold,fold_preds)
        print('TPR:'+str(tp)+' TNR:'+str(tn)+' M-ACC:'+str(m_acc))
