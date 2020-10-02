# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import RobustScaler, StandardScaler
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")

scores=[]

for i in  list(range(1,11)):
    df = df.sample(frac=1, random_state = i)
    #divid frauds in train (80%) and test (20%)
    fraud_df_train = df.loc[df['Class'] == 1][:int(492*0.8)]
    fraud_df_test = df.loc[df['Class'] == 1][int(492*0.8):]

    # undersmpling of norml data: the normal data represent 90% of the new base
    normal_df_train_sup= df.loc[df['Class'] == 0][:int(492*0.8*9)]
    normal_df_test= df.loc[df['Class'] == 0][int(492*0.8)*9:int(492*0.8*9)+int(284807*0.2)]
    new_df_train = pd.concat([normal_df_train_sup, fraud_df_train])
    new_df_test = pd.concat([normal_df_test, fraud_df_test])

    #normalization of the time and amount
    RS=RobustScaler()
    new_df_train['Amount'] = RS.fit_transform(new_df_train['Amount'].values.reshape(-1, 1))
    new_df_train['Time'] = RS.fit_transform(new_df_train['Time'].values.reshape(-1, 1))
    new_df_test['Amount'] = RS.fit_transform(new_df_test['Amount'].values.reshape(-1, 1))
    new_df_test['Time'] = RS.fit_transform(new_df_test['Time'].values.reshape(-1, 1))

    X_train_sup = new_df_train.drop('Class', axis=1)
    y_train = new_df_train['Class']
    X_test=new_df_test.drop('Class', axis=1)
    y_test=new_df_test['Class']

    #random forest 
    rfc = RandomForestClassifier(n_estimators = 1600,
     min_samples_split = 2,
     min_samples_leaf = 1,
     max_features = 'sqrt',
     max_depth = 100,
     bootstrap = False)

    rfc.fit(X_train_sup,y_train)
    prediçtion_rfc = rfc.predict_proba(X_test.values)
    tresholds = np.linspace(0 , 1 , 200)
    scores_rfc=[]
    for treshold in tresholds:
        y_hat_rfc = (prediçtion_rfc[:,0] < treshold).astype(int)
        scores_rfc.append([metrics.recall_score(y_pred=y_hat_rfc, y_true=y_test),
                 metrics.precision_score(y_pred=y_hat_rfc, y_true=y_test),
                 metrics.fbeta_score(y_pred=y_hat_rfc, y_true=y_test, beta=2),
                 metrics.cohen_kappa_score(y1=y_hat_rfc, y2=y_test)])
    scores_rfc = np.array(scores_rfc)
    #choice the model with best f2 score
    best_scores = scores_rfc[scores_rfc[:, 2].argmax(),:]
    scores.append(best_scores)
    
for i in range((len(scores))):
    recall_score = np.mean(scores[i][0])
    precision_score = np.mean(scores[i][1])
    fbeta_score = np.mean(scores[i][2])
    cohen_kappa_score = np.mean(scores[i][3])
    
print('The recall score is": %.2f' % recall_score)
print('The precision score is": %.2f' % precision_score)
print('The f2 score is": %.2f' % fbeta_score)
print('The Kappa score is": %.2f' % cohen_kappa_score)