# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
customer_id = "ncodpers"
usecols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
       
print("Preparing train...")

df_train = pd.read_csv("../input/train_ver2.csv", usecols=usecols)
sample = pd.read_csv("../input/sample_submission.csv")

customer_ids = pd.Series(df_train[customer_id].unique())

df_train.drop_duplicates(customer_id, keep="last", inplace=True)

df_train.fillna(0, inplace=True)

models = {}
model_preds = {}

ids = df_train[customer_id].values
id_preds = defaultdict(list)
for c in usecols[1:]:
    if c != customer_id:
        print(c)
        y_train = df_train[c]
        x_train = df_train.drop([c, customer_id], axis=1)
        
        clf = ensemble.RandomForestClassifier(n_estimators=70, n_jobs=-1, max_depth=10, min_samples_split=10, verbose=0)
        
        clf.fit(x_train, y_train)
        p_train1 = clf.predict_proba(x_train)[:,1]

        clf1 = LogisticRegression()
        #clf1 = ensemble.AdaBoostClassifier(n_estimators=70)
        clf1.fit(x_train, y_train)
        p_train2 = clf1.predict_proba(x_train)[:, 1]
        p_train = p_train1*0.8 + 0.2*p_train2

        for id,p in zip(ids, p_train):
            id_preds[id].append(p)

print("Checking already activated banking solutions...")
already_active={}
for row in df_train.values:
    row = list(row)
    id = row.pop(0)
    active = [c[0] for c in zip(usecols[1:], row) if c[1] > 0]
    already_active[id] = active

train_preds = {}
for id, p in id_preds.items():
    preds = [i[0] for i in sorted([i for i in zip(usecols[1:], p) if (i[0] not in already_active[id])], key=lambda i:i [1], reverse=True)[:7]]
    train_preds[id] = preds

print("Get predictions on test data...")
test_preds = []
for row in sample.values:
    id = row[0]
    p = train_preds[id]
    test_preds.append(' '.join(p))

sample['added_products'] = test_preds
sample.to_csv("ensemble_rt_lg_solution.csv", index=False)