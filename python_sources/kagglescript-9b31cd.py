# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import os 
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
MAJOR_THRESHOLD = 0.6
SPECIFIC_THRESHOLD = 0.5
DATAFRAME_LENGTH = 30450
TEST_LENGTH = 12452
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
print(os.getcwd())
df = pd.read_csv('../input/saftey_efficay_myopiaTrain.csv')
df_test = pd.read_csv('../input/saftey_efficay_myopiaTest.csv')

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

df = df[:DATAFRAME_LENGTH]
to_drop = []
to_drop_test = []
one_class = df[df['Class']==1]


one_class_length = len(one_class)
for col in df.columns:
    x = df[col]
    if(col != 'Class'):
        x_test = df_test[col]
        x_test = x_test.dropna()
    if len(x_test) < (1-MAJOR_THRESHOLD) * TEST_LENGTH:
        to_drop_test.append(col)
    x_one = one_class[col]
    x_one = x_one.dropna()
    x = x.dropna()
    if len(x) < (1 - MAJOR_THRESHOLD) * DATAFRAME_LENGTH:
        if len(x_one) > (1-SPECIFIC_THRESHOLD) * one_class_length:
            print("HIHIHHIH")
        if(col != 'Class'):
            to_drop.append(col)
    elif len(x_one) < (1-SPECIFIC_THRESHOLD) * one_class_length:
        to_drop.append(col)
to_drop.append('D_L_Dominant_Eye')
to_drop.append('T_L_Year')
to_drop.append('Pre_L_Contact_Lens')
to_drop = list(set(to_drop) | set(to_drop_test))
df = df.drop(to_drop, axis=1)

dummies = []
cols = ['D_L_Sex', 'D_L_Eye', 'D_L_Dominant_Eye', 'Pre_L_Contact_Lens', 'T_L_Laser_Type', 'T_L_Treatment_Type',
        'T_L_Cust._Ablation',
        'T_L_Micro', 'T_L_Head', 'T_L_Therapeutic_Cont._L.', 'T_L_Epith._Rep.']
intersection = list(set(to_drop) & set(cols))
cols = [col for col in cols if col not in intersection]
for col in cols:
    dummies.append(pd.get_dummies(df[col]))

laser_eye_dumm = pd.concat(dummies, axis=1)
df = pd.concat((df, laser_eye_dumm), axis=1)
df = df.drop(cols, axis=1)
df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

X = df.copy()
X = X.drop(['Class'], axis=1)

df_test = df_test.drop(to_drop, axis=1)
dummies = []
for col in cols:
    dummies.append(pd.get_dummies(df_test[col]))
laser_eye_dumm = pd.concat(dummies, axis=1)
df_test = pd.concat((df_test, laser_eye_dumm), axis=1)
df_test = df_test.drop(cols, axis=1)
df_test = df_test[:TEST_LENGTH]
df_test = df_test.apply(lambda x: x.fillna(x.mean()), axis=0)

for col in X.columns:
    if col not in df_test.columns:
        X = X.drop(col, axis=1)


for col in df_test.columns:
    if col not in X.columns:
        df_test = df_test.drop(col, axis=1)


X = X.values
Y = df['Class'].values

X_test = df_test.copy()
X_test.info()
X_test = X_test.values





# one_examples = X[Y==1]
# zero_examples = X[Y==0]
# random_zero_indexes = np.random.choice(range(len(zero_examples)), len(one_examples)*2, replace=False)
# chosen_zeros = zero_examples[random_zero_indexes]
# print(chosen_zeros.shape)
# print(one_examples.shape)
# X_resample = np.concatenate([one_examples, chosen_zeros])
# Y_resample = np.concatenate([Y[Y==1], Y[random_zero_indexes]])
# X_resample, Y_resample = shuffle_in_unison(X_resample, Y_resample)
# print(X_resample.shape)
# print(Y_resample.shape)

# X, X_test, Y, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)
# tl = TomekLinks(return_indices=True, ratio='majority')


sm = RandomUnderSampler(random_state=12, ratio=1)
X_resample, Y_resample = sm.fit_resample(X, Y)

print(len(Y_resample[Y_resample == 1]))
print(len(Y_resample[Y_resample == 0]))

print('X_train ---------------------------')
print(X_resample.shape)
print('X_test ---------------------------')
print(X_test.shape)


# max_auc = 0
# iteration  = 0
# max_params = []
# for min_child_weight in range(1, 8):
#     for gamma_before_process in range(1,3):
#         gamma = gamma_before_process/10
#         for colsmaple_before_process in range(5,10):
#             colsample = colsmaple_before_process/10
#             gbm = xgb.XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.01,
#             min_child_weight=6, reg_alpha=100, colsample_bytree=0.2,gamma=0.8, scale_pos_weight=1).fit(X_resample, Y_resample)
#             pred = gbm.predict(X_test)
#             fpr, tpr, threshold = metrics.roc_curve(Y_test, pred, pos_label=1)
#             acc = metrics.auc(fpr, tpr) 
#             if acc > max_auc:
#                 max_auc = acc
#                 max_params = [min_child_weight, gamma, colsample]
#             iteration += 1
#             print(iteration)
# print("This is the maximal auc {} with the following params : {}".format(max_auc, max_params))

# for colsample in range(40, 60):
#     cols = colsample/100
#     gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01,min_child_weight=6,
#     reg_alpha=100, colsample_bytree=cols,gamma=0.8, scale_pos_weight=1, eta=0.02).fit(X_resample, Y_resample)
#     pred = gbm.predict(X_test)
#     fpr, tpr, threshold = metrics.roc_curve(Y_test, pred, pos_label=1)
#     acc = metrics.auc(fpr, tpr) 
#     print(acc)

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01,min_child_weight=6,reg_alpha=100, colsample_bytree=0.90,gamma=0.8, scale_pos_weight=1, eta=0.02).fit(X_resample, Y_resample)
pred = gbm.predict_proba(X_test)

# params = []
# for scale in range(7, 14):
#     gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01,
#                 min_child_weight=6, reg_alpha=100, colsample_bytree=0.8,gamma=0.8, scale_pos_weight=scale/10, eta=).fit(X_resample, Y_resample)
#     pred = gbm.predict(X_test)
#     fpr, tpr, threshold = metrics.roc_curve(Y_test, pred, pos_label=1)
#     acc = metrics.auc(fpr, tpr)
#     print(acc)






# best_dept = 2
# best_res = 0
# for depth in range(2,8):
#     gbm = xgb.XGBClassifier(max_depth=depth, n_estimators=300, learning_rate=0.01).fit(X_resample,Y_resample)
#     pred = gbm.predict(X_test)
#     fpr, tpr, threshold = metrics.roc_curve(Y_test, pred, pos_label=1)
#     acc = metrics.auc(fpr, tpr)
#     if acc > best_res:
#         best_res = acc
#         best_dept = depth
# print("this is the best accuracy {} with depth {}".format(best_res, best_dept))
    



dict_df = {}
dict_df['id'] = [i for i in range(1, len(pred) + 1)]
dict_df['class'] = [y for x, y in pred]
df = pd.DataFrame.from_dict(dict_df)
print(df)
df.to_csv('results.csv', index=False)