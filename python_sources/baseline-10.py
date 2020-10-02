import numpy as np
import pandas as pd
import seaborn as sns

import os
import json
import copy
import colorama
from hyperopt import fmin, hp, Trials, tpe, STATUS_OK
import catboost
import lightgbm as lgb
import xgboost as xgb

from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split

sns.set(style="whitegrid")


def load_data(base_dir):
    print("base_dir = {}".format(base_dir))
    print(os.listdir("{}".format(base_dir)))
    x_train = None
    for i in range(1, 5):
        filename = "{}/x_train_{}.npz".format(base_dir, i)
        with np.load(filename) as data:
            print("files in {}: {}".format(filename, data.files))
            temp_data = data[data.files[0]]
            if x_train is None:
                x_train = temp_data
            else:
                x_train = np.concatenate((x_train, temp_data))


    with np.load('{}/y_train.npz'.format(base_dir)) as data:
        print("files in {}/y_train.npz: {}".format(base_dir, data.files))
        y_train = data[data.files[0]]

    with np.load('{}/x_test.npz'.format(base_dir)) as data:
        print("files in {}/x_test.npz: {}".format(base_dir, data.files))
        x_test = data[data.files[0]]
    return x_train, y_train, x_test

def save_y(y, name='submit.csv'):
    resdf = pd.DataFrame(y, columns=['Label'])
    resdf.index += 1
    resdf.to_csv(name, index_label='Id')


fstr = [('43', 8.803457128907672), ('102', 5.657777285583846), ('3', 4.921183061033979), ('73', 3.9014327557898456), ('96', 3.4027423502815153), ('56', 3.230180779813066), ('16', 2.7135719515168577), ('131', 2.536484446361014), ('40', 2.3941274897830613), ('111', 2.0682999536797255), ('83', 1.8910743066642743), ('144', 1.8200146087139368), ('118', 1.8134802913574384), ('98', 1.8062444831909843), ('20', 1.7943152373034696), ('101', 1.772391409822346), ('80', 1.7435754846224794), ('70', 1.4053119681427961), ('127', 1.2104250993356198), ('25', 1.2018906425089664), ('82', 1.160574667599466), ('34', 1.1305451445222972), ('141', 1.024304364413712), ('44', 0.953797466566064), ('50', 0.9403124303742796), ('60', 0.82849363029066), ('132', 0.8090197292814137), ('140', 0.7739597427006878), ('69', 0.7253347980235474), ('42', 0.7095907547903215), ('55', 0.7070827090143172), ('29', 0.7057713659728362), ('136', 0.7052355249532832), ('134', 0.6821534521396463), ('93', 0.6434576263255404), ('91', 0.6413076390760933), ('129', 0.6385910999948107), ('1', 0.6347626193506928), ('92', 0.6311484282067014), ('72', 0.6262047275112423), ('130', 0.6213252227073323), ('64', 0.6093981432360506), ('47', 0.5803451106723009), ('23', 0.579370663830151), ('71', 0.5762997099121144), ('21', 0.576231040209138), ('39', 0.5647054968665781), ('87', 0.5633090350667319), ('95', 0.549279105615663), ('31', 0.5465646078022413), ('112', 0.5149376785265147), ('51', 0.4995431321310032), ('24', 0.49816891011371606), ('79', 0.4981491431517481), ('106', 0.4940407261902606), ('57', 0.4857787148953463), ('77', 0.4782484532072758), ('13', 0.474691133035778), ('26', 0.47459521541065186), ('0', 0.46714423976825203), ('7', 0.4639834414750478), ('6', 0.46044322023386486), ('53', 0.45628191907384946), ('2', 0.44877230324975326), ('143', 0.4454117286119567), ('115', 0.4263562427061913), ('89', 0.42169360825698937), ('146', 0.4210520524235943), ('110', 0.41605655411659676), ('139', 0.4119267613876458), ('108', 0.41071521144111034), ('148', 0.4064853384159514), ('38', 0.39879260444095727), ('78', 0.39650919548130287), ('117', 0.3827237707391293), ('59', 0.37413817770347124), ('135', 0.37243679381198486), ('116', 0.36781149423477916), ('99', 0.36771526249890013), ('86', 0.36098410504398987), ('22', 0.3468723953965951), ('123', 0.34528444663391833), ('76', 0.3447334520874348), ('88', 0.32838751657874116), ('28', 0.31810878818473626), ('5', 0.3175095962049685), ('124', 0.3086509072387901), ('46', 0.3042994515856961), ('107', 0.3038520287030859), ('68', 0.29646961522155796), ('137', 0.286528707123406), ('145', 0.28009813942373907), ('11', 0.2800509980105112), ('142', 0.27961175547920397), ('17', 0.2720813015432389), ('74', 0.26442479336846214), ('19', 0.26322702871983583), ('75', 0.2629279851216979), ('63', 0.24349782246162505), ('8', 0.21796328295333064), ('94', 0.2027034015187897), ('33', 0.20009395155279464), ('62', 0.1837336371663767), ('37', 0.167063651400448), ('66', 0.16456679756398487), ('27', 0.16355185848336157), ('121', 0.1600271775548028), ('119', 0.14242793294927664), ('120', 0.14188666474929124), ('58', 0.13300917837384243), ('84', 0.13078303911398584), ('67', 0.12848922568754462), ('104', 0.10930067101929475), ('48', 0.10852730386149807), ('15', 0.10759477787856286), ('138', 0.10330830073096173), ('114', 0.0981188131174161), ('41', 0.09334892498553195), ('126', 0.08608785458566304), ('4', 0.08010089978921768), ('14', 0.07649890510482647), ('113', 0.07275768521075111), ('49', 0.07270113051413307), ('85', 0.06667220278910788), ('147', 0.06416900081201313), ('133', 0.06390966574786004), ('128', 0.05010589714043179), ('97', 0.04560087560640617), ('54', 0.03897230365866812), ('36', 0.03448472650186945), ('125', 0.034269287598868804), ('45', 0.03396362193396653), ('61', 0.027568966504437555), ('105', 0.02325329492287922), ('100', 0.022014819040498984), ('10', 0.01658187030478275), ('12', 0.014908176614815652), ('65', 0.012562540961396654), ('32', 0.012265445111763165), ('52', 0.011869781759321902), ('81', 0.006413436448372318), ('35', 0.006253990567024333), ('109', 0.00017283842875254947), ('9', 0.00015262532519028937), ('30', 0.00013082730476186787), ('103', 0.00012211847734717792), ('18', 0.00010449126368586217), ('122', 9.915929329534674e-05), ('90', 7.231771313121308e-05)]
factors_sorted = list(map(lambda x: int(x[0]), fstr))

FACTORS_COUNT = 90
BINS_COUNT = 21

X, y, X_submit = load_data("../input")

X = X[:, factors_sorted[:FACTORS_COUNT]]
X_submit = X_submit[:, factors_sorted[:FACTORS_COUNT]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=228)

train_pool_cb = catboost.Pool(X_train, label=y_train)
test_pool_cb = catboost.Pool(X_test, label=y_test)

train_pool_lgb = lgb.Dataset(X_train, label=y_train)
test_pool_lgb = lgb.Dataset(X_test, label=y_test)

params_cb = {
    'loss_function': 'RMSE',
    'iterations': 15000,
    'early_stopping_rounds': 400,
    'use_best_model': True,
    'task_type': 'GPU',
    'max_bin': 254,
    'fold_len_multiplier': 1.2,
    'bagging_temperature': 0.615,
    'depth': 11,
    'learning_rate': 0.1,
    'l2_leaf_reg': 10.81,
    'metric_period': 500
}
model_cb = catboost.CatBoost(params_cb).fit(train_pool_cb, eval_set=test_pool_cb)

params_lgb = {
    'objective': 'rmse',
    'num_iterations': 10000,
    'num_threads': 2,
    'early_stopping_rounds': 300,
    'silent': True,
    'num_leaves': 55,
    'max_depth': 8,
    'learning_rate': 0.15,
    'lambda_l2': 6.5,
}
model_lgb = lgb.train(params_lgb, train_pool_lgb, valid_sets=[test_pool_lgb], verbose_eval=400)

def gen_splits(X, y):
    pred_cb = model_cb.predict(X).reshape((-1))
    pred_lgb = model_lgb.predict(X).reshape((-1))
    pred = (pred_cb + pred_lgb) / 2.0
    
    print(spearmanr(pred, y).correlation)
    splits = [-np.inf, np.inf]
    
    pred_sorted = np.sort(pred)
    for i in range(BINS_COUNT - 1):
        cur_score = -2.0
        best_score = -2.0
        best_ind = -1
        for cur in range(0, len(X), 250):
            if pred_sorted[cur] in splits:
                continue
            vals = pd.cut(pred, np.sort(splits + [pred_sorted[cur]]), labels=False)
            cur_score = spearmanr(vals, y).correlation
            if cur_score > best_score:
                best_score = cur_score
                best_ind = cur
        splits.append(pred_sorted[best_ind])
        prev = best_ind
        print(best_score, best_ind)
    return splits
    
splits = np.sort(gen_splits(X_test, y_test))
print(splits)

def calc_predict(X, y=None, log_file=None):
    pred_cb = model_cb.predict(X).reshape((-1))
    pred_lgb = model_lgb.predict(X).reshape((-1))
    pred = (pred_cb + pred_lgb) / 2.0
    
    res = pd.cut(pred, splits, labels=False)
    
    if y is not None:
        for p in (res, pred, pred_cb, pred_lgb):
            print(spearmanr(y, p), end=' ')
            print('')
            if log_file is not None:
                print(spearmanr(y, p), end=' ', file=log_file)
                print('', file=log_file)
        
    return res

save_y(calc_predict(X_submit))

log_file = open('log.txt', 'w')

save_y(calc_predict(X_test, y_test, log_file), name='test.csv')
save_y(y_test, name='test_true.csv')

log_file.close()