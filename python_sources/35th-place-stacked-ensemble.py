#!/usr/bin/env python
# coding: utf-8

# A lot of features and a small training dataset makes this comptetion a challenge to generalize for the samples with different behaviour. My teammate [@kumar_shubham](https://www.kaggle.com/ks2019) found that site 2 samples can be effectively predicted by adding the difference of mean of site-1 and mean of known site-2 samples to site-2 samples, other details can be found in [this](https://www.kaggle.com/ks2019/35th-place-blend-best-3) kernel. This kernel has the stacked ensemble which we used for predictions. <br>
# (Note: We used [this](https://www.kaggle.com/ttahara/trends-simple-nn-baseline) public kernel for NN predictions.)

# # **Level-1**
# At level-1,
# + We have 46 different models
# + 7 fold cross-validation is used
# + Hyper-parameters of each model are tuned using bayesian hyperparameter optimization
# + Models are ElasticNet, Lasso, Ridge, SVR, Bagging Regressor, Bayesian Ridge, Ard Regression, Kernel Ridge, NuSVR, Linear SVR, GLM (power=0, power=1, and power=2), MultiTasking ElasticNet, MultiTasking Lasso and MultiTasking Neural Network
# + Three different variants of each model are prepared using three different data pre-processing techniques - scaling fnc features, feature standarization and minmax scaling of features. 

# # Level - 2 & 3
# At level 2,
# + We have 6 different models
# + Models are - Bayesian Ridge, Bagging Regressor, ElasticNet, Ridge, SVR, and NuSVR
# 
# At level 3, we took the weighted average of predictions by 6 different models of level two, where weights are given on the basis of their out-of-fold prediction scores.

# In[ ]:


import numpy as np
import pandas as pd
import random
import gc

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import ElasticNet, Ridge, Lasso, BayesianRidge, ARDRegression
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.ensemble import BaggingRegressor


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


def metric(y_true, y_pred):
    m1 = np.sum(np.abs(y_true - y_pred), axis=0)
    mae = m1/len(y_true)
    return mae, np.mean(m1/np.sum(y_true, axis=0))


# In[ ]:


## Load test and train fnc and loading features

fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")

labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()

df.shape, test_df.shape


# In[ ]:


## Shift features of site two samples by the difference of mean of site 1 and known site 2 samples

df = df.reset_index(drop = True)
test_df = test_df.reset_index(drop = True)

features = loading_features + fnc_features
features = [i for i in features if i!='IC_20']

site_2_samples = pd.read_csv('../input/trends-assessment-prediction/reveal_ID_site2.csv')['Id'].to_list()

test_df_1 = test_df[test_df.columns]
test_df_2 = test_df[test_df.columns]
test_df_2[features] = test_df_2[features] + df[features].mean() - test_df.set_index('Id').loc[site_2_samples][features].mean() 

A = [0]*(len(df)//7)+[1]*(len(df)//7) + [2]*(len(df)//7)+[3]*(len(df)//7) + [4]*(len(df)//7)+[5]*(len(df)//7) + [6]*(len(df) - 6*len(df)//7 + 3)
random.seed(242)
random.shuffle(A)

df['fold_column'] = A

df[['Id','fold_column']].to_csv('Fold.csv')


# # Level - 1

# In[ ]:


## Parameter dictionary of all the models
## Bayesian Optimization is performed to get the parameter values

parameters = {"svr":{"age": {"scale": [492.5262413687941, 69.60339043492256], "standarize": [26.68867995074479] ,"minmax": [24.77771582775962]} ,
                     "domain1_var1": {"scale": [669.9958281136152, 10.129864960288367], "standarize": [6.444806955515817],"minmax": [4.80302732891827] },
                     "domain1_var2": {"scale": [10.0, 0.5900323195931239], "standarize": [0.419552161343763] ,"minmax": [0.35000260353760754] },
                     "domain2_var1": {"scale": [187.26035099407156, 6.209128731900369], "standarize": [3.3428562888199176] ,"minmax": [2.714687711798464]},
                     "domain2_var2": {"scale": [217.2048608764603, 3.495537661134675] , "standarize": [1.8514027605800645] ,"minmax": [1.715123371502329] }},
             "enet":{"age": {"scale": [1043.096643138724, 1e-05, 1.0], "standarize": [0.07623488779397786, 0.5521844228202069],"minmax": [0.006883473190049635, 1.0]} ,
                     "domain1_var1": {"scale": [122.24848596303966, 1e-05, 0.01083596277157703] , "standarize": [0.2989438656376394, 0.5049192027013706] ,"minmax":[0.021381372803152998, 1.0]  },
                     "domain1_var2": {"scale": [203.9103318225802, 2.5638626472832602e-05, 2.1139739815894028e-05], "standarize": [1.0, 1.0],"minmax": [0.24025111111794356, 1.4665060755095122e-05]},
                     "domain2_var1": {"scale": [97.3794261549355, 1e-05, 0.03158534564456641], "standarize": [0.616837113004086, 0.11581009464872395],"minmax": [0.018701442894351585, 0.5800401393240704]},
                     "domain2_var2": {"scale": [135.56649737157719, 1.0625964729570441e-05, 0.007895224389801971], "standarize": [0.5318561320573426, 1.0] ,"minmax": [0.05113816834656483, 2.2729917560510718e-05]}},
             "lasso":{"age": {"scale": [87.27496554104196, 0.00011811992287947767], "standarize": [0.053591469701310505],"minmax": [0.007091211302720107]} ,
                     "domain1_var1": {"scale": [4493.160497947272, 1e-05], "standarize": [0.16738307730624005],"minmax": [0.021483066649991587]},
                     "domain1_var2": {"scale": [7403.58647502064, 0.0003124334926440588], "standarize": [0.3415539161049886],"minmax": [0.05052060866913413]},
                     "domain2_var1": {"scale": [3014.1643937179506, 1e-05], "standarize": [0.15570482251801265],"minmax": [0.018670197689250108]},
                     "domain2_var2": {"scale": [572.2741674526715, 7.235349978379449e-05], "standarize": [0.19463950211830205],"minmax": [0.026324889974712697]}},
             "ridge":{"age": {"scale": [2981.19407531989, 1e-05], "standarize": [835.5424030320825],"minmax": [13.241076418640132]} ,
                     "domain1_var1": {"scale": [2974.9135224877314, 9.207208910407203e-05], "standarize": [6097.898352449914],"minmax": [109.1407510448333]},
                     "domain1_var2": {"scale": [1165.1948996221117, 0.0633286354054813], "standarize": [10000.0],"minmax": [1130.1525442567247]},
                     "domain2_var1": {"scale": [221.72697425311426, 0.010485943145327647], "standarize": [7509.783693100307],"minmax": [140.27822544908207]},
                     "domain2_var2": {"scale": [397.91938242690884, 0.0039534304599317775], "standarize": [10000.0],"minmax": [256.18335952365607]}},
             "bagging_regressor":{"age": {"scale": [836.8450355986731, 5.842699153641068e-05, 99, 0.7980778723979312, 0.8553728294699361], "standarize": [1e-06, 100, 0.6, 1.0] ,"minmax": [0.7772232840469472, 100, 0.6, 0.8339762285063499]} ,
                     "domain1_var1": {"scale": [6226.685140116834, 8.062292860122505e-06, 20, 0.5985935749843405, 0.7023122872651539], "standarize": [0.08296550200208345, 52, 1.0, 0.24081559768307517],"minmax": [ 0.00013331958105521457, 58, 0.577752531444629, 0.17834040671908657] },
                     "domain1_var2": {"scale": [8508.354139514926, 1.8589590180720532e-05, 90, 0.4640510986867237, 0.6414842839258486], "standarize": [1e-06, 100, 0.5977205133375352, 0.1] ,"minmax": [ 1.0, 84, 1.0, 0.1] },
                     "domain2_var1": {"scale": [1558.1747599874511, 8.471138739339728e-05, 57, 0.7486934751239087, 0.6064330217872465], "standarize": [8.809009860834274e-05, 21, 0.8107088693735142, 0.16684953796049679] ,"minmax": [0.0031809362709809643, 63, 0.8448463504161474, 0.18151653061841017]},
                     "domain2_var2": {"scale": [249.06441572172292, 0.0026501818830299132, 71, 0.39085489144145025, 0.8101067709206109] , "standarize": [1.0, 56, 1.0, 0.12137532990594137] ,"minmax": [1e-06, 83, 1.0, 0.12854547267982455] }},
             "bayesian_ridge":{"age": {"scale": [169.93122783930556], "standarize": [26.68867995074479] ,"minmax": [24.77771582775962]} ,
                     "domain1_var1": {"scale": [200], "standarize": [6.444806955515817],"minmax": [4.80302732891827] },
                     "domain1_var2": {"scale": [146.56814055758105], "standarize": [0.419552161343763] ,"minmax": [0.35000260353760754] },
                     "domain2_var1": {"scale": [200], "standarize": [3.3428562888199176] ,"minmax": [2.714687711798464]},
                     "domain2_var2": {"scale": [147.03591195143287] , "standarize": [1.8514027605800645] ,"minmax": [1.715123371502329] }},
             "ardregression":{"age": {"scale": [169.93122783930556], "standarize": [26.68867995074479] ,"minmax": [24.77771582775962]} ,
                     "domain1_var1": {"scale": [200], "standarize": [6.444806955515817],"minmax": [4.80302732891827] },
                     "domain1_var2": {"scale": [146.56814055758105], "standarize": [0.419552161343763] ,"minmax": [0.35000260353760754] },
                     "domain2_var1": {"scale": [200], "standarize": [3.3428562888199176] ,"minmax": [2.714687711798464]},
                     "domain2_var2": {"scale": [147.03591195143287] , "standarize": [1.8514027605800645] ,"minmax": [1.715123371502329] }},
             "kernel_ridge":{"age": {"scale":  [218.25312369655543, 1.4956781741767941e-05], "standarize": [26.68867995074479] ,"minmax": [0.016034788688041528]} ,
                     "domain1_var1": {"scale": [621.3861221113601, 2.8212635925665707e-06] , "standarize": [6.444806955515817],"minmax": [0.14177151020560527] },
                     "domain1_var2": {"scale":  [610.6037663955775, 2.26327745611855e-05], "standarize": [0.419552161343763] ,"minmax": [1.0] },
                     "domain2_var1": {"scale": [1177.188601274852, 1e-06], "standarize": [3.3428562888199176] ,"minmax": [0.19666007449883396]},
                     "domain2_var2": {"scale": [2065.8127095964114, 8.714379924658953e-06] , "standarize": [1.8514027605800645] ,"minmax": [0.36572817396633966]  }},
             "nusvr":{"age": {"scale":  [413.66352024507285, 0.30586824827165876, 192.7364622961058], "standarize": [0.7661214333172722, 31.23665506792682] ,"minmax": [0.8255583612455453, 22.779047246626615]} ,
                     "domain1_var1": {"scale": [208.522847268132, 0.7538169413317486, 2.3961220451735383] , "standarize": [0.7135776229598471, 6.4016655709129475],"minmax": [0.6912174356470082, 6.536167703644929] },
                     "domain1_var2": {"scale":  [216.1974537027112, 0.8173591733778397, 0.5482189695163461], "standarize": [0.961358505480476, 0.3282382578756948] ,"minmax": [0.9991027770074039, 0.17301045865711506] },
                     "domain2_var1": {"scale": [261.73100829747614, 0.6225071836999567, 2.4341415966735354] , "standarize": [0.6315442586778667, 4.296746217326582] ,"minmax": [0.5835277539151535, 4.605191484904114] },
                     "domain2_var2": {"scale": [223.1976334943342, 0.7807010597122734, 3.1841909940192195] , "standarize": [0.8691858535045746, 1.5804030121351031] ,"minmax": [0.7194666704568422, 1.9191151694259774]}},
             "linear_svr":{"age": {"scale":   [131.90871225961428, 800.5475555641115] , "standarize": [0.7661214333172722, 31.23665506792682] ,"minmax": [0.43568842406380115]} ,
                     "domain1_var1": {"scale": [10000.0, 456.18462937926853] , "standarize": [0.7135776229598471, 6.4016655709129475],"minmax": [0.057999064969975776]  },
                     "domain1_var2": {"scale":  [375.9609066943138, 139.43946061565356] , "standarize": [0.961358505480476, 0.3282382578756948] ,"minmax": [0.007176080716423259]},
                     "domain2_var1": {"scale": [10000.0, 420.34207749811713] , "standarize": [0.6315442586778667, 4.296746217326582] ,"minmax": [0.07061991957605784]},
                     "domain2_var2": {"scale": [10.0, 2.1357134106514724] , "standarize": [0.8691858535045746, 1.5804030121351031] ,"minmax": [0.023951883407266204]}},
             "power0":{"age": {"scale":   [157.33583611404953, 1e-06] , "standarize": [0.16597684517161287] ,"minmax": [0.002645919092791747]} ,
                     "domain1_var1": {"scale": [288.59271602571766, 2.625473985218113e-06] , "standarize": [1.0],"minmax": [0.023180405930276215]},
                     "domain1_var2": {"scale": [159.97687787439807, 1.9524622248687853e-05] , "standarize": [1.0] ,"minmax": [0.23941016046328426]},
                     "domain2_var1": {"scale": [135.6366302805808, 4.899060757019108e-06] , "standarize": [1.0] ,"minmax": [0.03016669508052957]},
                     "domain2_var2": {"scale": [181.663465549777, 4.495467291813097e-06] , "standarize": [1.0] ,"minmax": [0.05240332035181045]}},
             "power1":{"age": {"scale":   [109.4692528739947, 0.00011800535892912307] , "standarize": [0.7661214333172722, 31.23665506792682] ,"minmax": [0.19027501418938883]} ,
                     "domain1_var1": {"scale": [193.24898317054877, 0.00021547271211618736] , "standarize": [0.7135776229598471, 6.4016655709129475],"minmax": [1.0]  },
                     "domain1_var2": {"scale": [31.728634639636926, 0.020224460668951207] , "standarize": [0.961358505480476, 0.3282382578756948] ,"minmax": [1.0]},
                     "domain2_var1": {"scale": [124.57018781462524, 0.00027272661869402835] , "standarize": [0.6315442586778667, 4.296746217326582] ,"minmax": [1.0]},
                     "domain2_var2": {"scale": [169.1220187616997, 0.0002639424761593445] , "standarize": [0.8691858535045746, 1.5804030121351031] ,"minmax": [1.0]}},
             "power2":{"age": {"scale":   [184.70387463740076, 1e-06] , "standarize": [0.26352752038983795] ,"minmax": [0.004841131997346627]} ,
                     "domain1_var1": {"scale": [55.88885309194922, 5.791934559291647e-06] , "standarize": [1.0],"minmax": [0.027400581972783424]},
                     "domain1_var2": {"scale":  [433.90185279217513, 0.1205169267601079] , "standarize": [1.0] ,"minmax": [0.2975556654409895]},
                     "domain2_var1": {"scale": [31.997927215173238, 0.00012695504821855731] , "standarize": [1.0] ,"minmax": [0.03098204335491305]},
                     "domain2_var2": {"scale": [24.90099861684024, 0.0001322209114073255] , "standarize": [1.0] ,"minmax": [0.054167605149726517]}},
             'multi_lasso':{'scaling': [3791.751469250612, 1e-05], 'standarize': [0.21959169710006501], 'minmax': [0.029412486001607965]},
             'multi_enet':{'scaling': [3791.751469250612, 1e-05, 0.9999], 'standarize': [0.2196602883675783, 0.9999], 'minmax': [0.028758339435775493, 0.9999]}}


# In[ ]:


targets = ["age","domain1_var1","domain1_var2","domain2_var1","domain2_var2"]
models = ['enet','lasso','ridge', 'svr', 'bagging_regressor', 'bayesian_ridge', 'ardregression', 'kernel_ridge', 'nusvr', 'linear_svr', 'power0', 'power1', 'power2']
data_preprocessings = ['scale','standarize','minmax']

no_folds = 7

results_test_1 = pd.DataFrame()
results_test_2 = pd.DataFrame()
results_train = df[['Id','fold_column']+targets]
results_test_1['Id'] = test_df_1['Id']
results_test_2['Id'] = test_df_2['Id']

w = {"age":0.3, "domain1_var1":0.175, "domain1_var2":0.175, "domain2_var1":0.175,"domain2_var2":0.175}

for target in targets:
    for model in models:
        for pp in data_preprocessings:
            print("Model: ",model," | ","Target: ",target," | ","Preprocessing: ",pp)
            y_oof = np.zeros(df.shape[0])
            y_test_1 = np.zeros((test_df.shape[0], 7))
            y_test_2 = np.zeros((test_df.shape[0], 7))
            for i in range(no_folds):
                gc.collect()
                train_df = df[df['fold_column']!=i]
                val_df = df[df['fold_column']==i]
                train_df = train_df[train_df[target].notnull()]
                parameter = parameters[model][target][pp]
                if pp == 'scale':
                    train_df[fnc_features] *= 1/parameter[0]
                    val_df[fnc_features] *= 1/parameter[0]
                    t1 = test_df_1[features]
                    t1[fnc_features] *= 1/parameter[0]
                    t2 = test_df_2[features]
                    t2[fnc_features] *= 1/parameter[0]
                    t1 = t1.values
                    t2 = t2.values
                    parameter = parameter[1:]
                    
                if pp == 'standarize':
                    scaler = StandardScaler()
                    scaler.fit(train_df[features])
                    train_df[features] = scaler.transform(train_df[features])
                    val_df[features] = scaler.transform(val_df[features])
                    t1 = scaler.transform(test_df_1[features])
                    t2 = scaler.transform(test_df_2[features])
                    
                if pp == 'minmax':
                    scaler = MinMaxScaler()
                    scaler.fit(train_df[features])
                    train_df[features] = scaler.transform(train_df[features])
                    val_df[features] = scaler.transform(val_df[features])
                    t1 = scaler.transform(test_df_1[features])
                    t2 = scaler.transform(test_df_2[features])
                    
                if model == 'svr':
                    M = SVR(C=parameter[0], cache_size=3000.0)
                
                if model == 'enet':
                    M = ElasticNet(alpha = parameter[0], l1_ratio = parameter[1], normalize = False, max_iter = 5000, tol = 1e-5)
                    
                if model == 'lasso':
                    M = Lasso(alpha = parameter[0], normalize = False, max_iter = 5000, tol = 1e-5)
                
                if model == 'ridge':
                    M = Ridge(alpha = parameter[0], normalize = False, max_iter = 5000, tol = 1e-5)
                    
                if model == 'bagging_regressor':
                    M = BaggingRegressor(Ridge(alpha = parameter[0]), n_estimators=parameter[1], random_state=42, 
                                     max_samples=parameter[2], max_features=parameter[3])
                
                if model == 'bayesian_ridge':
                    M = BayesianRidge( n_iter=1000, tol=10e-05)
                    
                if model == 'ardregression':
                    M = ARDRegression()
                    
                if model == 'kernel_ridge':
                    M = KernelRidge(alpha= parameter[0],kernel ='rbf')
                    
                if model == 'nusvr':
                    M = NuSVR(nu=parameter[0], C=parameter[1])
                    
                if model == 'linear_svr':
                    M = LinearSVR(C=parameter[0])
                    
                if model in ['power0', 'power1', 'power2']:
                    M = TweedieRegressor(power = int(model[-1]), alpha = parameter[0])
                    
                M.fit(train_df[features].values, train_df[target].values)
                
                y_oof[val_df.index] = M.predict(val_df[features])
                y_test_1[:, i] = M.predict(t1)
                y_test_2[:, i] = M.predict(t2)
                
            results_train[target+'.'+model+'.'+pp] = y_oof
            results_test_1[target+'.'+model+'.'+pp] = y_test_1.mean(axis=1)
            results_test_2[target+'.'+model+'.'+pp] = y_test_2.mean(axis=1)
            
            score = metric(df[df[target].notnull()][target].values, results_train[df[target].notnull()][target+'.'+model+'.'+pp].values)
            print('Score: ',score)
            print()


# In[ ]:


## For multi-tasking models

for model in ['multi_enet', 'multi_lasso']:
    for pp in data_preprocessings:
        print("Model: ",model," | ","Preprocessing: ",pp)
        y_oof = np.zeros((df.shape[0], 5))
        y_test_1 = np.zeros((test_df.shape[0], 5))
        y_test_2 = np.zeros((test_df.shape[0], 5))
        for fold_no in range(7):
            gc.collect()
            train_df = df[df['fold_column']!=fold_no]
            val_df = df[df['fold_column']==fold_no]
            for t in target:
                train_df = train_df[train_df[t].notnull()]
            parameter = parameters[model][pp]
            if pp == 'scaling':
                train_df[fnc_features] *= 1/parameter[0]
                val_df[fnc_features] *= 1/parameter[0]
                t1 = test_df_1[features]
                t1[fnc_features] *= 1/parameter[0]
                t2 = test_df_2[features]
                t2[fnc_features] *= 1/parameter[0]
                t1 = t1.values
                t2 = t2.values
                parameter = parameter[1:]

            if pp == 'standarize':
                scaler = StandardScaler()
                scaler.fit(train_df[features])
                train_df[features] = scaler.transform(train_df[features])
                val_df[features] = scaler.transform(val_df[features])
                t1 = scaler.transform(test_df_1[features])
                t2 = scaler.transform(test_df_2[features])

            if pp == 'minmax':
                scaler = MinMaxScaler()
                scaler.fit(train_df[features])
                train_df[features] = scaler.transform(train_df[features])
                val_df[features] = scaler.transform(val_df[features])
                t1 = scaler.transform(test_df_1[features])
                t2 = scaler.transform(test_df_2[features])
                
            if model == 'multi_enet':
                M = MultiTaskElasticNet(alpha = parameter[0], l1_ratio = parameter[1],
                                         normalize = False,
                                         max_iter = 5000, tol = 1e-5)
            
            if model == 'multi_lasso':
                M = MultiTaskLasso(alpha = parameter[0],
                                         normalize = False,
                                         max_iter = 5000, tol = 1e-5)
            
            M.fit(train_df[features].values, train_df[target].values)
            y_oof[val_df.index] = M.predict(val_df[features])
            y_test_1 += M.predict(t1)
            y_test_2 += M.predict(t2)
            
        for x, t in enumerate(target):
            results_train[t+'.'+model+'.'+pp] = y_oof[:,x]
            results_test_1[t+'.'+model+'.'+pp] = y_test_1[:,x]/7
            results_test_2[t+'.'+model+'.'+pp] = y_test_2[:,x]/7
            mae, score = metric(results_train[df[t].notnull()][t].values, results_train[df[t].notnull()][t+'.'+model+'.'+pp].values)
            print('Feature: ', t, " | ", 'Score: ', score, " | ", 'MAE: ', mae)
        print()


# In[ ]:


results_train.to_csv('OOF_preds.csv')
results_test_1.to_csv('Test_preds_1.csv')
results_test_2.to_csv('Test_preds_2.csv')


# In[ ]:


oof = pd.read_csv('OOF_preds.csv')

for i in sorted(oof.columns):
    t = i.split('.')[0]
    if (t in targets) and i!=t:
        mae, score = metric(oof[oof[t].notnull()][t].values, oof[oof[t].notnull()][i].values)
        print("Target", t ,' | ',i ,' | ','MAE', mae, '|', 'Score', score)


# # Level-2

# In[ ]:


## Correlation matrices

for i in targets:
    c = []
    print("#######################",i,"############################")
    for j in oof.columns:
        if (i in j) and (j!=i):
            c.append(j)
    corrMatrix = oof[c].corr()
    sns.heatmap(corrMatrix)
    plt.show()


# In[ ]:


test_1 = pd.read_csv('Test_preds_1.csv')
test_2 = pd.read_csv('Test_preds_1.csv')


# In[ ]:


model_names = ['bayesian_ridge', 'bagging_regressor', 'ElasticNet', 'Ridge', 'SVR',  'NuSVR']


# In[ ]:


results_test_1 = pd.DataFrame()
results_test_2 = pd.DataFrame()
results_train = oof[['Id','fold']+targets]
results_test_1['Id'] = test_1['Id']
results_test_2['Id'] = test_2['Id']

w = {'age': 0.3,'domain1_var1': 0.175,'domain1_var2': 0.175,'domain2_var1': 0.175,'domain2_var2': 0.175}

overal_score = 0
for target in targets:
    print("Target",target)
    all_scores = []
    y_oof = np.zeros(oof.shape[0])
    y_test_1 = np.zeros((test_1.shape[0], len(model_names)))
    y_test_2 = np.zeros((test_2.shape[0], len(model_names)))
    features = [i for i in oof.columns if (target in i) and (i!=target)]
    y_oof = np.zeros((oof.shape[0],len(model_names)))
    for k,model in enumerate([BayesianRidge(), BaggingRegressor(Ridge()),ElasticNet(max_iter=5000, tol=10e-05), Ridge(tol=10e-05), SVR(max_iter=5000, tol=10e-05), NuSVR(tol=10e-05)]):
        for fold_no in range(7):
            gc.collect()
            train_df = oof[oof['fold']!=fold_no]
            val_df = oof[oof['fold']==fold_no]
            train_df = train_df[train_df[target].notnull()]

            model.fit(train_df[features].values, train_df[target].values)

            y_oof[val_df.index,k] = model.predict(val_df[features])
            y_test_1[:, k] += model.predict(test_1[features])/7
            y_test_2[:, k] += model.predict(test_2[features])/7
        
        mae, score = metric(oof[oof[target].notnull()][target].values, y_oof[oof[target].notnull(),k])
        all_scores.append(mae)
        print(model_names[k],mae,score)
    weights = 0 - np.array(all_scores)
    weights = 0.001+weights - weights.min()
    weights = weights/weights.sum()
    TTTT = pd.DataFrame(np.corrcoef(y_oof.T))
    TTTT.index = model_names
    TTTT.columns = model_names
    display(TTTT)
    print(weights)
    results_train[target] = (y_oof*weights).sum(axis=1)  
    results_test_1[target] = (y_test_1*weights).sum(axis=1)  
    results_test_2[target] = (y_test_2*weights).sum(axis=1)  
 
    mae, score = metric(oof[oof[target].notnull()][target].values, results_train[oof[target].notnull()][target].values)
    print('Target: ', target, " | ",'Score: ',score," | ","MAE: ", mae)
    overal_score += w[target]*score
    print()

print('Overall Score: ', overal_score)


# In[ ]:


results_train.to_csv('OOF_preds.csv')
results_test_1.to_csv('Test_preds_1.csv')
results_test_2.to_csv('Test_preds_2.csv')


# In[ ]:


sub_df = pd.melt(results_test_1[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
sub_df


# In[ ]:


sub_df.to_csv("submission_test_1.csv", index=False)


# In[ ]:


sub_df = pd.melt(results_test_2[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
sub_df


# In[ ]:


sub_df.to_csv("submission_test_2.csv", index=False)


# In[ ]:


r1 = pd.read_csv('submission_test_1.csv').set_index('Id')
r2 = pd.read_csv('submission_test_2.csv').set_index('Id')
p = pd.read_csv('../input/classifier9siteprobs/prob_8.csv').drop(columns = ['Unnamed: 0']).set_index('Id')

results = []
for i in tqdm(r1.index):
    
    prob = p.loc[int(i.split('_')[0])]['prob']
    
    if prob>=0.7:
        prob = 1
    if prob<=0.3:
        prob = 0
    
    value = r1.loc[i]['Predicted']*(1-prob) + r2.loc[i]['Predicted']*(prob)
    
    results.append({'Id':i, 'Predicted':value})
    
r = pd.DataFrame(results)
r.to_csv("submission.csv", index=False)

