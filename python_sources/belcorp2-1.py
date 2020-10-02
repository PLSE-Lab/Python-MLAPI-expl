# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
# data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

pd.options.display.max_columns=400
from matplotlib_venn import venn2
from sklearn import preprocessing
import matplotlib.pyplot as plt
pd.options.display.max_rows=100
import seaborn as sns
from glob import glob
from tqdm import tqdm
import numpy as np
%matplotlib inline
import itertools
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
## valores missing
def total_missing(data):
    total=data.isnull().sum()
    percent=(data.isnull().sum()/data.isnull().count()*100)
    tt=pd.concat([total,percent],axis=1,keys=["Total","percent"])
    types=[]
    for  col in data.columns:
        dtype=str(data[col].dtype)
        types.append(dtype)
    tt["Types"]=types
    return (np.transpose(tt))

# %% [code]
#frecuencia enconding
def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm,', ',end='')

# %% [code]
#frecuencia enconding
def encode_FE1(df1,cols):
    for col in cols:
        df = df1[[col]]
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        #df2[nm] = df2[col].map(vc)
        #df2[nm] = df2[nm].astype('float32')
        print(nm,', ',end='')

# %% [code]
sub=pd.read_csv("../input/datathon-belcorp-prueba/predict_submission.csv")
camp=pd.read_csv("../input/datathon-belcorp-prueba/campana_consultora.csv")
maestro=pd.read_csv("../input/datathon-belcorp-prueba/maestro_consultora.csv")
venta=pd.read_csv("../input/datathon-belcorp-prueba/dtt_fvta_cl.csv")
ma_producto=pd.read_csv("../input/datathon-belcorp-prueba/maestro_producto.csv")


# %% [code]
print(" dim",sub.shape,camp.shape,maestro.shape,venta.shape,ma_producto.shape)

# %% [markdown]
# EDA

# %% [code]
camp.drop("Unnamed: 0",axis=1,inplace=True)

# %% [code]
camp.head(2)

# %% [code]
## total de missing
total_missing(camp)

# %% [code]
print(camp.shape)
camp.Flagpasopedido.value_counts(normalize=True)

# %% [code]
print(" min ",camp.campana.min()),print(" max ",camp.campana.max());

# %% [code]
len(camp.IdConsultora.unique())

# %% [code]
## tenemos solo 123771 consultras en el periodo de estudio
## lo que debeira de parecer un
#han desjado de comprar 

# %% [code]
print(sub.shape)
len(sub.idconsultora.unique())

# %% [code]
camp["MES"]=(camp["campana"]%100).astype("category")

# %% [code]
camp.head()

# %% [code]
#camp.set_index('MES', inplace=True)
camp.groupby('MES')['Flagpasopedido'].mean().plot(legend=True)
plt.xlabel('campañas')
plt.ylabel('Fraccion de compras por campañas')

# %% [code]
camp.groupby('segmentacion')['Flagpasopedido'].mean().plot(legend=True)

# %% [code]
## calculando prabilidades a nivel de segemntacion
camp.groupby('codigofactura')['Flagpasopedido'].mean().plot(legend=True)

# %% [code]
### ND <> no definido?
camp["codigofactura"].replace("ND",np.nan,inplace=True)

# %% [code]
sub["idconsultora"].isin(camp["IdConsultora"]).value_counts()

# %% [code]
camp[camp["IdConsultora"]==175391].sort_values(by="campana")

# %% [code]
camp_unique=camp.groupby("IdConsultora",as_index=False).agg({"campana":"max"})

# %% [code]
camp_unique1=camp_unique.copy()

# %% [code]
test=pd.merge(camp_unique1,camp,how="left",on=["IdConsultora","campana"])

# %% [code]
sub=sub.rename(columns={"idconsultora":"IdConsultora"})

# %% [code]
test1=pd.merge(sub,test,how="left",on="IdConsultora")

# %% [code]
test1.shape

# %% [code]
test1.drop("flagpasopedido",axis=1,inplace=True)
test1.drop("Flagpasopedido",axis=1,inplace=True)

# %% [code]
var=test1.iloc[:,0:2]

# %% [code]
camp.head()

# %% [code]
camp[camp["IdConsultora"]==21821].shape

# %% [code]
test1[test1["IdConsultora"]==21821]

# %% [code]
result1 = pd.merge(camp,var,on='IdConsultora',how='right')

# %% [code]
ismael_df=result1[result1['campana_x'] < result1['campana_y']]

# %% [code]
ismael_df[ismael_df["IdConsultora"]==21821]

# %% [code]
ismael_df.drop("campana_y",axis=1,inplace=True)

# %% [code]
ismael_df=ismael_df.rename(columns={"campana_x":"campana"})

# %% [code]
ismael_df["Flagpasopedido"].isnull().sum()

# %% [code]
ismael_df.head()

# %% [code]
cat_frecu=[col for col in ismael_df.columns if ismael_df[col].dtypes==np.object]

# %% [code]
cat_frecu

# %% [code]
encode_FE(ismael_df,test1,cat_frecu)

# %% [code]
ismael_df.head()

# %% [code]
ismael_df["grupo"]="is_train"
test1["grupo"]="is_test"

# %% [code]
df=pd.concat([ismael_df,test1],axis=0,sort=False)

# %% [code]
for col in ["IdConsultora"]:
    temp_df = df[[col]]
    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   
    df[col+'_fq_enc'] =df[col].map(fq_encode)
    #test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)

# %% [code]
df.head()

# %% [code]
cat_camp=[col for col in df.columns if df[col].dtypes==np.object]
cat_camp.remove("grupo")

# %% [code]
## alto missing
df.drop("codigocanalorigen",axis=1,inplace=True)

# %% [code]
df.head()

# %% [code]
var_bin=["flagpasopedidocuidadopersonal","flagpasopedidomaquillaje","flagpasopedidotratamientocorporal","flagpasopedidotratamientofacial","flagpedidoanulado","flagpasopedidofragancias"]

# %% [code]
### EDA VENTA
venta.head()

# %% [code]
df.shape

# %% [code]
venta[venta["idconsultora"]==638877]

# %% [code]
venta_feature=[col for col in venta.columns if col not in ["campana","idconsultora"]]
venta_feature

# %% [code]
print(venta.shape)
len(venta["idconsultora"].unique())

# %% [code]
print(" fecha",venta["campana"].min(),venta["campana"].max())

# %% [code]
df[df["IdConsultora"]==544706]

# %% [code]
venta[venta["idconsultora"]==544706]

# %% [code]
c=['count','nunique']
n=['mean','max','min','sum','std']
n1=["sum"]
nn=['mean','max','min','sum','std','quantile']
agg_c={'codigotipooferta':c,'canalingresoproducto':c,'codigopalancapersonalizacion':c,"grupooferta":c,'descuento':n,'ahorro':n,'preciocatalogo':n,'realanulmnneto':n1,'realdevmnneto':n1,'realuuanuladas':n1,
       "realuudevueltas":n1,"realuufaltantes":n1,'realuuvendidas':n,'realvtamnfaltneto':n1,'realvtamnneto':nn,'realvtamncatalogo':nn,'realvtamnfaltcatalogo':n1}
venta1=venta.groupby(['campana','idconsultora']).agg(agg_c)
venta1.head()

# %% [code]
venta1.columns=['F_' + '_'.join(col).strip() for col in venta1.columns.values]
venta1.reset_index(inplace=True)
venta1.head()

# %% [code]
venta1=venta1.rename(columns={"idconsultora":"IdConsultora"})

# %% [code]
df=pd.merge(df,venta1,how="left",on=["campana","IdConsultora"])

# %% [code]
maestro.drop("Unnamed: 0",axis=1,inplace=True)

# %% [code]
maestro.head()

# %% [code]
maestro.shape

# %% [code]
len(maestro["IdConsultora"].unique())

# %% [code]
maestro['antiguedad']=(maestro["campanaultimopedido"]//100)-(maestro["campanaingreso"]//100)

# %% [code]
maestro["dia_ingreso"]=maestro["campanaingreso"]%100
maestro["dia_ultimo"]=maestro["campanaultimopedido"]%100

# %% [code]
maestro.drop("campanaprimerpedido",axis=1,inplace=True)

# %% [code]
maestro=maestro.rename(columns={"idConsultora":"IdConsultora"})

# %% [code]
df=pd.merge(df,maestro,how="left",on="IdConsultora")

# %% [code]
cat_feature=[col for col in df.columns if df[col].dtypes==np.object]
cat_feature.remove("grupo")

# %% [code]
ma_producto.drop("Unnamed: 0",axis=1,inplace=True)

# %% [code]
for col in cat_feature:
    df[col], _ = pd.factorize(df[col])

# %% [code]
cat_feature

# %% [code]
# # GROUP AGGREGATION NUNIQUE
# def encode_AG2(main_columns, uids, train_df=X_train, test_df=X_test):
#     for main_column in main_columns:  
#         for col in uids:
#             comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)
#             mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
#             train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')
#             test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')
#             print(col+'_'+main_column+'_ct, ',end='')

# %% [code]
# def encode_AG2(main_columns, uids, train_df=df):
#     for main_column in main_columns:  
#         for col in uids:
#             comb = train_df[[col]]+[main_column]
#             mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
#             train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')
#             #test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')
#             print(col+'_'+main_column+'_ct, ',end='')

# %% [code]
train=df[df["grupo"]=="is_train"]
test=df[df["grupo"]=="is_test"]

# %% [code]
def encode_AG(main_columns, uids, aggregations=['mean'], train_df=train, test_df=test, 
              fillna=True, usena=False):
    # agregaccion estadistico
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')
                
                if fillna:
                    train_df[new_col_name].fillna(-1,inplace=True)
                    test_df[new_col_name].fillna(-1,inplace=True)
                
                print("'"+new_col_name+"'",', ',end='')
                


# %% [code]
del df,ma_producto,maestro,

# %% [code]
encode_AG(['flagactiva'],['IdConsultora'],['mean'],train,test,fillna=True,usena=True)

# %% [code]
features=[col for col in train.columns if col not in ["IdConsultora","campana","Flagpasopedido","grupo"] ]

# %% [code]
train["Flagpasopedido"]=train["Flagpasopedido"].astype(np.int16)
y=train["Flagpasopedido"]

# %% [code]
print('Train and test columns: {} {}'.format(len(train[features].columns), len(test[features].columns)))

# %% [code]
random_state = 42
params = {
    "objective" : "binary",
    "metric" : "auc",
    'learning_rate': 0.1,
    "max_depth" : -1,
    "num_leaves" :32,
    'num_threads': 15,
    "bagging_freq": 5,
    "bagging_fraction" : 0.7,
    "feature_fraction" : 0.5,
    "min_child_samples":120,
    "min_data_in_leaf": 150,
    "min_sum_heassian_in_leaf": 20,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}

# %% [code]
random_state=44000
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=256)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, y.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=y.iloc[val_idx])


    clf = lgb.train(params, trn_data, num_boost_round=2000, valid_sets = [trn_data, val_data], verbose_eval=25, early_stopping_rounds =50)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) /folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(y, oof)))


# %% [code]
cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:150].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,28))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
plt.savefig('FI.png')

# %% [code]
sub = pd.DataFrame({"IdConsultora":test["IdConsultora"].values})
sub["Flagpasopedido"] =predictions
sub.to_csv("submission_ligthgbm.csv", index=False,sep=",")

# %% [code]
sub = pd.DataFrame({"IdConsultora":train["IdConsultora"].values})
sub["Flagpasopedido"] =oof
sub.to_csv("submission_train.csv", index=False,sep=",")