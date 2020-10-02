#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import fun_py as fp


# In[ ]:


train= pd.read_csv("../input/credit-score-prediction/CreditScore_train.csv")
test= pd.read_csv("../input/credit-score-prediction/CreditScore_test.csv")


# In[ ]:


print("Train Data Shape b4 adding target col : ",train.shape)
print("Test Data Shape b4 adding target col : ",test.shape)


# In[ ]:


train["source"] = "train"
test["source"] = "test"
print("Train Data Shape aftr adding target col : ",train.shape)
print("Test Data Shape aftr adding target col : ",test.shape)


# In[ ]:


df = pd.concat([train,test])


# In[ ]:


fp.data_duplicates(df,0)


# In[ ]:


fp.data_isna(df)


# In[ ]:



#fp.data_groupcols(df)


# In[ ]:


df.shape


# In[ ]:


lst=[]
lst=df.columns


# In[ ]:


row=df.shape[0]
cols=[]


# In[ ]:


len(cols)


# In[ ]:


[cols.append(i) for i in lst if df[i].isnull().sum()/row*100 > 70 ]


# In[ ]:


len(cols)


# In[ ]:


cols


# Removing columns which has more than 70% of NA Values . 10 cols (x098,x155,x242,x255,x256,x257,x259,x295,x302,x304) are removed

# In[ ]:


data=df.drop(cols,axis=1)


# In[ ]:


data.shape


# In[ ]:


fp.data_nullcols(data,0)


# In[ ]:


#fp.data_groupcols(data)


# In[ ]:


pd.options.display.max_rows = 4000

colg=data.corr()['y'].sort_values() > 0.3
coll=data.corr()['y'].sort_values() <-0.3


# In[ ]:


data.shape


# In[ ]:


lstg=[]
lstl=[]
lstg.clear()
lstg.clear()
len(lstg)
len(lstl)
        


# In[ ]:


[lstg.append(i) for i,j in colg.items() if j == True]
[lstl.append(i) for i,j in coll.items() if j == True]


# In[ ]:


print("Length of lstl",len(lstl))
print("Length of lstg",len(lstg))


# In[ ]:


lstd=[]
#lstd=lstg+lstl
print("Length of lstd",len(lstd))


# In[ ]:


lstd=lstg+lstl
print("Length of lstd",len(lstd))


# In[ ]:


data_cols=data.columns


# In[ ]:


#data.corr('x002')['y']
#data['x002'].corr(data['y'])
#df['A'].corr(df['B'])
#pd.options.display.max_columns = 4000
#data.corr()


# In[ ]:


cor_target = abs(data.corr()["y"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target<0.3]
relevant_features


# In[ ]:


lst_key=[]


# In[ ]:


for i,j in relevant_features.items():
    lst_key.append(i)
#print(lst_key.count())


# In[ ]:


len(lst_key)


# In[ ]:


#lstd


# In[ ]:


#lstd.remove('y')
m_cols=[]


# In[ ]:


lstd1= ['x017','x047','x015','x043','x251','x248','x018','x019','x028','x020','x004','x027','x030','x224','x260','x261','x229','x262','x247','x250','x246','x245','x014','x023','x002','x239','x025','x225','x237','x244','x046','x228','x226','x249','x227','x022','x236','x005','x235','x041','x057','x058','x287','x148','x253','x059','x065','x064','x162','x297','x293','x063','x168','x173','x056','x278','x036','x172','x277','x276','x099']


# In[ ]:


len(lstd1)


# In[ ]:


drop_cols=['x062','x066','x067','x068','x069','x070','x071','x072','x073','x074','x075','x076','x077','x078','x079','x080','x081','x082','x083','x084','x085','x086','x087','x088','x089','x090','x091','x092','x093','x094','x095','x096','x097','x100','x101','x102','x103','x104','x105','x106','x107','x108','x109','x110','x111','x112','x113','x114','x115','x116','x117','x118','x119','x120','x121','x122','x123','x124','x125','x126','x127','x128','x129','x130','x131','x132','x133','x134','x135','x136','x137','x138','x139','x140','x141','x142','x143','x144','x145','x146','x147','x149','x150','x151','x152','x153','x154','x156','x157','x158','x159','x160','x161','x163','x164','x165','x166','x167','x169','x170','x171','x174','x175','x176','x177','x178','x179','x180','x181','x182','x183','x184','x185','x186','x187','x188','x189','x190','x191','x192','x193','x194','x195','x196','x197','x198','x199','x200','x201','x202','x203','x204','x205','x206','x207','x208','x209','x210','x211','x212','x213','x214','x215','x216','x217','x218','x219','x220','x221','x222','x223','x230','x231','x232','x233','x234','x238','x240','x241','x243','x252','x254','x258','x263','x264','x265','x266','x267','x268','x269','x270','x271','x272','x273','x274','x275','x279','x280','x281','x282','x283','x284','x285','x286','x288','x289','x290','x291','x292','x294','x296','x298','x299','x300','x301','x303','x001','x003','x006','x007','x008','x009','x010','x011','x012','x013','x016','x021','x024','x026','x029','x031','x032','x033','x034','x035','x037','x038','x039','x040','x042','x044','x045','x048','x049','x050','x051','x052','x053','x054','x055','x060','x061']


# In[ ]:


len(drop_cols)


# In[ ]:


data.shape


# In[ ]:


prep_data=data.copy()


# In[ ]:


prep_data.drop(columns=drop_cols,axis=1,inplace=True)


# In[ ]:


prep_data.shape


# In[ ]:


#prep_data.columns


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
coL=list(prep_data.columns)


# In[ ]:


type(coL)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
train_test_b4_split_data=prep_data.copy


# In[ ]:


null_cols=fp.data_nullcols(prep_data,1)


# In[ ]:


for i in null_cols:
    prep_data[i].fillna(prep_data[i].mean(),inplace=True)


# In[ ]:


train_final = prep_data[prep_data.source=="train"]
test_final = prep_data[prep_data.source=="test"]


# In[ ]:


train_final = train_final.drop(columns='source',axis=1)
test_final = test_final.drop(columns='source',axis=1)
train_final.shape


# In[ ]:


train_final.columns


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# In[ ]:


X = train_final.drop("y", axis=1)
Y = train_final["y"]
print(X.shape)
print(Y.shape)


# In[ ]:


seed      = 42
test_size = 0.30

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


# user variables to tune
folds   = 5
metric  = "neg_mean_absolute_error"

# hold different regression models in a single dictionary
models = {}
models["Linear"]        = LinearRegression()
models["Lasso"]         = Lasso()
models["Ridge"]         = Ridge()
models["ElasticNet"]    = ElasticNet()
models["DecisionTree"]  = DecisionTreeRegressor()
models["KNN"]           = KNeighborsRegressor()
models["RandomForest"]  = RandomForestRegressor()
models["AdaBoost"]      = AdaBoostRegressor()
models["GradientBoost"] = GradientBoostingRegressor()
models["XGBoost"] = XGBRegressor()

# 10-fold cross validation for each model
model_results = []
model_names   = []
for model_name in models:
	model   = models[model_name]
	k_fold  = KFold(n_splits=folds, random_state=seed)
	results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)
	
	model_results.append(results)
	model_names.append(model_name)
	print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))


# In[ ]:


# box-whisker plot to compare regression models
import matplotlib.pyplot as plt 
figure = plt.figure(figsize = (35,15))

figure.suptitle('Regression models comparison')
axis = figure.add_subplot(111)
plt.boxplot(model_results)
axis.set_xticklabels(model_names, rotation = 45, ha="right")
axis.set_ylabel("Mean Absolute Error (MAE)")
plt.margins(0.05, 0.1)


# In[ ]:


model = RandomForestRegressor()
model.fit(X_train,Y_train)

##print("Intercept : ", model.intercept_)
##print("Slope : ", model.coef_)

#Predicting TEST & TRAIN DATA
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

error_percent = np.mean(np.abs((Y_train - train_predict) / Y_train)) * 100
print("MAPE - Mean Absolute Percentage Error (TRAIN DATA): ",error_percent )
Y_train, train_predict = np.array(Y_train), np.array(train_predict)


# In[ ]:


model = RandomForestRegressor()
model.fit(X_test,Y_test)

##print("Intercept : ", model.intercept_)
##print("Slope : ", model.coef_)

#Predicting TEST & TRAIN DATA
#train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

error_percent = np.mean(np.abs((Y_test - test_predict) / Y_test)) * 100
print("MAPE - Mean Absolute Percentage Error (TEST DATA): ",error_percent )
Y_test, test_predict = np.array(Y_test), np.array(test_predict)


# In[ ]:


dtrain_predictions = model.predict(X_train)


# In[ ]:


#Print model report:
print("\nModel Report")
print("RMSE : %.4g" % np.sqrt(mean_squared_error(Y_train, dtrain_predictions)))
    
#Predict on testing data:
#test_final["res_linear"] =  model.predict(X_train)

