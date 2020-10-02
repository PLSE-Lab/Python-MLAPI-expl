# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print (os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_data= pd.read_csv("../input/train.csv")
eval_data = pd.read_csv("../input/test.csv")
#print (train_data.head())
#print (len(train_data.columns))
#print (len(eval_data.columns))

# To choose the target - prices

target = ["SalePrice"]

y = train_data[target]

#ev_y = eval_data[target]

# Analizar columnas más importantes
columns = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF","FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

x = train_data[columns]

ev_x = eval_data[columns]

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# To split the model
train_x, val_x, train_y, val_y = train_test_split(x,y,random_state=0)

# 1. To define the model
model = DecisionTreeRegressor()

# 2. To fit the model
model.fit(train_x,train_y)


#print (y.head())
#pred = model.predict(val_x)
#print (mean_absolute_error(val_y, pred))

def get_mae (max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor (max_leaf_nodes = max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return (mae)

valores=[]
mln=[]
for max_leaf_nodes in range(2,1000,10):
    my_mae = get_mae (max_leaf_nodes, train_x, val_x, train_y, val_y)
    mln.append(max_leaf_nodes)
    valores.append(my_mae)
    #print ("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))
import matplotlib.pyplot as plt

#plt.plot(valores)
#plt.show()

""" CREAMOS UN NUEVO MODELO DE PREDICCION: EL RANDOM FOREST """

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor ()
forest_model.fit(train_x,train_y)
preds = forest_model.predict(val_x)
a=mean_absolute_error(val_y,preds)
print (a)

""" NUEVO MODELO DE PREDICCIÓN : REGRESIÓN LOGÍSTICA """

from sklearn.linear_model import LogisticRegression

modelo_log = LogisticRegression ()
modelo_log.fit(train_x, train_y)
preds_log = modelo_log.predict(val_x)
print (len(preds_log))
b=mean_absolute_error(val_y,preds)
print ("La predicción logística es: %d" %(b))

val = [min(valores), a,b]
names = ["Tree","Random_Forest","Reg_Log"]

plt.plot (val)
plt.xlabel("Modelos")
plt.ylabel("Errores")
plt.show()

test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
print (test_data.head())
# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[columns]

# make predictions which we will submit. 
test_preds = forest_model.predict(test_X)
print (test_preds)
# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)