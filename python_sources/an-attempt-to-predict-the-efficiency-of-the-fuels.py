# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

### car fuel consumption

import pandas as pd

df=pd.read_excel(r"../input/measurements2.xlsx")
x = df.iloc[:,[0,2,3,4,5,6,7,8,9]]
y = df.iloc[:,1]

# =============================================================================
# Found missing values in temp_inside and Specials
# Imputing values of temp_inside
#=============================================================================
from sklearn.preprocessing import Imputer
imputer = Imputer()
x.loc[:,['temp_inside']] = imputer.fit_transform(x.loc[:,['temp_inside']])

# =============================================================================
#  Since NaN of specials cannot be imputed hence replacing all NaN by 0
# =============================================================================
x.loc[:,['specials']] = x.loc[:,['specials']].fillna(' ')

# =============================================================================
# Label Encoding
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
x = pd.get_dummies(x, prefix=['specials', 'gas_type'], drop_first=True)

# =============================================================================
# Splitting train test
# =============================================================================
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=9, train_size = 0.8)

# =============================================================================
# Feature Scaling  ---- 
# =============================================================================
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
x_train.iloc[:,0:4] = scale_x.fit_transform(x_train.iloc[:,0:4])
x_test.iloc[:,0:4] = scale_x.fit_transform(x_test.iloc[:,0:4])

# =============================================================================
# Modelliong
# =============================================================================
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

# =============================================================================
# predicting
# =============================================================================
y_pred = lin_reg.predict(x_test)

# =============================================================================
# Metrics
# =============================================================================
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_pred)


# =============================================================================
# average of car when using SP98
# =============================================================================
y_sp98 = lin_reg.predict(x_test[x_test.loc[:,'gas_type_SP98']==1])
y_E10 = lin_reg.predict(x_test[x_test.loc[:,'gas_type_SP98']==0])

# =============================================================================
# This shows that E10 takes 0.33l more per 100kms than sp98
# =============================================================================
y_sp98.mean() 
y_E10.mean()
diff_in_fuel_100km = y_E10.mean() - y_sp98.mean() 

# =============================================================================
# money matter
# =============================================================================
# per 100km on e10 will cost 
e10_100km = y_E10.mean() * 1.38
sp98_100km = y_sp98.mean() * 1.46

# =============================================================================
# difference shows that for a 100km drive e10 costs more than sp98
# =============================================================================
cost_diff = e10_100km - sp98_100km