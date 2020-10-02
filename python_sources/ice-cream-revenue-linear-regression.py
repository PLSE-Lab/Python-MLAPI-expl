# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#%%Import data
df=pd.read_csv("/kaggle/input/ice-cream-revenue/IceCreamData.csv")
#%%
print(np.corrcoef(df["Temperature"],df["Revenue"]))
sns.regplot(x="Temperature",y="Revenue",data=df)
plt.show()
#%%Test Train split
from sklearn.model_selection import train_test_split
X=df["Temperature"].values.reshape(-1,1)
y=df["Revenue"].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#%%Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
lin_model=LinearRegression().fit(X_train,y_train)
y_pred=lin_model.predict(X_test)
print("R2:",r2_score(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))