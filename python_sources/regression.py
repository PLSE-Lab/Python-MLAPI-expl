#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[ ]:


dataset= pd.read_csv("/kaggle/input/calcofi/bottle.csv")


# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


dataset.columns


# In[ ]:


dataset.isnull().sum().plot()


# In[ ]:


for col in dataset.columns:
#     if ((dataset[col].isnull().sum()) == 0):
#         print("{0} : ".format(col),dataset[col].isnull().sum())
    if ((dataset[col].isnull().sum()) > 300000):
#         print("{0} : ".format(col),dataset[col].isnull().sum())
        print("'"+col+"',")


# In[ ]:


dataset.drop(['BtlNum','T_qual','S_qual','O_qual','SThtaq','O2Satq','ChlorA','Phaeop','PO4uM','PO4q','SiO3uM','SiO3qu','NO2uM','NO2q','NO3uM','NO3q','NH3uM','C14As1','C14A1p','C14As2','C14A2p','DarkAs','DarkAp','MeanAs','MeanAp','IncTim','LightP','R_SIO3','R_PO4','R_NO3','R_NO2','R_NH4','R_CHLA','R_PHAEO','R_SAMP','DIC1','DIC2','TA1','TA2','pH2','pH1','DIC Quality Comment'], axis = 1, inplace=True)


# In[ ]:


dataset.shape


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.isnull().sum().plot()


# In[ ]:


for col in dataset.columns:
    if(dataset[col].dtype !=  "object"):
        mean_data = dataset[col].mean()
        dataset[col].fillna(mean_data, inplace=True)


# In[ ]:


dataset.head()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


work_data = dataset.select_dtypes(exclude="object")
work_data.columns


# In[ ]:


work_data.head()


# In[ ]:


work_data.drop(['R_Depth', 'R_TEMP', 'R_POTEMP', 'R_SALINITY', 'R_SIGMA', 'R_SVA',
       'R_DYNHT', 'R_O2', 'R_O2Sat', 'R_PRES'], axis=1, inplace=True)


# In[ ]:


work_data.columns


# In[ ]:


data = work_data.corr()["Salnty"].values
for index, value in enumerate(data):
    value = value * 100
    if(value > 20 or value < -20):
        print(index+1, value)


# In[ ]:


work_data.shape

work_data.head()
# In[ ]:


work_data = work_data.iloc[:500,:]


# In[ ]:


sns.pairplot(work_data, vars=['Depthm', 'T_degC', "Salnty",'STheta', 'T_prec'], kind="reg")


# In[ ]:


sns.distplot(work_data["Salnty"])


# In[ ]:


plt.boxplot(work_data["Salnty"])


# In[ ]:


# X = work_data.drop(["Salnty"], axis=1)
X = work_data[["T_degC"]]
Y = work_data[["Salnty"]]


# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 5)


# In[ ]:


xtest.shape


# In[ ]:


sns.lmplot(x="Salnty", y="T_degC", data=work_data,
           order=2, ci=None)


# In[ ]:


model = LinearRegression()
model.fit(xtrain, ytrain)


# In[ ]:


ypredict = model.predict(xtest)


# In[ ]:


plt.scatter(xtest, ytest, color='r')
plt.plot(xtest, ypredict, color='g')
plt.show()


# In[ ]:


ypredict.shape


# In[ ]:


accuracy = model.score(xtest, ytest)
print(accuracy * 100)


# In[ ]:


print(mean_absolute_error(ypredict, ytest))


# In[ ]:


print(mean_squared_error(ypredict, ytest))


# In[ ]:


print(np.sqrt(mean_squared_error(ypredict, ytest)))


# In[ ]:


print(r2_score(ytest, ypredict))

