#!/usr/bin/env python
# coding: utf-8

# Explore **Google** stocks

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from statsmodels.tsa.arima_model import ARIMA
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


data = pd.read_csv("../input/Google.csv",sep=",",parse_dates=['Date'],index_col='Date')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.index


# In[ ]:


data.loc[data['High']==1228.880000]


# In[ ]:


data.loc[data['High']==101.740000]


# In[ ]:


data.hist(figsize=(12,10))


# In[ ]:


data.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), fmt="f",ax=ax)


# In[ ]:


plt.figure(figsize=(12,6))
#Volume	Ex-Dividend	Split Ratio	Adj. Open	Adj. High	Adj. Low	Adj. Close	Adj. Volume
plt.plot(data['High'])
plt.title("Date vs Stock High Price",fontsize=40,color='g')
plt.xlabel("Date",fontsize=20,color='r')
plt.ylabel("Stock High Price",fontsize=30,color='r')
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
#Volume	Ex-Dividend	Split Ratio	Adj. Open	Adj. High	Adj. Low	Adj. Close	Adj. Volume
plt.plot(data['Volume'])
plt.title("Year wise volume",fontsize=40,color='g')
plt.xlabel("Year",fontsize=20,color='r')
plt.ylabel("Volume",fontsize=30,color='r')
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(data['Adj. Low'])
plt.title("Year wise Adj. Low",fontsize=40,color='g')
plt.xlabel("Year",fontsize=20,color='r')
plt.ylabel("Adj. Low",fontsize=30,color='r')
plt.show()


# In[ ]:


data['High'].plot("kde")


# In[ ]:


data['Adj. High'].plot("kde")


# In[ ]:


data['Ex-Dividend'].value_counts().plot("bar")


# In[ ]:


data['Split Ratio'].value_counts().plot("bar")


# In[ ]:


datax= data.drop(['Split Ratio','Ex-Dividend'],axis=1)
sns.heatmap(datax.corr(), fmt="f")


# In[ ]:


plt.figure(figsize=(12,8))

#gopen = data['Adj. Open']
#ghigh = data['Adj. High']
glow = data['Adj. Low']
#plt.plot(gopen)
#plt.plot(ghigh,color='r')
plt.plot(glow,color='g')
plt.plot(glow.rolling(window=2).mean(),color='r')
plt.plot(glow.rolling(window=2).std())


# In[ ]:


from pandas.tools.plotting import autocorrelation_plot
plt.figure(figsize=(12,8))
autocorrelation_plot(data)
plt.show()


# In[ ]:


model = ARIMA(data['Adj. Low'], order=(5,1,0))
model_fit = model.fit(disp=0)
model_fit.summary()


# In[ ]:


resd = pd.DataFrame(model_fit.resid)
resd.plot()


# In[ ]:


resd.plot(kind='kde')


# In[ ]:


X = data['Adj. High'].astype('float')
size = int(len(X)*0.66)

train,test = X[0:size],X[size:len(glow)]
print(len(train))
print(len(test))
for t in range(5):
    model = ARIMA(train,order=(5,1,0))
    fit = model.fit(disp=0)
    preds = fit.forecast()
    print("Predicted",preds[0],", Expected",test[t],"Month",test.index[t].date())

