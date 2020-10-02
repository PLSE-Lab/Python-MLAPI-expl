#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/Advertising.csv", usecols = [1,2,3,4]).copy()
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


df.isnull().sum()


# In[ ]:


df.corr()


# In[ ]:


sns.pairplot(df, kind="reg");


# In[ ]:


sns.jointplot(x="TV", y="sales", data = df, kind= "reg");


# In[ ]:


# Modelling With StatsModels


# In[ ]:


import statsmodels.api as sm


# In[ ]:


X = df[["TV"]]
X[:5]


# In[ ]:


X = sm.add_constant(X)
X[:5]


# In[ ]:


y = df["sales"]
y[:5]


# In[ ]:


lm = sm.OLS(y,X)
model = lm.fit()
model.summary()


# In[ ]:


import statsmodels.formula.api as smf
lm = smf.ols("sales ~ TV", df)
model = lm.fit()
model.summary()


# In[ ]:


model.params


# In[ ]:


model.conf_int()


# In[ ]:


print("F Prob. Value : ", "%.5f" % model.f_pvalue)


# In[ ]:


print("F Statistic : ", "%.5f" % model.fvalue)


# In[ ]:


print("T Value : ", "%.5f" % model.tvalues[:1])


# In[ ]:


print("Model MSE : ", "%.3f" %  model.mse_model, "\nSales Mean : ", np.mean(df.sales))


# In[ ]:


print("Model R Squared : ", "%.5f" % model.rsquared)


# In[ ]:


print("Adjusted R Squared : ", "%.5f" % model.rsquared_adj)


# In[ ]:


model.fittedvalues[:5]


# In[ ]:


print("Sales = " + str("%.2f" % model.params[0]) + " + TV" + "*" + str("%.2f" % model.params[1]))


# In[ ]:


g = sns.regplot(df["TV"], df["sales"], ci =None, scatter_kws = {"color": "r", "s":9})
g.set_title("Model Equation: Sales = 7.03 + TV*0.05")
g.set_ylabel("Number of sales")
g.set_xlabel("TV spending")
plt.xlim(-10,310)
plt.ylim(bottom = 0);


# In[ ]:


from sklearn.linear_model import LinearRegression
X = df[["TV"]]
y = df["sales"]
reg = LinearRegression()
model = reg.fit(X,y)


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


model.score(X,y)


# In[ ]:


#Prediction Phase


# In[ ]:


model.predict([[30]])


# In[ ]:


model.predict(X)[:10]


# In[ ]:


new_data = [[5],[90],[200]]
model.predict(new_data)


# In[ ]:


# Residuals 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


lm = smf.ols("sales ~ TV", df)
model = lm.fit()
model.summary()


# In[ ]:


mse = mean_squared_error(y, model.fittedvalues)
mse


# In[ ]:


rmse = np.sqrt(mse)
rmse


# In[ ]:


reg.predict(X)[:10]


# In[ ]:


y[:10]


# In[ ]:


comparison = pd.DataFrame({"real_y": y[:10],
                          "pred_y": reg.predict(X)[:10]})
comparison


# In[ ]:


comparison["error"] = comparison["real_y"] - comparison["pred_y"]
comparison


# In[ ]:


comparison["error_squared"] = comparison["error"]**2
comparison


# In[ ]:


np.sum(comparison["error_squared"])


# In[ ]:


np.mean(comparison["error_squared"])


# In[ ]:


np.sqrt(np.mean(comparison["error_squared"]))


# In[ ]:


model.resid[:10]


# In[ ]:


plt.plot(model.resid);

