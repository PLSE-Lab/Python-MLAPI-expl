#!/usr/bin/env python
# coding: utf-8

# ## Modeling

# Our idea is to run simultaneously many parallel linear regressions for every id (the 42k series) and use the resids to estimate quantiles. Let imagine that we have $N$ series of length $T$ in a vector $Y$ with shape $(N, T)$ and covariates of temporal features $Z$ with shape $(T, F)$. For id $i$ we will run the following regression:
# $$ Y_i^T = Z \beta_i^T + \epsilon_i $$
# 
# So if we compact all the individual OLS parameters $\beta_i$ in a vector $\beta$. The global model will be written as:
# $$ Y = \beta Z^T + \epsilon $$
# 
# **STEP 1: PARAMETER OPTIMIZATION**
# 
# Therefore we can solve the following problem including regularization:
# 
# $$ \hat{\beta}_{\lambda} = \arg\min \mid \mid Y - \beta Z^T \mid \mid_F^2 + \lambda \mid \mid \beta \mid \mid_F^2  $$
# 
# It is easy to see that:
# $$ \hat{\beta}_{\lambda} = YZ \left( Z^T Z + \lambda I_F \right)^{-1} $$
# 
# 
# **STEP 2: QUANTILE PREDICTION**
# 
# we have compute then the resid:
# $$  \hat{Y}^{\lambda} = \hat{\beta}_{\lambda} Z^T, \quad  \hat{\epsilon} = Y - \hat{Y}^{\lambda} $$
# 
# And we finally compute the quantile $q$ by:
# 
# $$\hat{Y}_{it}^q =\hat{Y}^{\lambda}_{it} + quantile(\hat{\epsilon}_{it}, q) $$
# 
# $\lambda$ can be learned by cross-validation.
# 
# Now let's code it out !!!
# 

# In[ ]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


class MultipleRegression():
    def __init__(self, reg=0.01):
        self.reg = reg
        self.BETA = None
    def fit(self, Y, Z):
        F = Z.shape[1]
        U = np.linalg.inv(Z.T.dot(Z)+self.reg*np.eye(F))
        self.BETA = Y.dot(Z).dot(U)
    def predict(self, Z):
        return  self.BETA.dot(Z.T)
#=======================


# In[ ]:


#======================================#
def preprocess_calendar(calendar):
    global maps, mods
    calendar["event_name"] = calendar["event_name_1"]
    calendar["event_type"] = calendar["event_type_1"]

    map1 = {mod:i for i,mod in enumerate(calendar['event_name'].unique())}
    calendar['event_name'] = calendar['event_name'].map(map1)
    map2 = {mod:i for i,mod in enumerate(calendar['event_type'].unique())}
    calendar['event_type'] = calendar['event_type'].map(map2)
    calendar['nday'] = calendar['date'].str[-2:].astype(int)
    maps["event_name"] = map1
    maps["event_type"] = map2
    mods["event_name"] = len(map1)
    mods["event_type"] = len(map2)
    calendar["wday"] -=1
    calendar["month"] -=1
    calendar["year"] -= 2011
    mods["month"] = 12
    mods["year"] = 6
    mods["wday"] = 7
    mods['snap_CA'] = 2
    mods['snap_TX'] = 2
    mods['snap_WI'] = 2

    calendar.drop(["event_name_1", "event_name_2", "event_type_1", "event_type_2", "date", "weekday"], 
                  axis=1, inplace=True)
    return calendar
#=========================================================


# In[ ]:


maps, mods = {}, {}
calendar = pd.read_csv("../input/m5-forecasting-uncertainty/calendar.csv")
calendar = preprocess_calendar(calendar)
calendar['nb'] = calendar.index + 1
calendar['n_week'] = (calendar.index - 2) // 7


# In[ ]:


calendar.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "FE = ['snap_CA','snap_TX','snap_WI']\nfor col in tqdm(['wday','month','year','event_name','nday']):\n    _temp = pd.get_dummies(calendar[col], prefix=col)\n    new_cols = list(_temp.columns)\n    FE += new_cols\n    calendar = calendar.join(_temp)\n#gc.collect()")


# In[ ]:


sales = pd.read_csv("../input/walmartadd/sales_aug.csv")


# In[ ]:


START = 1000
H = 28
B = 28
COLS = [f"d_{i}" for i in range(START, 1942)]
calendar = calendar[calendar.nb>=START]
ids = sales.id.values


# In[ ]:


sc = sales['scale1'].values


# In[ ]:


"""
POLY_COLS = ['t1']
calendar['t1'] = calendar.n_week / 300
MAX_DEG = 10
for deg in range(2, MAX_DEG+1):
    calendar[f't{deg}'] = calendar['t1']**deg
    POLY_COLS.append(f't{deg}')
"""
#


# In[ ]:


calendar.t1


# ## Data

# In[ ]:


X = sales[COLS].values
sw = sales['sales1'].values
sc = sales['scale1'].values
Z = calendar[FE].values
#Z = calendar[FE+POLY_COLS].values
Xs = X / sc[:, np.newaxis]
print(X.shape, Z.shape)


# In[ ]:





# ## Parameter Estimation

# In[ ]:


get_ipython().run_cell_magic('time', '', 'step1 = MultipleRegression(reg=0.001)\nstep1.fit(Xs, Z[:-28])')


# In[ ]:


Xp = step1.predict(Z)


# In[ ]:


INF = 0#40_000
k = np.random.randint(INF, 42840)
plt.plot(Xs[k, 800:])
plt.plot(Xp[k, 800:])
plt.title(ids[k])
plt.show()


# In[ ]:


Xs.shape, Xp.shape


# In[ ]:


print("mae:", mean_absolute_error(Xs[:, -28:], Xp[:,-56:-28]) )
print("mse:", mean_squared_error(Xs[:, -28:], Xp[:,-56:-28], squared=False))


# In[ ]:





# ## Quantile Prediction

# In[ ]:


resid = Xs - Xp[:, :-28]


# In[ ]:


qs = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
cp = np.quantile(resid, qs, axis=1).T


# In[ ]:


cp = cp[:,np.newaxis, :]


# In[ ]:


Zq = Xp[:,:,np.newaxis] + cp


# In[ ]:


Zq  = Zq.clip(0)


# In[ ]:


INF = 0#40_000
k = np.random.randint(INF, 42840)
plt.plot(Xs[k, -28:], label="groud truth")
plt.plot(Zq[k, -56:, 3], label="q25")
plt.plot(Zq[k, -56:, 4], label="q50")
plt.plot(Zq[k, -56:, 5], label="q75")
plt.title(ids[k])
plt.legend(loc="best")
plt.show()


# In[ ]:


pv = Zq[:,-56:-28, :]
pe = Zq[:,-28:, :]


# In[ ]:


pv.shape, pe.shape


# In[ ]:


sc= sc[:, np.newaxis]


# In[ ]:





# In[ ]:


names = [f"F{i+1}" for i in range(28)]
piv = pd.DataFrame(ids, columns=["id"])


# In[ ]:


QUANTILES = ["0.005", "0.025", "0.165", "0.250", "0.500", "0.750", "0.835", "0.975", "0.995"]
VALID = []
EVAL = []

for i, quantile in tqdm(enumerate(QUANTILES)):
    t1 = pd.DataFrame(pv[:,:, i]*sc, columns=names)
    t1 = piv.join(t1)
    t1["id"] = t1["id"] + f"_{quantile}_validation"
    t2 = pd.DataFrame(pe[:,:, i]*sc, columns=names)
    t2 = piv.join(t2)
    t2["id"] = t2["id"] + f"_{quantile}_evaluation"
    VALID.append(t1)
    EVAL.append(t2)
#============#


# In[ ]:


sub = pd.DataFrame()
sub = sub.append(VALID + EVAL)
del VALID, EVAL, t1, t2


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:




