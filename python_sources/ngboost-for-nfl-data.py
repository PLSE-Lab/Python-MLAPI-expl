#!/usr/bin/env python
# coding: utf-8

# Stanford ML group have release a package call NGBoost (Natural Gradient Boosting) which aims to provide a probablistic estimation on top of a gradient boosting model. Since the competition don't allow internet connection nor external data it's unlikly to be helpful to competition score so it shall only be a demostration.

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().system('pip install ngboost')


# Motivation: in NFL big data bowl 2020 the task is to predict rushing yards gain by offensive formation and result would be judged by Continuous Ranked Probability Score (CRPS). Yards gain data is not distributed by a normal distribution: in fact it is right skewed since the chance of getting positive yardage is much higher than getting negative yards. Therefore we want to try to fit a lognormal distribution which is also righted skewed to see the goodness of fit.
# 
# NGBoost package both natively support CRPS and lognormal distribution which is a perfect candidate to test its performance

# In[ ]:


data = pd.read_csv('../../kaggle/input/nfl-big-data-bowl-2020/train.csv')

data['YardLine100'] = data['YardLine'] #Normalize yardline to 1-99
data.loc[data['FieldPosition'] == data['PossessionTeam'], 'YardLine100'] = 50+  (50-data['YardLine'])
data['Touchdown'] =  data.Yards == data.YardLine100

temp_data = data[data.Touchdown == 0][['YardLine100','Down','Distance','DefendersInTheBox','Yards']].dropna().drop_duplicates()
X = np.array(temp_data[['YardLine100','Down','Distance','DefendersInTheBox']])
y = np.array(temp_data['Yards'])*1.0


# In[ ]:


from scipy.stats import lognorm
shape, loc, scale = lognorm.fit(y)

fig, ax = plt.subplots()
x_axis = np.linspace(-10,50,100)
ax.hist(y,bins=np.arange(-10, y.max() + 1.5) -0.5,density=True)
ax.plot(x_axis,lognorm.pdf(x_axis, shape, loc, scale))
plt.title('Histogram of rushing yards gain excluding touchdown')
plt.xlabel('rushing yard gain')


# Here is the histogram of rushing yards and the fit of lognormal distribution. Here we can see the fit is just soso since football data have a sharp peak of 3-4 yards gain and it's difficult to fit by a smooth distribution function. Next we will use NGBoost to see how it works

# In[ ]:


from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.scores import CRPS, MLE
from ngboost.distns import LogNormal, Normal


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In lognormal distribution the data has to be distributed above zero so a transform is needed to fit the data

# In[ ]:


ngb =  NGBoost(n_estimators=100, learning_rate=0.1,
              Dist=LogNormal,
              Base=default_tree_learner,
              natural_gradient=False,
              minibatch_frac=1.0,
              Score=CRPS())
ngb.fit(X_train,y_train-min(y_train)+0.001)


# In[ ]:


y_preds = ngb.predict(X_test)
y_dists = ngb.pred_dist(X_test)


# In[ ]:


print('mean of lognormal scale = %f'% y_dists.scale.mean())
print('standard deviation of lognormal scale = %f'%y_dists.scale.std())


# In[ ]:


from ngboost.evaluation import *
pctles, observed, slope, intercept = calibration_regression(y_dists, y_test-min(y_test)+0.001)
plt.subplot(1, 2, 1)
plot_pit_histogram(pctles, observed, label="CRPS", linestyle = "-")


# In an ideal fit we would expect the bars are close to the dotted line, indicate that it's a somewhat bad fit which is the same case as above.
# 
# To sum up even this competition require using cummulative distribution as competition score, the data is hard to fit by one single distribution so a combination of distribution like Gaussian Mixture model is required.
