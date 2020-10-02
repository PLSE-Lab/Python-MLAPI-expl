#!/usr/bin/env python
# coding: utf-8

# # General Linear Models vs. Linear Models
# One of the assumption for linear models is that the outputs are normally distributed (Gaussian), but in many cases they are not. For instance, number of coffees drank per day follows a Bernoulli distribution with lambda depends on the stress levels, sleep quality in the previous night, and whether the day is work day. In this case, simply using Linear Model will not work, it requires General Linear Model with a link function mapping the expected outputs to a normal distribution (E.g., ln of the bernoulli distribution gets back to normal => now we can use linear model)

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


# Generate uniform random stress level for 200 days
def stressLevels():
    return np.random.randint(1, 10, 200)
# Generate uniform random sleep qualities from previous nights
def sleptQualities():
    return np.random.randint(1, 10, 200)
# Generate randomly (50/50 chance) if that day is work day
def workDays():
    return np.array([1 if x >= 0.5 else 0 for x in np.random.random(200)])
# Generate lambda => expected bernoully parameber for an yes
def lambdaVal(stressLevel, sleepQuality, workDay):
    x = (stressLevel - sleepQuality - workDay * 2)
    return (1/(1 + np.exp(-x)))/2.0


# In[ ]:


# Generate simulated data
stresses = stressLevels()
sleeps = sleptQualities()
works = workDays()
lambdas = np.array([lambdaVal(stresses[i], sleeps[i], works[i]) for i in range(200)])


# In[ ]:


fig = pd.DataFrame(lambdas).hist()


# In[ ]:


# Generate number of coffees that you might drink on a day => maximum is 5
coffees = np.array([np.random.binomial(5, ld) for ld in lambdas])


# In[ ]:


fig = pd.DataFrame(coffees).hist()


# In[ ]:


# It is observable from the previous picture that number of coffees (outcomes) follow a bernoulli distribution => Linear regression will not work so we will try and see
from sklearn.linear_model import LinearRegression
df = pd.DataFrame()
df['stresses'] = stresses
df['sleeps'] = sleeps
df['works'] = works
X = df.values
y = coffees


# In[ ]:


model = LinearRegression()
model.fit(X, y)


# In[ ]:


r_sq = model.score(X, y)
print("coefficient of determination:", r_sq)


# In[ ]:


print('Intercept: ', model.intercept_)
print('slope: ', model.coef_)


# In[ ]:


predicts = model.predict(X)


# In[ ]:


fig = pd.DataFrame(predicts).hist()


# It is observable that the predictions are Gaussian while the actuals are Bernoulli

# # Convert the data into log (poisson)
# So in this case, it is suggested that there is a link (ln function) to convert the expected values of the outcome to a normal distribution, thus, can apply the linear model again.

# In[ ]:


import statsmodels.api as sm


# In[ ]:


binomial_model = sm.GLM(y, X, family=sm.families.Poisson())


# In[ ]:


binomial_results = binomial_model.fit()
print(binomial_results.summary())


# In[ ]:


print("Parameters; ", binomial_results.params)


# In[ ]:


yhat = binomial_results.mu


# In[ ]:


yhat


# In[ ]:


pd.DataFrame(yhat).hist()


# Now the result looks a lot better.

# In[ ]:




