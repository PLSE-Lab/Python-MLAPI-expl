#!/usr/bin/env python
# coding: utf-8

# **Market Mix Modeling:**
# 
# Market Mix Modeling is a technique which helps to quantify the impact of several marketing inputs on sales or revenue. The primary motive of this model is to understand how each marketing input contributes to sales (dependent variable). 
# 
# The key indicators (independent variables) are under these categories:
# 
# **Price**
# 
# **Distribution**
# 
# **Seasonality**
# 
# **Macro-economic variables**
# 
# **Advertising**
# 
# **Sales promotion**
# 
# **Public relations**
# 
# The model uses statistical concepts, inferences and Regression techniques to understand the dependent variable with the help of it's indicators. In this notebook I have also addressed the effect of interaction among the independent variables.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data = pd.read_excel('/kaggle/input/market-mix/market.xlsx')


# In[ ]:


data.head()


# In[ ]:


data.columns


# As we can see here there is a column called 'BrandName'. Let's see the different brands present.

# In[ ]:


data['BrandName'].unique()


# In[ ]:


len(data['BrandName'].unique())


# There are 27 distinct types of vodka brands in it.

# In[ ]:


data.groupby(['BrandName']).size().reset_index(name='counts')


# Let's work one just one brand for now to get some statistical inference.

# **statsmodels:**
# 
# The Python module used is 'statsmodels', which provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration.

# In[ ]:


# Selected a single brand to work on
Absolut_Vod = data[data['BrandName'] == 'Absolut']
Absolut_Vod.head()


# We can say that price is an important indicator (for consumer goods) to understand sales.

# In[ ]:


Price_Absolut = Absolut_Vod[['LnSales','LnPrice']]


# In[ ]:


plt.scatter(Price_Absolut['LnPrice'],Price_Absolut['LnSales'])
plt.title('Normalized price vs sales')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.show()


# In[ ]:


# Regression Plot using seaborn
import seaborn as sns; sns.set(color_codes=True)
plot = sns.regplot(x = Price_Absolut['LnPrice'],y = Price_Absolut['LnSales'], data=Price_Absolut)


# In[ ]:


import statsmodels.formula.api as sm
from statsmodels.compat import lzip
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.regressionplots


# Regression model for LnSales (Y - dependent variable) vs LnPrice (X - independent variable)

# In[ ]:


reg_result = sm.ols(formula = 'LnSales ~ LnPrice',data = Price_Absolut).fit()


# In[ ]:


reg_result.summary()


# **p-value:**
# 
# The p-value of the price is zero (less than the test statistic), indicates that the price is significant indicator of sales.
# 
# **R-squared:**
# 
# An R-squared of 1 or 100% means that all movements of a dependent variable is completely explained by movements of the independent variable we are interested in
# 
# In this case, the value of R-squared is 0.688 i.e the price variable indicates nearly 69% of the sales data points.
# 
# **co-efficient of price (LnPrice - coef):**
# 
# The co-efficient of price means that, every unit increase in price, there is 1.13 times increase in sales.
# 

# In[ ]:


name = sm.ols(formula = 'LnSales ~ LnPrice', data = Price_Absolut)
name.endog_names #dependent variable


# In[ ]:


name.exog_names #intercept and predictor


# In[ ]:


r = name.fit()


# In[ ]:


r.params


# In[ ]:


name.loglike(r.params)

# We can see the log likelihood ratio from the above summary stats and here


# In[ ]:


name.predict(r.params, [[1, 4.7]])

# In terms of linear regression, y = mx + c, c = 2.836674 and m = 1.130972, 
# we are passing two values x = 4.7 and 
# the the value 1 is passed as multiplier for c (so, c remains at 2.836674 as per our model)


# In[ ]:


# Statsmodels Regression Plots
fig = plt.figure(figsize=(15,8))
fig = statsmodels.graphics.regressionplots.plot_regress_exog(reg_result, "LnPrice", fig=fig)


# In[ ]:


#Let's add more indicators to the regression and to monitor the R-squared value, 
# our aim is to increase R-squared (or to determine the optimum level)
Additional_Absolut = Absolut_Vod[['LnSales','LnMag','LnNews','LnOut','LnBroad','LnPrint','LnPrice']]


# In[ ]:


result_2 = sm.ols('LnSales ~ LnMag + LnNews + LnOut + LnBroad + LnPrint + LnPrice',data=Additional_Absolut).fit()


# In[ ]:


result_2.summary()


# In[ ]:


# Statsmodels Multivariate Regression Plots
fig = plt.figure(figsize=(15,8))
fig = statsmodels.graphics.regressionplots.plot_partregress_grid(result_2, fig=fig)


# Since the number of indeicators we have used are more, the Adj. R-squared value is at 0.86+. It is able to explain 87% of the data points. 
# 
# But here the p-values of some variables are high which can be accounted due to interaction effect and some other factors.
# 
# Let's try out the interation effect method between variables LnBroad and LnPrint

# In[ ]:


interaction = sm.ols('LnSales ~ LnMag + LnNews + LnOut + LnBroad * LnPrint + LnPrice',data=Additional_Absolut).fit()


# In[ ]:


interaction.summary()


# The presence of the interaction indicates that the effect of one predictor variable on the response variable is different at different values of the other predictor variable.
# 
# The R-squared value has increased.

# In[ ]:


# Plots
fig = plt.figure(figsize=(15,8))
fig = statsmodels.graphics.regressionplots.plot_partregress_grid(interaction, fig=fig)


# We have noticed in all the summary stats, there is a problem of strong multicollinearity, I will update the notebook soon, addrressing this issue.
