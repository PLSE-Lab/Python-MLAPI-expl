#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn


# In[ ]:


print(np.__version__)
print(pd.__version__)
import sys
print(sys.version)
print(sklearn.__version__)


# In[ ]:


from sklearn.datasets import load_boston


# In[ ]:


boston_data = load_boston()


# In[ ]:


df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


X = df


# In[ ]:


y = boston_data.target


# In[ ]:


y


# In[ ]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[ ]:


X_constant = sm.add_constant(X)


# In[ ]:


pd.DataFrame(X_constant)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'sm.OLS')


# In[ ]:


model = sm.OLS(y, X_constant)


# In[ ]:


lr = model.fit()


# In[ ]:


lr.summary()


# In[ ]:


form_lr = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', 
              data=df)
mlr = form_lr.fit()


# In[ ]:


mlr.summary()


# In[ ]:


form_lr = smf.ols(formula = 'y ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT', 
              data=df)
mlr = form_lr.fit()
mlr.summary()


# In[ ]:


pd.options.display.float_format = '{:,.2f}'.format
corr_matrix = df.corr()
corr_matrix


# In[ ]:


corr_matrix[np.abs(corr_matrix) < 0.6] = 0
corr_matrix


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.show()


# In[ ]:


eigenvalues, eigenvectors = np.linalg.eig(df.corr())


# In[ ]:


pd.options.display.float_format = '{:,.4f}'.format
pd.Series(eigenvalues).sort_values()


# In[ ]:


np.abs(pd.Series(eigenvectors[:,8])).sort_values(ascending=False)


# In[ ]:


print(df.columns[2], df.columns[8], df.columns[9])


# In[ ]:


df.head()


# In[ ]:


plt.hist(df['TAX']);


# In[ ]:


plt.hist(df['NOX']);


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', 
              data=df)
benchmark = linear_reg.fit()
r2_score(y, benchmark.predict(df))


# In[ ]:


linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B', 
              data=df)
lr_without_LSTAT = linear_reg.fit()
r2_score(y, lr_without_LSTAT.predict(df))


# In[ ]:


linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT', 
              data=df)
lr_without_AGE = linear_reg.fit()
r2_score(y, lr_without_AGE.predict(df))

