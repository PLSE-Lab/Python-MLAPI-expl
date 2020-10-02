#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import ggplot as gp
from ggplot import *
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

url = "../input/wine_data.csv"
wine_df = pd.read_csv(url, header = None, names = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total_phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"])


# In[ ]:


wine_df.head()


# In[ ]:


wine_df.info()


# In[ ]:


wine_df.corr()


# In[ ]:


plt.subplots(figsize=(15,10))
sns.heatmap(wine_df.corr(), xticklabels=wine_df.columns.values, yticklabels=wine_df.columns.values, annot=True, cmap="YlGnBu")
plt.show()


# In[ ]:


wine_df['Alcohol (Scaled)'] = round(wine_df['Alcohol'],0).astype(str).str.zfill(4)
wine_df['Color (Scaled)'] = round(wine_df['Color intensity'],0).astype(str).str.zfill(4)
wine_df['Hue (Scaled)'] = round(wine_df['Hue'],0).astype(str).str.zfill(4)
wine_df['Ash (Scaled)'] = round(wine_df['Ash'],0).astype(str).str.zfill(4)


# In[ ]:


wine_df.head()


# In[ ]:


p = ggplot(wine_df, aes(x='Flavanoids',y='Total_phenols',color='Color (Scaled)',size='Hue (Scaled)',shape='Ash (Scaled)'))
p + geom_point(size=50) +     facet_wrap("Alcohol (Scaled)")


# In[ ]:


train, test = train_test_split(wine_df, test_size=0.2)


# In[ ]:


len(train)


# In[ ]:


len(test)


# In[ ]:


train_regression = smf.ols('Flavanoids ~ Total_phenols', data=train).fit()
print(train_regression.summary())


# In[ ]:


test_regression = smf.ols('Flavanoids ~ Total_phenols', data=test).fit()
print(test_regression.summary())


# In[ ]:


full_regression = smf.ols('Flavanoids ~ Total_phenols', data=wine_df).fit()
print(full_regression.summary())


# ### Discussion Topics

# * What issues came up with doing faceted plotting in ggplot?
#   
#   Some of the issues that came up w/ faceted plotting in ggplot, included not being able to leverage float values
#   for faceting, and the inability to sort string-converted integers effectively w/out padding the strings with leading
#   zeros.  These were by no means show-stoppers in leveraging the functionality, but required additional columns to be
#   added to the data frame I was leveraging, and some additional manipulation to allow for proper sorting.
# 
# 
# * Why does clustering fit well with faceted plotting?
#   
#   Clustering fits well with facted plotting, as the facet strategy essentially creates clusters automatically, w/out
#   having to leverage any advanced algorithmic approach.  It is a great EDA strategy, and allows for the ability to
#   pivot quickly.
#   
# 
# * What did you discover using a correlation heatmap on your dataset?
# 
#   What was interesting about leveraging a correlation heatmap, is that was not only aesthetically pleasing, but it
#   was much easier to identify correlations between variables that were subsequently used in additional visualizations.
# 
# 
# * What conclusions can be made from the experimentations on test and training Data?
# 
#   When using a linear regression model against the training data, the statistical measure of how close the data are to
#   the fitted regression line (Adj. R-Squared) was much higher when compared to the Adj. R-Squared of the test model.
#   The Adj. R-Squared value for the training model was actually closer to the Adj. R-Squared value returned when
#   running the model against the full data set, which helps validate the central limit theorem . . . as sample size
#   increases, the sum/average of a sufficiently large number of independent random variables of the sample means will
#   approach the mean of the population, or a normal distribution.  Given the smaller sample size for the test data,
#   this sample was not normally distribvuted, or an accurate representation of the population, but repeated samples
#   should yield a result that is very close to that of the full population, which is what we were able to demonstrate.
