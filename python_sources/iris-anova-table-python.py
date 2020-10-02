#!/usr/bin/env python
# coding: utf-8

# # Iris dataset ANOVA table
# Here is a brief example of an ANOVA table in python using the popular Iris dataset. Note, we usually try to predict the plant's species based off the data but in order to get an ordinary least squares model to train we are predicting the plant's sepal length instead. Also we wanted to train an OLS model to easily create an ANOVA table. We also show a Type 2 ANOVA table, [there are 3 types I, II, and III](https://stats.stackexchange.com/questions/60362/choice-between-type-i-type-ii-or-type-iii-anova).
# 
# ![Iris plant anatomy](https://proxy.duckduckgo.com/iu/?u=http%3A%2F%2Fblog.kaggle.com%2Fwp-content%2Fuploads%2F2015%2F04%2Firis_petal_sepal.png&f=1)

# In[ ]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
df = pd.read_csv('../input/Iris.csv')
df.head()


# In[ ]:


df.describe()


# ## ANOVA with one categorical feature

# In[ ]:


iris_lm_one_categorical=ols('SepalLengthCm ~ C(Species)', data=df).fit() #Specify C for Categorical
sm.stats.anova_lm(iris_lm_one_categorical, typ=2)


# The Probability of the F statistic is statistically significant (PR(>F) < 0.05). Meaning the information in `Species` has predictive power for `SepalLengthCm`. You should follow up with t-tests for every combination of species to detect differences.

# ## ANOVA with categorical and numerical features (all features)
# Now let's show an ANOVA with several features.

# In[ ]:


iris_lm=ols('SepalLengthCm ~ C(Species) + SepalWidthCm + PetalLengthCm + PetalWidthCm', data=df).fit() #Specify C for Categorical
sm.stats.anova_lm(iris_lm, typ=2)


# In[ ]:


iris_lm.summary()


# The `.summary()` is useful when you do ANOVA with a model with more than one feature because the statsmodel `.anova_lm()` does not print the total F-statistic for the model by default. By default it shows the F-statistic contributed by each feature. Here we see the F-statistic is 188 for the whole model with a Prob (F-statistic) < 0.05, so we reject the null hypothesis.
# 
# For more information see [Interactions and ANOVA](https://devdocs.io/statsmodels/examples/notebooks/generated/interactions_anova) statsmodels docs.
