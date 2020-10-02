#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In this notebook, we will do logistic regression to predict `Survived` using `Age` variable. For simplicity, I'll skip EAD part (which has been nicely done in many other popular kernels). I'll use [NumPyro](https://github.com/pyro-ppl/numpyro) for modelling, sampling, and making predictions.

# In[ ]:


get_ipython().system('pip install numpyro')


# In[ ]:


from jax import ops, random
from jax.scipy.special import expit

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive


# ### prepare data

# After loading data, I recognize that there are many missing values for `Age` column. We know (by intuition or from other kernels) that `Age` is correlated with the title of the name: e.g. those with `Mrs.` would be older than those with `Miss.`. Let's make a new column `Title` for that purpose.

# In[ ]:


train_df = pd.read_csv("../input/titanic/train.csv")
d = train_df.copy()
d.Embarked.fillna("S", inplace=True)  # filling 2 missing data points with the mode "S"
d["Title"] = d.Name.str.split(", ").str.get(1).str.split(" ").str.get(0).apply(
    lambda x: x if x in ["Mr.", "Miss.", "Mrs.", "Master."] else "Misc.")
title_cat = pd.CategoricalDtype(categories=["Mr.", "Miss.", "Mrs.", "Master.", "Misc."], ordered=True)
age_mean, age_std = d.Age.mean(), d.Age.std()
embarked_cat = pd.CategoricalDtype(categories=["S", "C", "Q"], ordered=True)
data = dict(age=d.Age.pipe(lambda x: (x - age_mean) / age_std).values,
            pclass=d.Pclass.values - 1,
            title=d.Title.astype(title_cat).cat.codes.values,
            sex=(d.Sex == "male").astype(int).values,
            sibsp=d.SibSp.clip(0, 1).values,
            parch=d.Parch.clip(0, 2).values,
            embarked=d.Embarked.astype(embarked_cat).cat.codes.values,
            survived=d.Survived.values)


# Note that I don't use other features such as `Fare` or `Cabin` for simplicity. I also don't do much of feature engineering for the same reason.

# ### modelling

# If you are not familiar with NumPyro, you can take a look at [its documentation](https://github.com/pyro-ppl/numpyro#numpyro) which includes some tutorials, examples, and translated code for Statistical Rethinking book (which is a good reference IMO if you are not familiar with Bayesian methods).

# In[ ]:


def model(age, pclass, title, sex, sibsp, parch, embarked, survived=None):
    # create a variable for each of Pclass, Title, Sex, SibSp, Parch,
    b_pclass = numpyro.sample("b_Pclass", dist.Normal(0, 1), sample_shape=(3,))
    b_title = numpyro.sample("b_Title", dist.Normal(0, 1), sample_shape=(5,))
    b_sex = numpyro.sample("b_Sex", dist.Normal(0, 1), sample_shape=(2,))
    b_sibsp = numpyro.sample("b_SibSp", dist.Normal(0, 1), sample_shape=(2,))
    b_parch = numpyro.sample("b_Parch", dist.Normal(0, 1), sample_shape=(3,))
    b_embarked = numpyro.sample("b_Embarked", dist.Normal(0, 1), sample_shape=(3,))

    # impute Age by Title
    age_mu = numpyro.sample("age_mu", dist.Normal(0, 1), sample_shape=(5,))
    age_mu = age_mu[title]
    age_sigma = numpyro.sample("age_sigma", dist.Normal(0, 1), sample_shape=(5,))
    age_sigma = age_sigma[title]
    age_isnan = np.isnan(age)
    age_nanidx = np.nonzero(age_isnan)[0]
    if survived is not None:
        age_impute = numpyro.param("age_impute", np.zeros(age_isnan.sum()))
    else:  # for prediction, we sample `age_impute` from Normal(age_mu, age_sigma)
        age_impute = numpyro.sample("age_impute", dist.Normal(age_mu[age_nanidx], age_sigma[age_nanidx]))
    age = ops.index_update(age, age_nanidx, age_impute)
    numpyro.sample("age", dist.Normal(age_mu, age_sigma), obs=age)

    a = numpyro.sample("a", dist.Normal(0, 1))
    b_age = numpyro.sample("b_Age", dist.Normal(0, 1))
    logits = a + b_age * age

    logits = logits + b_title[title] + b_pclass[pclass] + b_sex[sex]         + b_sibsp[sibsp] + b_parch[parch] + b_embarked[embarked]
    # for prediction, we will convert `logits` to `probs` and record that result
    if survived is None:
        probs = expit(logits)
        numpyro.sample("probs", dist.Delta(probs))
    numpyro.sample("survived", dist.Bernoulli(logits=logits), obs=survived)


# ### sampling

# After making a model, sampling is pretty fast in NumPyro.

# In[ ]:


mcmc = MCMC(NUTS(model), 1000, 1000)
mcmc.run(random.PRNGKey(0), **data)
mcmc.print_summary()


# As we can see, using Bayesian, we can get uncertainties of our results: e.g. imputing values, coefficients of being male or female,... (and you can make nice plots with them ;)

# ### make predictions

# To make predictions on the new data, we will maginalize those `age_imput` variables (in other words, removing them from posterior samples) and use the remaining variables for predictions.

# In[ ]:


test_df = pd.read_csv("../input/titanic/test.csv")
d = test_df.copy()
d["Title"] = d.Name.str.split(", ").str.get(1).str.split(" ").str.get(0).apply(
    lambda x: x if x in ["Mr.", "Miss.", "Mrs.", "Master."] else "Misc.")
test_data = dict(age=d.Age.pipe(lambda x: (x - age_mean) / age_std).values,
                 pclass=d.Pclass.values - 1,
                 title=d.Title.astype(title_cat).cat.codes.values,
                 sex=(d.Sex == "male").astype(int).values,
                 sibsp=d.SibSp.clip(0, 1).values,
                 parch=d.Parch.clip(0, 2).values,
                 embarked=d.Embarked.astype(embarked_cat).cat.codes.values)

posterior = mcmc.get_samples().copy()
posterior.pop("age_impute")
survived_probs = Predictive(model, posterior).get_samples(random.PRNGKey(2), **test_data)["probs"]
d["Survived"] = (survived_probs.mean(axis=0) >= 0.5).astype(np.uint8)
d[["PassengerId", "Survived"]].to_csv("submission.csv", index=False)


# Submiting the result gives me the score about 79 (top 16%). It is great for the first attempt. :)

# ### further improvements
# 
# + Using other features such as `Cabin` or `Fare`.
# + The above model assumes a linear relationship of `Survived` w.r.t. other latent variables. The result is intuitive but is not enough to beat tree-based models. We can build more complicated models or construct a [Bayesian neural network](http://pyro.ai/numpyro/bnn.html) model to capture more complex relationship.
