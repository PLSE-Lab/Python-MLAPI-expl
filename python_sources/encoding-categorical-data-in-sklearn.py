#!/usr/bin/env python
# coding: utf-8

# # Encoding categorical data in sklearn
# 
# `scikit-learn`, `pandas`, and `numpy` all come with various utilities for transforming data into categorical types. I personally am particularly fond of `pandas.get_dummies`. This notebook explores `category_encoders`, a `scikit-contrib` module specifically dedicated to categorical encoders, which is the recommended way of applying more complicated categorical transforms on your data. It also comes with a host of features useful for advanced use cases. This is that notebook.
# 
# Naturally one might ask: what is the point of categorical encoding?
# 
# When storing data, categorical encoding can be useful as a technique for reducing dataset size on disk (via dictionary encoding). However, the primary usefulness of categorical encoding, and the reason that it is such a well-developed set of techniques within classical statistics, is that it can be used to improve the explanatory power of statistical models. This is done by transforming input data into an output representation that is more ammenable to modeling.
# 
# Encoders have many of the same advantages and disadvantages as the techniques of [dimensionality reduction](https://www.kaggle.com/residentmario/dimensionality-reduction-and-pca-for-fashion-mnist/) overall. They make data easier to model, but also potentially lose information which can be taken advantage of by more complex models, like neural networks. For this reason they are not used very often outside of regression settings.

# In[ ]:


import numpy as np
import pandas as pd
import pandas as pd
from sklearn import preprocessing
import category_encoders as ce


def get_cars_data():
    """
    Load the cars dataset, split it into X and y, and then call the label encoder to get an integer y column.
    :return:
    """
    df = pd.read_csv(
        'https://raw.githubusercontent.com/scikit-learn-contrib/categorical-encoding/master/examples/source_data/cars/car.data.txt'
    )
    X = df.reindex(columns=[x for x in df.columns.values if x != 'class'])
    y = df.reindex(columns=['class'])
    y = preprocessing.LabelEncoder().fit_transform(y.values.reshape(-1, ))

    mapping = [
        {'col': 'buying', 'mapping': [('vhigh', 0), ('high', 1), ('med', 2), ('low', 3)]},
        {'col': 'maint', 'mapping': [('vhigh', 0), ('high', 1), ('med', 2), ('low', 3)]},
        {'col': 'doors', 'mapping': [('2', 0), ('3', 1), ('4', 2), ('5more', 3)]},
        {'col': 'persons', 'mapping': [('2', 0), ('4', 1), ('more', 2)]},
        {'col': 'lug_boot', 'mapping': [('small', 0), ('med', 1), ('big', 2)]},
        {'col': 'safety', 'mapping': [('high', 0), ('med', 1), ('low', 2)]},
    ]

    return X, y, mapping

X, y, mapping = get_cars_data()


# ### One-hot encoding
# 
# One-hot encoding is the simplest type of categorical encoding. It takes each unique class in a list of class observations, and turns that into a `True`/`False` matrix where each unique class has its own column and its own binary value. This is the most common encoding used for e.g. preprocessing boolean data as input to a `keras` neural network.

# In[ ]:


bus_or_taxi = pd.DataFrame(
    np.where(np.random.random((1, 1000)) > 0.5, 'bus', 'taxi').T,
    columns=['mode']
)


# In[ ]:


encoder = ce.OneHotEncoder()
encoder.fit_transform(bus_or_taxi).head()


# In[ ]:


encoder.category_mapping


# If you want to carry over the category name, use `use_cat_names` parameter. Otherwise the category index mapping will be used; this is what was done above!

# In[ ]:


encoder = ce.OneHotEncoder(use_cat_names=True)
encoder.fit_transform(bus_or_taxi).head()


# ## Ordinal
# 
# Ordinal encoding transforms a column of class names into a column of class integers. Ordinal encoding scheme maps columns to integers randomly, unless you give it a mapping yourself, in which case it will use your mapping instead. This encoding is called ordinal because the output values are ordinal (e.g. sortable categorical), even if the input values are not.

# In[ ]:


encoder = ce.OrdinalEncoder()
encoder.fit_transform(bus_or_taxi).head()


# ### Digression into statistics
# 
# The two transformers we have seen so far have been very simple, and have the property that they are 1:1, e.g. every single input category is represented by a single output category. The rest of the transformers available in the library are more sophisticated, and may choose or choose not to merge particular categories together based on statistical tests that are applied to the dataset at fitting time.
# 
# The idea here is that by choosing a categorical encoding that is appropriate for the given dataset and what we want to do with it, we may have the benefits of categorization (e.g. dictionary-mapped encoding) whilst still preserving those properties of the dataset which are analytically valuable.
# 
# A bit of a digression into statistics...
# 
# A **contrast** is a linear combination of variables whose coefficients add up to zero. In other words, the quantity $\sum_{i=1}^t a_i\theta_i$ is a contrast iff $\sum_{i=1}^t a_i = 0$. An example of a case in which this property holds is the difference of two weighted means. A constrast is furthermore **orthogonal** if it has the additional property that $\sum_{i=1}^t a_i b_I = 0$.
# 
# Each target or source category is known as a **level**.
# 
# Many of the more advanced encoding schemes available in this library are what are known as contrast coders. They are formulated using a well-defined contrast matrix, and may or may not also be orthogonal.
# 
# ### Polynomial
# 
# `PolynomialEncoder` takes categorical values are input:
# 
# ```
# (28,40] (40,52] (52,64] (64,76]
#      22      93      55      30
# ```
# 
# And outputs adjusted categorical values as output:
# 
# ```
#  (28,40]  (40,52]  (52,64]  (64,76]
# 42.77273 49.97849 56.56364 61.83333
# ```
# 
# These values are chosen by fitting a polynomial regressor to the data, solving for the linear/quadratic/cubic coefficients that minimizes the adjusted $r^2$ (or some other test statistic) of the regression, and replacing the original categorical values with the new coefficient values.
# 
# Some explanations of words:
# 
# * **Polynomial regression** &mdash; an extension to linear regression which adds higher-degree terms (e.g. $x^2$, $x_3$, etc.) and their corresponding weights to the equation.
# * **Adjusted r^2** &mdash; $r^2$ is a goodness of fit test for linear models that measures the proportion of explained variance of a model. If we are trialing many different models, $r^2$ is misleading due to the [file drawer effect](https://en.wikipedia.org/wiki/Publication_bias). Adjusted $r^2$ adds the adjustment to $r^2$ that makes it so that the $r^2$ of a new model is only an improvement to that of a previous model if the amount of explanatory power added exceeds the amount of explanatory power that could reasonably be added by random chance.
# 
# For algorithmic purposes, the coefficients chosen are subject to the constraint that they form an orthogonal contrast. In this example case this looks thusly:
# 
# ```
# Level of readcat	Linear (readcat.L)	Quadratic (readcat.Q)	Cubic (readcat.C)
# 1 (28,40]	 -.671	.5	-.224
# 2 (40,52]	 -.224	-.5	.671
# 3 (52,64]	 .224	-.5	-.671
# 4 (64,76]	 .671	.5	.224
# ```
# 
# Polynomial regression is a good encoding scheme if you wish to apply a parametric model (particularly a regression model) to the data afterwards. However, it can only be used for ordinal categorical variables.

# In[ ]:


X.head()


# In[ ]:


import category_encoders as ce
X_trans = ce.PolynomialEncoder().fit_transform(X, y)
X_trans.head()


# ### Helmert
# 
# Helmert coding is a contrast coding scheme that encodes every category as the difference between the mean value of that category and the mean of the mean values of the groups that follow. For example, suppose that there are four categorical values $\mu_n$. After applying Helmert encoding, the new $\mu_1$ level would be a function of its own mean and the mean of the mean observed values of $\mu_2 \ldots \mu_4$. The new $\mu_2$ level would be a function of its own mean and the mean of the mean observed values of $\mu_3$ and $\mu_4$. And so on.
# 
# Helmert coding is a good fit for scenarios where there are significant "stacked" differences between groups in the data.
# 
# See [here](https://stats.stackexchange.com/questions/411134/how-to-calculate-helmert-coding) for an extended mathematical explanation.

# In[ ]:


import category_encoders as ce
X_trans = ce.HelmertEncoder().fit_transform(X, y)
X_trans.head()


# ### Binary
# 
# Binary encoding is similar to one-hot, except it outputs to a bytestring.

# In[ ]:


import category_encoders as ce
X_trans = ce.BinaryEncoder().fit_transform(X)
X_trans.head()


# ### Hashing encoder
# 
# The above are all examples of dictionary encoders, e.g. encoders that rely on a dictionary mapping from the original value to the new value. A hashing encoder maps values using a one-way hashing function instead. This means that this encoder doesn't save on space as much when the arity is low, like the others do, but also means that there is no dictionary the opposite is true when the arity is high. 

# In[ ]:


import category_encoders as ce
X_trans = ce.HashingEncoder().fit_transform(X)
X_trans.head()


# ### BaseN
# 
# `BaseN` encodes the column in base-N. When arity > `n`, more than one column will be generated, with each column representing one digit of the output representation. When arity < `n`, there will be just one column, so the representation is equivalent to ordinal encoding. When `n = 1` this encoding is equivalent to one-hot encoding.
# 
# An important point about base-N is that when a column gets broken up into multiple columns, the information that was previously encoded in a single column is now "broken" up across two (or more). This is a form of induced non-linearity, and many models cannot take advantage of information split across a column like this. Decision trees are a good example of a type of model that does not care, and may even benefit from this split, because it uses splitting rules. Support vector machines are also very good at overcoming this shortcoming, I suspect. But old-school regression models will suffer.

# In[ ]:


import category_encoders as ce
X_trans = ce.BaseNEncoder(base=2).fit_transform(X)
X_trans.head()
# up to three columns per class


# In[ ]:


import category_encoders as ce
X_trans = ce.BaseNEncoder(base=4).fit_transform(X)
X_trans.head()
# up to two columns per class


# In[ ]:


import category_encoders as ce
X_trans = ce.BackwardDifferenceEncoder().fit_transform(X)
X_trans.head()


# ### Backwards difference
# 
# Backwards difference is another contrast coding scheme. In backwards difference coding, the mean of the dependent variable (`y`) for a level is compared with the mean of the dependent variables (`X`) for the immediately previous level.
# 
# Backwards difference coding can be contrasted with Helmert coding. Helmert coding constructs new values using the mean of the remaining values (there is also backwards Helmert coding, which does the same with the means of the *previous* values), whilst backwards difference does so with just one class at a time.
# 
# Backwards difference coding is therefore an attractive option for the encoding of data that has pairwise relationships between adjacent values, because this pairwise information will be preserved in the choice of the output value.

# In[ ]:


import category_encoders as ce
X_trans = ce.TargetEncoder().fit_transform(X, y)
X_trans.head()


# ### Target
# 
# Target encoding is a supervised categorical encoding technique (e.g. unlike certain other encoders, it uses information from your target variable `y`).
# 
# In the case that the target variable is categorical, features are replaced with a blend of posterior probability of the target given particular categorical value and the prior probability of the target over all the training data. The first element, **Posterior probability** is the probability of output observation $y$ given input data $X$, e.g. $p(y|X)$. The second element, prior probability of the target over all the training data, is the probability of $y$ occurring at all: $p(y)$.
# 
# Why do we blend these two variables? In small datasets, there are very few observations of $y|X$, so our estimate of $P(y|X)$ is unreliable. For example, consider the following dataset:
# 
# ```
# X   y
# 20  True
# 10  False
# 10  False
# 20  False
# ```
# 
# In this dataset there are *no* observations of `y=True|X=10`, so $P(y|X)$ is completely undefined! It is easy to imagine another, slightly larger dataset where there are such observations, but very few of them, which makes our estimate of this quantity unreliable. In either case, a "safer" estimate for $P(y|X)$ is $P(y)$. If on the other hand we have a lot of observations like these, we have a lot of condifence that our estimate is reliable, so this isn't appropriate to do. Target encoding uses a weighted mean of these two values internally, emphasizing the second value when there is little support and the first when there is a lot of it.
# 
# If the target column is continuous (or more accurately, near-continuous) instead, posterior and base probability is substitued for expected value. The same principles and rationale applies.
# 
# Target encoding is generally considered to be a pretty good encoding scheme because the values it maps to are weighted averages of the target value. Mapping categorical values close to their expected value in this way is helpful in both probabilistic algorithms, like naive Bayes, as well as in regression. For this reason it is probably the second most common encoding scheme of the ones in this post, after one-hot.
# 
# A good reference on target encoding is [this blog post](https://maxhalford.github.io/blog/target-encoding-done-the-right-way/).

# In[ ]:


import category_encoders as ce
X_trans = ce.TargetEncoder().fit_transform(X, y)
X_trans.head()


# ## Leave-one-out
# 
# Leave-one-out is a very slight modification on target encoding. It makes the change that the current row's observation is excluded from consideration when calculating $P(y|X)$ and/or $P(y)$. This very slightly reduces the amount of information that the target encoder has to work with, but also increases the robustness of coding scheme against outliers.
# 
# This is a similar technique at least in name to leave-one-out cross validation, which is a similar modification on k-fold cross validation.

# In[ ]:


import category_encoders as ce
X_trans = ce.LeaveOneOutEncoder().fit_transform(X, y)
X_trans.head()


# ## CatBoost
# 
# CatBoost is an categorical encoder taken from the CatBoost gradient boosted machine library which performs leave-one-out encoding "on the fly". I'm not entirely sure what is intended by "on the fly", but I suspect it means that the value assigned to each observation only uses the set of observations observed so far. This would cause some noise in the earlier parts of the dataset, but also makes the algorithm streaming, as you don't have to load (or even be able to fit) the entirety of the dataset into memory. Of course in the case of `category_encoders`, you're using a `DataFrame`, so it pretty much has to be in memory anyway...
# 
# Because this algorithm is streaming, it's important that the data be randomly permuted, e.g. that there is no information about the target variable in the sort order of the `DataFrame`.

# In[ ]:


import category_encoders as ce
X_trans = ce.CatBoostEncoder().fit_transform(X, y)
X_trans.head()


# ### M-estimate
# 
# M-estimate is a simplified version of a target encoder with one tunable parameter (`m`) instead of two (`min_samples_leaf` and `smoothing`). It has certain statistical implications that I didn't really dig into.

# In[ ]:


import category_encoders as ce
X_trans = ce.MEstimateEncoder().fit_transform(X, y)
X_trans.head()


# ### Sum
# 
# Sum coding is reversed backwards difference contrast coding. Instead of using the difference of factor levels, it uses their sums. I have little information beyond that.

# In[ ]:


import category_encoders as ce
X_trans = ce.SumEncoder().fit_transform(X, y)
X_trans.head()


# ### Weight of Evidence
# 
# **Weight of evidence** is a statistical measure of the level of (binary) seperation between two groups under certain conditions. It evolved out of applications of logistic regression to credit default risk. Given that $E$ is a boolean event, e.g. something that happened, and $\neg E$ is something that did not happen, the WOE formula is:
# 
# $$\text{WOE} = \ln{\frac{P(\neg E)}{P(E)}}$$
# 
# Taking the logorithm of a (positive) value has the effect of squashing that value down, e.g. from a potentially very large number to something more reasonable. It's easy to see how this value came to be useful in credit default settings, since in this setting the failure or defect rate is the default rate, which is a rare event, order of 1/1000 or rarer. It's also attractive if you're using logistic regression because look, a logarithm!
# 
# The weight of evidence encoder stars by dividing the categorical intervals into many different buckets. It then calculates the weight of evidence for each bucket. Buckets that are located next to one another contiguously and also have close weight of evidence values are then categorically combined, until only buckets that have relatively distinct weight of evidence values remain. Finally, the category level value is replaced with the weight of evidence value.
# 
# Weight of evidence encoding is akin to choosing "good splits" in a histogram. It achieves the following two things:
# 
# * It combines levels based on the observed hit rate for the target event.
# * It replaces level values with a logistic transform on the event incidence rate, which linearizes distributional differences in the data and can help parametric models like e.g. regression deal with the data.
# 
# It has the following limitation that it requires a binary target variable to work.
# 
# A good reference on weight of evidence is [this blog post](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html).

# In[ ]:


import category_encoders as ce
X_trans = ce.WOEEncoder().fit_transform(X, (y > 1).astype(int))
X_trans.head()


# ### James-Stein
# 
# The James-Stein encoder is the target encoder with an enhanced formula for estimating $\text{mean}(y_i)$, the canonical James-Stein equation. I didn't dig into this too much, but the `category_encoders` documentation has way more detail about this encoder than the other ones, and is a good read: http://contrib.scikit-learn.org/categorical-encoding/jamesstein.html.

# In[ ]:


import category_encoders as ce
X_trans = ce.JamesSteinEncoder().fit_transform(X, y)
X_trans.head()

