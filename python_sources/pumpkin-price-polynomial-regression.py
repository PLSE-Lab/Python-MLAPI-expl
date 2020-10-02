#!/usr/bin/env python
# coding: utf-8

# # Pumpkin Price Polynomial Regression
# 
# Note: this notebook is a continuation on [Pumpkin Price Linear Regression](https://www.kaggle.com/residentmario/pumpkin-price-linear-regression). You should start there.
# 
# Suppose that we have a target variable $y$ and, for a single records of interest, a set of predictor variables $x_n$. A linear regression model solves for a sequence of weights, $w$, which when multiplied against the data values, produces an estimated $\hat{y}$:
# 
# $$\hat{y} = w_0+w_1 x_1 + w_2 x_2 + ...$$
# 
# In OLS, this equation is solved to optimize a squared distance fitness metric. An optimal solution to this equation will minimize the square of the distance between $y$ and $\hat{y}$, for some large number of records.
# 
# There are two ways of changing things around. The first way is to change the metric that we optimize for. For example, what if instead of solving for least squares, we solved for least absolute values? It's just a question of how mathematically difficult this is to do. Mathematically, least squares turn out to be the easiest metric possible to solve for.
# 
# The other way to change things is to change the model equation. This equation is linear because all of the weights $w$ are first-order. This makes it easy to solve this equation using linear algebra matrices (see [the previous notebook](https://www.kaggle.com/residentmario/pumpkin-price-linear-regression) for this solution). However, this also assumes that our features are related in a linear way. Oftentimes, this is not true!
# 
# An easy way to extend regression to more complex feature relationships is to use a polynomial model. A second-order polynomial model (for two variables in these examples) looks like:
# 
# $$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + w_4 x_1^2 + w_5 x_2^2$$
# 
# I said earlier that equations are easiest to solve when they're linear, and this equation is no longer linear. What now?
# 
# We can use a cute trick to *make* it linear. Just define the following variables:
# 
# $$z_n = [x_1, x_2, x_1 x_2, x_1^2, x_2^2]$$
# 
# Then, relabeling the points:
# 
# $$\hat{y} = w_0 + w_1 z_1 + w_2 z_2 + w_3 z_3 + w_4 z_4 + w_5 z_5$$
# 
# Tada! The equation is linear again. We can solve this equation using ordinary least squares, same as before, then "downcast" the $z_n$ variables into $x_n$ ones.
# 
# That's how polynomial regression works.
# 
# Now let's look at the `scikit-learn` implementation.
# 
# We'll use polynomial regression to estimate the size of pumpkins sold in New York City, given their average price. The next code cell transforms the data into the shape we need it in:

# In[ ]:


import pandas as pd
import numpy as np
nyc_pumpkins = pd.read_csv("../input/new-york_9-24-2016_9-30-2017.csv")
cat_map = {
    'sml': 0,
    'med': 1,
    'med-lge': 2,
    'lge': 3,
    'xlge': 4,
    'exjbo': 5
}
nyc_pumpkins = nyc_pumpkins.assign(
    size=nyc_pumpkins['Item Size'].map(cat_map),
    price=nyc_pumpkins['High Price'] + nyc_pumpkins['Low Price'] / 2,
    size_class=(nyc_pumpkins['Item Size'].map(cat_map) >= 2).astype(int)
)
nyc_pumpkins = nyc_pumpkins.drop([c for c in nyc_pumpkins.columns if c not in ['size', 'price', 'size_class']], 
                                 axis='columns')
nyc_pumpkins = nyc_pumpkins.dropna()


# In[ ]:


nyc_pumpkins.head()


# In[ ]:


prices = nyc_pumpkins.values[:, :1]
sizes = nyc_pumpkins.values[:, 1:2]


# Now the implementation follows.
# 
# `scikit-learn` implements polynomial processing as a general-purpose preprocessor on the data. Here's how you would create and run the model:

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=2)
prices_poly = poly.fit_transform(prices)

clf = LinearRegression()
clf.fit(prices_poly, sizes)
predicted_sizes = np.round(clf.predict(prices_poly))


# For practice, we implement this pipeline by hand. The following code block does that.
# 
# This is a wee bit of an algorithmic adventure. If you're interested in understanding how this implementation works, try poking around in the `fit_transform` function yourself. I recommend putting a `import pdb; pdb.set_trace()` statement on the first line of the `__init__`, and then running `PolynomialFeatures().fit_transform(prices)` in a separate code block. This will drop you into [pdb debug mode](https://docs.python.org/3.6/library/pdb.html), which will let you figure out how this thing goes.
# 
# Note that the `LinearRegression` implementation is the same one we used in the [previous notebook](https://www.kaggle.com/residentmario/pumpkin-price-linear-regression).

# In[ ]:


import numpy as np
import itertools

class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree
    
    def fit_transform(self, X):
        nvars = X.shape[1]
        var_combos = []
        
        for i in range(0, self.degree):
            var_combos += itertools.combinations_with_replacement(set(range(nvars)), i + 1)
        
        mat = np.zeros((X.shape[0], len(var_combos)))
        
        for i, var_combo in enumerate(var_combos):
            mat[:, i] = np.prod(X[:, var_combo], axis=1)
        
        return mat
    

class LinearRegression:
    def __init__(self, degree=2):
        self.degree = degree
    
    def fit(self, X, y):
        self.betas = np.linalg.inv(X.T @ X) @ X.T @ y
        
    def predict(self, X):
        return X @ self.betas


# In[ ]:


poly = PolynomialFeatures(degree=2)
prices_poly = poly.fit_transform(prices)

clf = LinearRegression()
clf.fit(prices_poly, sizes)
predicted_sizes = np.round(clf.predict(prices_poly))


# Let's now look at our polynomial model performance. Here's a plot of the classification errors:

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

pd.Series((sizes - predicted_sizes).flatten()).value_counts().sort_index().plot.bar(
    title='$y - \hat{y}$'
)


# If our goal is to be accurate within plus-minus one class, then our model is actually fantastic!

# In[ ]:


pd.Series(
    np.abs((sizes - predicted_sizes).flatten()) <= 1
).value_counts().plot.bar(title='Accuracy Within 1 Class')


# That concludes this little two-notebook primer on regression.
