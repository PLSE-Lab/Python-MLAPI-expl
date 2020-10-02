#!/usr/bin/env python
# coding: utf-8

# # ML visualization with yellowbrick (2)
# 
# ## Introduction
# 
# ...
# 
# ## Data
# 
# For this exploration we'll use the tennis odds dataset. For regression we'll explore how well bookies predict one another's odds, picking on `betrally` as the target variable (a case when one bookie is systematically lowballing or highballing would be an [arbitrage](https://www.investopedia.com/terms/a/arbitrage.asp) opportunity!). For classification we'll classify whether or not the player winning the match gets predicted from the betting odds.

# In[29]:


import pandas as pd
import numpy as np

matches = pd.read_csv("../input/t_odds.csv")
matches.head()
book = matches.loc[:, [c for c in matches.columns if'odd' in c]].dropna()

bookies = set([c.split("_")[0] for c in matches.columns if'odd' in c])

def deltafy(srs):
    return pd.Series(
        {b: srs[b + '_player_1_odd'] - srs[b + '_player_2_odd'] for b in bookies}
    )

book = book.apply(deltafy, axis='columns')
book = book.assign(
    winner=(matches.player_1_score > matches.player_2_score).astype(int).loc[book.index]
)

X = book.iloc[:, :-1].loc[:, [c for c in X if c != 'betrally']]
y = book.loc[:, 'betrally']

book.head()


# ## Regression

# In[30]:


from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = Ridge()


# ### ResidualsPlot
# 
# An important assumption behind linear regression is that of homoskedaity: that the error, or residuals, between the predicted values and the actual values is random. Non-random error indicates that there is a pattern to the dataset that the linear model is not sufficiently complex to capture, for example, you are trying to apply linear regression to a pattern which is occurring in a polynomial feature space. Data that does not exhibit randomly distributed residuals is thus a poor fit for linear regression (though it can still be modeled this way, in a pinch).
# 
# This plot type is a neat and tidy built-in for plotting this yourself.

# In[31]:


from yellowbrick.regressor import ResidualsPlot

vzr = ResidualsPlot(clf)
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()


# ### PredictionError
# 
# `PredictionError` summarizes the level of fit of your plot. It's similar to a pre-configured `seaborn` `jointplot` (see [here](https://seaborn.pydata.org/generated/seaborn.jointplot.html) for that).

# In[32]:


from yellowbrick.regressor import PredictionError

vzr = PredictionError(clf)
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()


# In this case bookies predict each other really, really, *really* well; the amount of error in the model is so tiny that the $R^2$ score is 1 to three decimal places! Here's a more illustrative example of what this will look like on well-posed problems, taken from the `yb` docs:
# 
# ![](https://i.imgur.com/pn37qvD.png)
# 
# The identity fit is meaningful because its $R^2$ score is exactly 0 (read more on that [here](https://www.kaggle.com/residentmario/model-fit-metrics/)).

# ### AlphaSelection
# 
# The last visualization in this chunk of `yellowbrick` is an alpha selection utility. $\alpha$ is a parameter controlling how much regularization (L1 regularization for Lasso, L2 for Ridge) the given model is given, and is used to tune a model that overfitting. I demonstrate cross validation in [this past notebook](https://www.kaggle.com/residentmario/gaming-cross-validation-and-hyperparameter-search/).
# 
# This method requries the version of the linear regression models with built-in cross validation&mdash;`RidgeCV` for ridge regression, `LassoCV` for lasso regression, and `ElasticNetCV` for elastic net.

# In[52]:


from yellowbrick.regressor import AlphaSelection
from sklearn.linear_model import RidgeCV

alphas = np.linspace(0.01, 10, 100)

clf = RidgeCV(alphas=alphas)
vzr = AlphaSelection(clf)

vzr.fit(X_train, y_train)
vzr.poof()


# In this case we see that the best $\alpha$ value is 2.835. Notice the y-axis, however. Regularization is only affecting the overall fit in the fourth decimal place!
# 
# For what it is this is a pretty performance chart which solves a need. However, I am again left with the desire to go further than this. How about tuning two hyperparameters simultaneously? Three? Why does it have to be $\alpha$&mdash;can't I use this to e.g. visualize myself tuning `l1_ratio` in an `ElasticNet` instead? Hmm.

# ## Classification

# In[76]:


X = book.iloc[:, :-1]
y = book.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ### ConfusionMatrix
# 
# A classic of the genre. The confusion matrix simply states the number of times each of the possible configurations of actual and predicted classes occurred. By inspecting these counts we can see what classes get "confused" for one another. In the binary case, as here, the boxes translate directly to the True Positive, False Positive, True Negative, and False Negative metrics. For more on this topic see [this notebook](https://www.kaggle.com/residentmario/classification-metrics-with-seattle-rain/). This visualization is just a convenient wrapper to `seaborn.heatmap`.

# In[77]:


from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ConfusionMatrix

clf = GaussianNB()
vzr = ConfusionMatrix(clf)
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()


# ### ClassificationReport
# 
# This one outputs the performance of the data on a per-class basis for three of the most common and important, and inter-related, classification measurements: precision (TP / (TP + FP)), recall (TP / (TP + FN)), and the F1 score (the precision-recall harmonic mean). Again [this notebook](https://www.kaggle.com/residentmario/classification-metrics-with-seattle-rain/) for more details.
# 
# It's interesting that our model performs marginally better on Player 2 wins than on Player 1 wins. In other words it predicts 0 with more accuracy than it predicts 1.

# In[82]:


from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ClassificationReport

clf = GaussianNB()
vzr = ClassificationReport(clf, classes=['P1 Win', 'P2 Win'][::-1])
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()


# ### ROCAUC
# 
# A classic of the genre. ROCAUC is probably the most heavily used more-complex-than-a-matrix classification report out there, and it's fantastic to have this available as such a convenient built-in! Again [this notebook](https://www.kaggle.com/residentmario/classification-metrics-with-seattle-rain/) for more details, if you're not familair.
# 
# Looks like the AUC for both P1 and P2 classes is the same, and the curves are almost the same as well. 

# In[85]:


from yellowbrick.classifier import ROCAUC

clf = GaussianNB()
vzr = ROCAUC(clf, classes=['P1 Win', 'P2 Win'][::-1])
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()


# ### ClassBalance
# 
# This method simply plots the number of times either class is predicted by the model. I don't find to be very useful, as you're not really saving yourself that much effort with this one&mdash;it's really just a simple bar chart...

# In[86]:


from yellowbrick.classifier import ClassBalance

clf = GaussianNB()
vzr = ClassBalance(clf, classes=['P1 Win', 'P2 Win'][::-1])
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()


# ### ClassPredictionError
# 
# This chart is similarly problematic. Not only is it easy to generate yourself, I also think that its current layout could be vastly improved! I find it very confusing, wouldn't use it.

# In[87]:


from yellowbrick.classifier import ClassPredictionError

clf = GaussianNB()
vzr = ClassPredictionError(clf, classes=['P1 Win', 'P2 Win'][::-1])
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()


# ### ThreshViz
# 
# Precision and recall are fundamentally related to one another, with an increase in one coming at the cost of a decrease in the other (read [here](https://www.kaggle.com/residentmario/classification-metrics-with-seattle-rain/) for a quick discussion on why this is and why it matters).
# 
# This is a very useful visualization, but takes a bit of thought to understand. To understand it, imagine that we are building a search engine, and that this classifier is being used to determine whether or not some set of records is relevant or not relevant to the given search query. The y-axis is how the given metric (precision or recall) scores (1 being best performance and 0 worst). The x-axis is the percentage of records withheld. Thus 0 in the x-axis is the case when 100% of all records are being returned; in this case we will trivially have a recall score of 1, as if we return everything, every relevant result will be in the user's hands. Simultaneously, precision will be the balance of the classes. Close to 1 on the x-axis corresponds with very few records being returned; in this case we will be extremely good at returning only relevant records, resulting in a precision score of 1, but our recall will be near 0 because very few of the total relevant records are returned. At each step on the curve in between, the next remaining highest-probability point is chosen, thus we are choosing sure winners at first, and sure losers later on.
# 
# The shape of the lines tells us something about the dataset overall.
# 
# The high slope to precision at the edge of the graph tells us that odds extremely heavily in favor of one player or another at the betting booth, once interpreted by our Naive Bayes classifier, do not actually correspond with better chance the player will win. The curve incredibly quickly stabilizes on a slightly-better-than-random precision score of 0.65, and only dips to 0.5 at the other extreme end. Winner classification seems to be about that performant across all matches (remember: not that many Federer versus nobody matches in this dataset!).
# 
# The recall curve tells a similar story.
# 
# The queue rate is just 1 - precision.
# 
# Very useful!

# In[94]:


from yellowbrick.classifier import ThreshViz

clf = GaussianNB()
vzr = ThreshViz(clf, classes=['P1 Win', 'P2 Win'][::-1])
vzr.fit(X_train, y_train)
vzr.poof()


# # Conclusion
# 
# The adventure continues in https://www.kaggle.com/residentmario/ml-visualization-with-yellowbrick-3/
