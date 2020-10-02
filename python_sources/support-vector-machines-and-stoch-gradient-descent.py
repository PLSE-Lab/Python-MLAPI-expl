#!/usr/bin/env python
# coding: utf-8

# # Support vector machines and stoch gradient descent
# 
# ## Support vector machines
# 
# **Support vector machines** (SVMs) are a popular supervised machine learning algorithm.
# 
# The natural setting for SVMs is classification. Suppose that we have a set of points in a two-dimensional feature space, with each of the points belonging to one of two classes. An SVM finds what is known as a **separating hyperplane**: a hyperplane (a line, in the two-dimensional case) which separates the two classes of points from one another. In the following diagram, $H_1$, $H_2$, and $H_3$ are all potential such hyperplanes:
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Svm_separating_hyperplanes_%28SVG%29.svg/512px-Svm_separating_hyperplanes_%28SVG%29.svg.png)
# 
# The points in this diagram are clearly **linearly separable**: there exist (infintely many) lines which may partition them from one another perfectly.  The $H_1$ candidate hyperplane is clearly suboptimal because it conflate the points belonging to either class. The $H_2$ candidate hyperplane is a good solution: it *linearly separates* the points. In fact, it separtes them so well that no points in the training set are ever misclassified!
# 
# Looking at the points, however, doesn't it seem like the $H_3$ candidate hyperplane is a better solution than $H_2$, however? That's because $H_3$ has the propertly that it maximizes the distance between itself and the nearest point in either class cluster. In other words, $H_3$ is the *maximum-margin hyperplane*. Put another way, while $H_2$ and $H_3$ will have the same training error (0%) we expect $H_3$ to generalize better to new, unseen data, as new points are less likely to "accidentally" stray into the other class's partition of the space.
# 
# A support vector machine works by finding this separating hyperplane (just a line in this case, but as many dimensions as in the feature space in the general case). Any new points get classified according to their orientation with respect to the separating hyperplane.
# 
# Sometimes the points are not linearly separable. For example, suppose that the classes are distributed according to the diagram on the left here:
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/1/1b/Kernel_Machine.png)
# 
# In that case there is no line we can draw that will separate the classes. However, what we *can* do in this case is transform the space somehow: just remap the points so  that in the new space, they *are* separable. Then build that separator in the new space! This is what is done on the right-hand side.
# 
# The function responsible for transforming the space so that the resulting points are separable is known as the **kernel**.
# 
# The simplest possible support vector machine, the linear SVM, has a linear kernel; it divides the feature space into a set of lines (hyperplanes in higher-dimensional spaces). Other possible kernels include degree-$n$ polynomial kernels, which separate the space using degree-$n$ functional curves, and other more exotic forms. Here's an example of a few results from the `sklearn` documentation:
# 
# ![](https://imgur.com/1pq2T2K)
# 
# SVMs can be extended to $n$-class multiclass classification using various tricks also used for other classifiers. One such trick is one-versus-many classification, where $n$ different SVMs are trained based on the set of points in the class and not in the class. These decision boundaries are then composed to determine overall boundaries. A different trick is to do one-against-one classification, in which you cross-join the classes and build antogonistic classifiers ($(n(n-1))/2$ of them).
# 
# The manner in which the boundaries are composed is technically involved, but is a solved problem computationally.
# 
# Finally, since SVMs are non-parametric, we don't have get classification probabilities from the algorithm itself. It's possible to back them out using expensive bootstrapping and cross-validation.

# ### Performance characteristics
# 
# The `sklearn` documentation has the following to say on the utility of SVMs:
# 
# > The advantages of support vector machines are:
# 
# >> Effective in high dimensional spaces.
# 
# >> Still effective in cases where number of dimensions is greater than the number of samples.
# 
# >> Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
# 
# >> Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
# The disadvantages of support vector machines include:
# 
# > The disadvantages of support vector machines include:
# 
# >> If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
# 
# >> SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
# 
# Support vector classifiers are provided in `sklearn` under the names `SVC`, `NuSVC`, and `LinearSVC`. Extensions to the regression case, not discussed here, are also provided. For more deets see [the documentation](http://scikit-learn.org/stable/modules/svm.html).
# 
# There are plenty of examples of kernels on Kaggle using SVMs, you can do a search to see a few. [Here's one example](https://www.kaggle.com/nirajvermafcb/support-vector-machine-detail-analysis?scriptVersionId=940166).

# ## Stochastic gradient descent
# 
# **Stochastic gradient descent** is a technique for converging on a solution to a problem by choosing an arbitary solution, measuring the goodness of fit (under a loss function), and then iteratively taking steps in the direction that minimizes loss (by stepping in the direction of the derivative). For a broad class of loss functions which have the property known as "convexity", stochastic gradient descent is an easy non-parametric way of solving a machine learning problem or implementing an algorithm. For example, I implemented ridge regression this way in [this old notebook](https://www.kaggle.com/residentmario/gradient-descent-with-linear-regression/).
# 
# The support vector machine algorithm used in `sklearn` is  implemented using stochastic descent to converge on a solution.  But you can also access the stochastic gradient descent algorithm directly through a pair of separate implementations in `sklearn`, `SGDClassifier` and `SGDRegressor`! `SGDClassifier` provides a handful of different loss functions, but of these, `loss="hinge"`, the default, is the one which causes `SGDClassifier` to perform equivalently to a linear SVM. Specifying `loss='squared_loss'` gets you linear regression, whilst `loss="log"` results in logorithmic regression!
# 
# If you can state your machine learning algorithm as "minimize this cost function", and the cost function in question is convex, stochastic gradient descent may be used to implement said algorithm, and so an option may be made available in `SGDClassifier` or `SGDRegressor` for doing so. So for example, since we can rewrite "perform linear regression on this data" as "minimize the sum of squares residual", we can implement a SGD solution for this problem.
# 
# This does not preclude other solutions to these algorithms. In most cases linear regression is better solved by performing matrix factorization, for example. But stochastic gradient descent has a number of attractive properties. SGD scale easily to thousands of samples and thousands of features, deals well with sparse data (making this technique popular in NLP for example), and expose a `partial_fit` function that can be used to fit on extremely large (chunked) datasets and on streaming data. The `sklearn` implementation furthermore has been designed to work well with huge ($>10^5$) datasets, as it takes a representative sample of the dataset and trains on that. `SVC`, by contrast, trains on every record in the dataset, and thus has vastly inferior scaling properties.
# 
# Thus `SGDClassifier` and `SGDRegressor` essentially curate a selection of implementations of machine learning algorithms, like linear support vector machine and logistic regression, from the perspective of its algorithmic implementation. To borrow from grade school mathematics, algorithms like `SVC` are a geometric solution whilst SGD is an algebraic solution.
# 
# For more on stochastic gradient descent, including notes on its implementation, [check out the requisite section of the `sklearn` documentation](http://scikit-learn.org/stable/modules/sgd.html).
