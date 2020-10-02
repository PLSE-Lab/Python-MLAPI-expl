#!/usr/bin/env python
# coding: utf-8

# # Vapnik's theory , Decomposition bias variance of the prediction error and Learning curve

# <img src="https://le-datascientist.fr/wp-content/uploads/2019/01/machineLearning-1024x698.jpg" alt="Meatball Sub" width="500"/>

# In[ ]:


from IPython.display import Latex
from IPython.display import Image


# # The robustness of a predictive model

# The model should depend as little as possible on the particular learning sample used and generalize well to other samples. It must be as sensitive as possible to the random fluctuations of certain variables. Even if the data evolves over time, the model must be able to continue to apply correctly to new samples, for a reasonable period of time, which is inevitably shortened in the event of significant technical or economic development, but which in any case depends on the speed of evolution of offers and audiences: an unscore for marketing will probably last less in mobile telephony than in banking. Stability over time will, if possible, be tested when the model is built, if there is a sample out of time for the tests, i.e. a sample drawn up at a period other than the training sample. The model must obviously not be based on variables that are insufficiently reliable, difficult to obtain or unstable from one sample to another or from one period to another.

# # Vapnik's theory 

# To assess the predictive quantity of a model, we can measure the prediction error by various loss functions. Among the most frequent, we find: The quadratic function when y is quantitative.
# \begin{equation}
# L(y,f(x)) = (y-f(x))^2
# \end{equation} 

# We define risk as the expectation (average value) of the loss function, but since we do not know the joint probability law of x and y, we can only estimate the risk. The most common estimate is the empirical risk formula:
# \begin{equation}
#     \frac{1}{n} \sum_{i=1}^n (y_i-f(x_i))^2 
# \end{equation}
# where n is the sample size. We find in this last formula the error rate.

# We know that the empirical risk measured on the learning sample presents an optimistic bias: it is generally lower than the real risk. The latter is best estimated by the empirical risk measured on a so-called test sample. The question arises of the convergence of the two curves (test data and learning data) towards a common value.
# Indeed, if from a certain value of n, the two curves are close, this means that the discriminating power of the fitted model on the n observations of the training sample will probably generalize well to other samples.

# On the theoretical level, Vladimir Vapnik was interested in the convergence of the empirical risk on the learning sample $ R_ {emp} $ towards the risk R (whose empirical risk on the test sample is intended to be an approximation) , and demonstrated two fundamental results on this convergence, one on the existence of a convergence and the other on the speed of convergence.

# We define the quantity linked to the model and called the Vapnik-Chervonenkis dimension (we denote VC-dimension). The VC-dimension is a measure of complexity of a model which is actually defined for any family of functions $\mathbb{R}^p \rightarrow \mathbb{R}$ whose point separating power of $\mathbb{R}^p$.
# 
# The importance of this notion comes from the two results of Vapnik:
# - The empirical risk on the learning sample $ R_{emp} $ of a model converges towards its risk R (We say that the model is consistent) if and only if its VC-dimension is finite.
# - When the VC-dimension h of a model is finished, we have, with a probability of error $\alpha$:
# 
# \begin{equation}
#     R < R_{emp}+ \sqrt{\frac{h(\log{(2n/h)}+1)-\log{(\alpha/4)}}{n}}  \text{  (*)  }
# \end{equation}

# ![](http://)

# The condition of finitude imposed on the VC-dimension, to ensure the convergence of $R_ {emp}$ towards $R$, is not trivial.
# The theoretical interest of the risk increase is that it is universal: it applies to all models, without any particular assumption on the joint law of x and y. This upper bound is universal, as is for example the berry-Esseen bound in the framework of the central-limit theorem. Of course, like this bound, the Vapnik upper bound is far from optimal in the special cases where a better upper bound can be found. Indeed, the VC-dimension of a model is in some simple cases (linear model) equal the number of parameters, but it is most often difficult to calculate and even to increase efficiently, which limits the practical interest of the increase (*).

# However, it should be noted that the increase (*) is only true with a given probability $\alpha $, and that the upper bound tends to infinity when $ \alpha $ tends to 0.
# 
# The inequality (*) gave rise to the SRM (Structural Risk Minimization) approach in which we consider nested models of increasing VC-dimension $ h_1 <h_2 <.. $, as we do with logistic models where by neural networks. When the VC-dimension increases, the empirical risk generally decreases on average while $\sqrt{\frac{h (\log{(2n / h)} + 1) - \log{(\alpha / 4)}} { n}} $ increases.

# # Decomposition bias variance of the prediction error

# Suppose you are given a well-adapted machine learning model $ \hat{f} $ that we want to apply to a set of test data. For example, the model could be a linear regression whose parameters were calculated using a learning set different from your test set. For each point x of our test, we want to predict the associated target $ y \in \mathbb {R} $ and calculate the mean square error MSE *(mean square error)*:
# \begin{align*}
#     MSE = E_{test}{|\hat{f}(x)-y|^2}
# \end{align*}
# 
# Suppose that the points in our training / test dataset all come from a similar distribution, with:
# \begin{equation*}
#     y_i = f(x_i) + \epsilon_i \text{ avec } E{[\epsilon_i]}=0, Var{[\epsilon_i]}=\sigma^2
# \end{equation*}
# 
# Our goal is to calculate $ f $. By looking at our training set, you get an estimate $ \hat{f} $. Now use this estimate with your test set, which means that for each example $ i $ in the test set, our prediction for $ y_i = f (x_i) + \epsilon_i $ is $ \hat {f} (x_i) $.
# Here, $ x_i $ is a fixed real number, this $ f(x_i) $ is fixed, and $ \epsilon_i $ is a real random variable with an average of 0 and a variance $ \sigma^2 $. We know that $ \hat {f} (x_i) $ is random because it depends on the i values of the training data set. This is why we talk about the bias $ E {[\hat {f} (x) -y]} $ and the variance of $ \hat{f} $. We can now calculate our MSE on the test dataset by calculating $ E {[\hat {f} (x) -y]} $ with respect to the training sets.
# 
# 
# \begin{align*}
#     MSE &= E_{test}{\left[(\hat{f}(x)-y)^2\right]} \\ &= E{\left[(\epsilon + f(x) - \hat{f}(x))^2\right]}
#     \\ &= \sigma^2 + E{\left[(f(x) - \hat{f}(x))^2\right]}
#     \\ &= \sigma^2 + \left(E{[(f(x) - \hat{f}(x))]}\right)^2 + Var{\left[f(x) - \hat{f}(x)\right]}
#     \\ &= \sigma^2 + Bias^2 + Variance
# \end{align*}
# 
# There is nothing we can do on the first term $ \sigma^2 $ because we cannot predict noise by definition. The term bias is due to under-adjustment, which means that, on average, $ \hat {f} $ does not predict $ f $. The last term is closely related to the over-adjustment, the prediction $ \hat {f} $ is too close to the values $ y_{train} $ and varies a lot with the choice of our training games.

# **To resume :**
# 1.  **Bias**: this component of the generalization error is due to bad assumptions, like for example to suppose that the data are linear when they are quadratic. A high bias model is more likely to under-adjust training data. High bias $ \longleftrightarrow $ Under-adjustment.
# 2.  **Variance**: this component of the error is due to the excessive sensitivity of the model to small variances in the training game. A model with many degrees of freedom (such as a high degree polynomial model) will likely have a high variance, and therefore will over-adjust the training data. High variance $ \longleftrightarrow $ Over-adjustment.
# 3. **Irreducible error**: it is due to the noise present in the data. The only way to reduce this component of the error is to clean the data (that is, to repair the data sources, for example faulty sensors, to detect and remove outliers, etc.) .

# Therefore, when analyzing the performance of a machine learning algorithm, we must always ask ourselves how to reduce the bias without increasing the variance and respectively how to reduce the variance without increasing the bias. Most of the time, reducing one will increase the other.

# # Learning curve: Example on polynomial regression

# Suppose that the complexity of our data cannot be modeled by a simple straight line?. One way to do this is to add the powers of each variable as new variables, then train a linear model on this new data set: this technique is called polynomial regression.
# 

# In[ ]:


Image("../input/polynome1/poly1.png")


# The previous figure, for example, applies a polynomial model of degree 30 to the training data, then compares the result to a linear model and to a quadratic model. We observe that the 30 degree model undulates to get as close as possible to training observations. 
# 
# Obviously, this high degree polynomial model significantly over-adjusts the training data, while the linear model under-adjusts them. The model that will generalize best in this case is the second degree model. How can we determine if our model over-adjusts the data?
# 
# There is a way which consists in looking at the **learning curves**: these are diagrams representing the results obtained by the model on the training game.

# In[ ]:


Image("../input/polynome2/poly2.png")


# When the training game has only one or two observations, the model can adjust them perfectly, which is why the curve in red starts at zero. But as more observations are added to the training game, the model has more and more difficulty adjusting them, partly because of the noise and because the model is not linear. This is why the error on the training game increases until it reaches a plateau: from there, the addition of new training observations does not change the average error much. 
# 
# Now let's look at the performance of the model on the validation data (the blue curve): when the model is trained from very few observations, it is unable to generalize the model correctly, this is why the validation error is important at the beginning. Then the model improves as it receives more training data, which is why the validation error decreases slowly.
# 
# These two learning curves are characteristic of a model that underfitting: the two curves reach a plateau, they are close and relatively high.

# Let us now examine the learning curves of a polynomial model of degree 2 on the same data:

# In[ ]:


Image("../input/polynome3/poly3.png")


# These two learning curves look a bit like the previous ones, but there is an important difference: the error on the training data is much lower than that of the linear regression model.
