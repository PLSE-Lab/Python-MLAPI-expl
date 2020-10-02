#!/usr/bin/env python
# coding: utf-8

# # Conceptual Exercise 1
# 
# ## Question
# 
# Using a little bit of algebra, prove that the logistic function representation and logit representation for the logistic regression model are equivalent. In other words, prove that 
# 
# \begin{equation}
#     p(X) = \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}
# \end{equation}
# 
# is equivalent to
# 
# \begin{equation}
#     \frac{p(X)}{1 - p(X)} = e^{\beta_0 + \beta_1X}.
# \end{equation}

# ## Answer
# 
# Starting with the logistic function representation, we multiply both sides by $1 + e^{\beta_0 + \beta_1X}$.
# 
# \begin{equation}
#     e^{\beta_0 + \beta_1X} = p(X) \left( 1 + e^{\beta_0 + \beta_1X} \right) = 
#     p(x) + p(x)e^{\beta_0 + \beta_1X}
# \end{equation}
# 
# Next, we subtract $p(x)e^{\beta_0 + \beta_1X}$ from both sides.
# 
# \begin{equation}
#     p(x) = e^{\beta_0 + \beta_1X} - p(x)e^{\beta_0 + \beta_1X} =
#     e^{\beta_0 + \beta_1X} \left( 1 - p(X) \right)
# \end{equation}
# 
# Finally, we divide both sides by $1 - p(X)$ to get the logit representation.
# 
# \begin{equation}
#     \frac{p(X)}{1 - p(X)} = e^{\beta_0 + \beta_1X}
# \end{equation}

# # Conceptual Exercise 2
# 
# ## Question
# 
# It was stated in the text that classifying an observation to the class for which 
# 
# \begin{equation}
#     p_k(x) = \frac{\pi_k \frac{1}{\sigma \sqrt{2\pi}}\exp \left( -\frac{1}{2\sigma^2} (x - \mu_k)^2 \right)}{\sum_{l = 1}^K \pi_l \frac{1}{\sigma \sqrt{2\pi}}\exp \left( -\frac{1}{2\sigma^2} (x - \mu_l)^2 \right)}
# \end{equation}
# 
# is largest is equivalent to an observation to the class for which
# 
# \begin{equation}
#     \delta_k(x) = x \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k)
# \end{equation}
# 
# is the largest. Prove that this is the case. In other words, under the assumption that the observations in the $k$th class are drawn from an $N(\mu_k, \sigma^2)$ distribution, the Bayes' classifier assigns an observation to the class for which the discriminant function is maximized.

# ## Answer
# 
# Since $\log(x)$ is a monotone increasing function, the class $k$ for which $p_k(x)$ is maximized is also the class for which $\log(p_k(x))$ is maximized. Taking the natural logarithm of both sides of the Bayes' classifier formula, we have
# 
# \begin{align}
#     \log(p_k(x)) &= \log \left( \frac{\pi_k \frac{1}{\sigma \sqrt{2\pi}}\exp \left( -\frac{1}{2\sigma^2} (x - \mu_k)^2 \right)}{\sum_{l = 1}^K \pi_l \frac{1}{\sigma \sqrt{2\pi}}\exp \left( -\frac{1}{2\sigma^2} (x - \mu_l)^2 \right)} \right) \\
#     &= \log \left( \frac{\pi_k}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu_k)^2}{2\sigma^2}} \right) - \log \left( \sum_{l = 1}^K \frac{\pi_l}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu_l)^2}{2\sigma^2}} \right) \\
#     &= \log \left( e^{-\frac{(x - \mu_k)^2}{2\sigma^2}} \right) + \log(\pi_k) - \log \left( \sigma\sqrt{2\pi} \right) - \log \left( \sum_{l = 1}^K \frac{\pi_l}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu_l)^2}{2\sigma^2}} \right) \\
#     &= \frac{-(x - \mu_k)^2}{2\sigma^2} + \log(\pi_k) - \log \left( \sigma\sqrt{2\pi} \right) - \log \left( \sum_{l = 1}^K \frac{\pi_l}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu_l)^2}{2\sigma^2}} \right) \\
#     &= \frac{-x^2 + 2x\mu_k - \mu_k^2}{2\sigma^2} + \log(\pi_k) - \log \left( \sigma\sqrt{2\pi} \right) - \log \left( \sum_{l = 1}^K \frac{\pi_l}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu_l)^2}{2\sigma^2}} \right) \\
#     &= x\frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k) - \frac{x^2}{2\sigma^2} - \log \left( \sigma\sqrt{2\pi} \right) - \log \left( \sum_{l = 1}^K \frac{\pi_l}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu_l)^2}{2\sigma^2}} \right).
# \end{align}
# 
# The last three terms are independent of the class $k$, so we can combine them all into a single constant $C$. Thus
# 
# \begin{equation}
#     \log(p_k(x)) = x\frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k) + C
#     = \delta_k(x) + C.
# \end{equation}
# 
# From this equation, we see that the class for which $\log(p_k(x))$ is maximized is the class $k$ for which $\delta_k(x)$ is maximized. Hence, under the assumption that the observations in the $k$th class are drawn from an $N(\mu_k, \sigma^2)$ distribution, the Bayes' classifier assigns an observation to the class $l$ for which the discriminant function $\delta_l(x)$ is maximized. This completes the proof.

# # Conceptual Exercise 3
# 
# ## Question
# 
# This problem relates to the QDA model, in which the observations within each class are drawn from a normal distribution with a class-specific mean vector and a class specific covariance matrix. We consider the simple case where $p = 1$; i.e. there is only one feature.
# 
# Suppose we have $K$ classes, and that if an observation belongs to the $k$th class then $X$ comes from a one-dimensional normal distribution $X \sim N(\mu_k, \sigma_k^2)$. Recall that the density function for the one-dimensional normal distribution is given by
# 
# \begin{equation}
#     f_k(x) = \frac{1}{\sigma_k \sqrt{2\pi}} \exp \left( -\frac{1}{2\sigma_k^2}(x - \mu_k)^2 \right)
# \end{equation}
# 
# Prove that in this case, the Bayes' classifier is *not* linear. Argue that in fact it is quadratic.
# 
# *Hint: For this problem you should follow the arguments laid out in Section 4.4.2, but without making the assumption that $\sigma_1^2 = \cdots = \sigma_K^2$.*

# ## Answer
# 
# Let $\pi_k$ represent the overall, or *prior*, probability that a randomly chosen observation comes from the $k$th class, and let $f_k(x)$ be the probability density function of $X$ for an observation that comes from the $k$th class. Then, Bayes' theorem states that $p_k(x)$, the probability that an observation $X$ comes from the $k$th class, given $X = x$, is given by
# 
# \begin{equation}
#     p_k(x) = \frac{\pi_k f_k(x)}{\sum_{l = 1}^K \pi_l f_l(x)}.
# \end{equation}
# 
# The Bayes' classifier assigns an observation $X = x$ to the class $k$ for which $p_k(x)$ is maximized. As discussed in Exercise 2 above, maximizing $p_k(x)$ is equivalent to maximizing $\log(p_k(x))$, so we take the natural logarithm of both sides.
# 
# \begin{equation}
#     \log(p_k(x)) = \log(f_k(x)) + \log(\pi_k) - \log \left( \sum_{l = 1}^K \pi_l f_l(k) \right)
# \end{equation}
# 
# Now we use the fact that $f_k(x)$ is the probability density function for a one-dimensional normal distribution $N(\mu_k, \sigma_k^2)$.
# 
# \begin{align}
#     \log(f_k(x)) &= \frac{-(x - u_k)^2}{2\sigma_k^2} - \log \left( \sigma_k \sqrt{2\pi} \right) \\
#     &= x^2 \left( \frac{-1}{2\sigma_k^2} \right) + x \left( \frac{\mu_k}{\sigma_k^2} \right) - \frac{\mu_k^2}{2\sigma_k^2} - \log \left( \sigma_k \sqrt{2\pi} \right)
# \end{align}
# 
# Plugging this back into the original equation for $\log(p_k(x))$ and recombining some terms, we get
# 
# \begin{equation}
#     \log(p_k(x)) = x^2 \left( \frac{-1}{2\sigma_k^2} \right) + x \left( \frac{\mu_k}{\sigma_k^2} \right) - \frac{\mu_k^2}{2\sigma_k^2} - \frac{1}{2}\log(\sigma_k^2) + \log(\pi_k) + C,
# \end{equation}
# 
# where $C$ is a constant consisting of all the terms which do not depend on $x$ or $k$. In other words,
# 
# \begin{equation}
#     C = -\frac{1}{2}\log(2\pi) - \log \left( \sum_{l = 1}^K \pi_l f_l(k) \right).
# \end{equation}
# 
# As we can see, $\log(p_k(x))$ is a function which is quadratic in $x$. Hence, the Bayes' classifier is in fact quadratic, and not linear. Also note that the part of $\log(p_k(x))$ which is dependant on $x$ and $k$ matches up with the quadratic discriminant $\delta_k(x)$ given in Section 4.4.4 for the special case of $p = 1$.
# 
# \begin{align}
#     \delta_k(x) &= -\frac{1}{2}(x - \mu_k)^T\Sigma_k^{-1}(x - \mu_k) - \frac{1}{2}\log|\Sigma_k| + \log\pi_k \\
#     &= -\frac{1}{2}x^T\Sigma_k^{-1}x + x^T\Sigma_k^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma_k^{-1}\mu_k - \frac{1}{2}\log|\Sigma_k| + \log\pi_k
# \end{align}

# # Conceptual Exercise 4
# 
# ## Question
# 
# When the number of features $p$ is large, there tends to be a deterioration in the performance of KNN and other *local* approaches that perform prediction using only observations that are *near* the test observation for which a prediction must be made. This phenonmenon is known as the *curse of dimensionality*, and it ties into the fact that non-parametric approaches often perform poorly when $p$ is large. We will now investigate this curse.
# 
# 1. Suppose we have a set of observations, each with measurements on $p = 1$ feature, $X$. We assume that $X$ is uniformly (evenly) distributed on $[0, 1]$. Associated with each observation is a response value. Suppose that we wish to predict a test observation's response using only observations that are within 10% of the range of $X$ closest to that test observation. For instance, in order to predict the response for a test observation with $X = 0.6$, we will use observations in the range $[0.55, 0.65]$. On average, what fraction of the available observations will we use to make the prediction?
# 
# 2. Now suppose that we have a set of observations, each with measurements on $p = 2$ features, $X_1$ and $X_2$. We assume that $(X_1, X_2)$ are uniformly distributed on $[0, 1] \times [0, 1]$. We wish to predict a test observation's response using only observations that are within 10% of the range of $X_1$ *and* within 10% of the range of $X_2$ closest to that test observation. For instance, in order to predict the response for a test observation with $X_1 = 0.6$ and $X_2 = 0.35$, we will use observations in the range $[0.55, 0.65]$ for $X_1$ and in the range $[0.3, 0.4]$ for $X_2$. On average, what fraction of the available observations will we use to make the prediction?
# 
# 3. Now suppose that we have a set of observations on $p = 100$ features. Again the observations are uniformly distributed on each feature, and again each feature ranges in value from 0 to 1. We wish to predict a test observation's response using observations within the 10% of each feature's range that are closest to the that test observation. What fraction of available observations will we use to make the prediction?
# 
# 4. Using your answers to parts (1)-(3), argue that a drawback of KNN when $p$ is large is that there are very few training observations "near" any given test observation.
# 
# 5. Now suppose that we wish to make a prediction for a test observation by creating a $p$-dimensional hypercube centered around the test observation that contains, on average, 10% of the training observations. For $p = 1, 2, 100$, what is the length of each side of the hypercube? Comment on your answer.
# 
# *Note: A hypercube is a generalization of a cube to an arbitrary number of dimensions. When $p = 1$, a hypercube is simply a line segment, when $p = 2$ it is a square, and when $p = 100$ it is a 100-dimensional cube.*

# ## Answer
# 
# 1. In the case of $p = 1$ feature, $X$, which is uniformly distributed on $[0, 1]$, if we wish to predict a test observation's response using only observations that are within 10% of the range of $X$ closest to that test observation, the interval containing the observations used for prediction will be $[X - 0.05, X + 0.05]$ if $0.5 \leq X \leq 0.95$, which has a length of $0.10$. If $0 \leq X < 0.05$, then the interval used for prediction will be $[0, X + 0.05]$, which has a length of $X + 0.05$. Similarly, if $0.95 < X \leq 1$ the the interval used for prediction will be $[X - 0.05, 1]$, which has a length of $1.05 - X$. Let $L(x)$ be the length of the interval used for prediction associated with a test observation $X = x$. Then, since $X$ is assumed to be uniformly distributed on $[0, 1]$, the average length of the interval used for prediction is the average of $L(x)$ on $[0, 1]$. From calculus, the average value of an integrable function $f(x)$ on the interval $[a, b]$ is given by
# \begin{equation}
#     \frac{1}{b - a} \int_a^b f(x) \, dx
# \end{equation}
# Thus, the average length of the interval used for prediction is
# \begin{equation}
#     \frac{1}{1 - 0} \int_0^1 L(x) \, dx = 
#     \int_0^{0.05} x + 0.05 \, dx + \int_{0.05}^{0.95} 0.1 \, dx + \int_{0.95}^1 1.05 - x \, dx
#     = .0975.
# \end{equation}
# In other words, on average 9.75% of the available observations will be considered when making the prediction. Note that this aligns with the intuition that about 10% of the available observations, on average, will be considered when making the prediction.
# 
# 2. Assuming $X_1$ and $X_2$ are independent and $(X_1, X_2)$ is uniformly distributed on $[0, 1] \times [0, 1]$, we know from Part 1 for a fixed value of $X_1$ on average 9.75% of the observations when varying $X_2$ will be available for making a prediction. Likewise, the same is true for a fixed value of $X_2$ and varying $X_1$. This tells us that on average the fraction of the available observations which will be considered when making the prediction for a given test observation is $0.0975^2 = 0.00950625 \approx 0.01$. We can also verify this, as well as do Part 3 below, using the multivariable calculus extension for the formula to compute the average value of an integrable function $f(x_1, \dots, x_p)$ on the region $R = [a_1, b_1] \times \dots \times [a_p, b_p]$
# \begin{equation}
#     \frac{1}{\prod_{i = 1}^p (b_i - a_i)} \idotsint \limits_{R} f(x_1, \dots, x_p) \, dx_1 \cdots dx_p
# \end{equation}
# to compute the average area of the region considered when making a prediction given predictors $X_p = x_p$.
# 
# 3. As we found in Parts 1 and 2, the fraction of the available observations which will be considered when making the prediction for a given test observation with predictor values $X_p = x_p$ is $0.0975^p \approx 0.1^p$. For $p = 100$, this is $7.95 \times 10^{-102}$, or essentially zero. In other words, essentially none of the available observations will be considered when making the prediction.
# 
# 4. As we saw in Parts 1 through 3, the fraction of observations for which $X_p$ is within 10% of the range of the test observation's value for the predictor $X_p = x_p$, for all $p$ is about $10^{-p}$, which is essentially zero for large values of $p$. More generally, the [volume of a $p$-dimensional ball of fixed radius $R$](https://en.wikipedia.org/wiki/Volume_of_an_n-ball) tends to zero as $p$ approaches infinity. This means that for any conventionally intuitive and desirable notion of "near" (e.g. all predictor values within a particular percent of the range of the test observation; staying within a fixed (Euclidean) distance of the test observation) there are very few training observations "near" any given test observation. Therefore, when performing KNN using data with a large number of predictors, the $k$-"nearest" neigbors for a given test observation are highly likely to actually be quite far from that test observation, in the sense that they are not "near" using one of those notions of near mentioned above. As a result, KNN with large values of $p$ attempts to make a prediction for the test observation using neighbors which are quite unlikely to be "near" enough to the test observation for the prediction to be meaningfully accurate.
# 
# 5. Creating a $p$-dimensional hypercube centered around the test observation that contains, on average, 10% of the training observations is, in the context of this exercise, equivalent to requiring that our hypercube have a volume of 0.1. Since the volume (or more technically $p$-volume of a $p$-dimensional hypercube with side length $s$ is given by $V = s^p$, a $p$-dimensional hypercube with a volume $V$ has a side length of $s = V^{1/p}$. Thus for $p = 1$, $s = 0.1$; for $p = 2$, $s = 0.1^{1/2} \approx 0.3162$; and when $p = 100$, $s = 0.1^{1/100} \approx 0.9772$. This means that for large values of $p$, the vast majority of the volume of a $p$-dimensional hypercube is located very close to its boundary. This further emphasizes the point from Part 4 that when performing KNN using data with a large number of predictors, even the $k$-"nearest" neighbors for a given test observation will be highly likely to have predictor values vastly different from the predictor values for the test observation.

# # Conceptual Exercise 5
# 
# ## Question
# 
# We now examine the differences between LDA and QDA.
# 
# 1. If the Bayes decision boundary is linear, do we expect LDA or QDA to perform better on the training set? On the test set?
# 
# 2. If the Bayes decision boundary is non-linear, do we expect LDA or QDA to perform better on the training set? On the test set?
# 
# 3. In general, as the sample size $n$ increases, do we expect the test prediction accuracy of QDA relative to LDA to improve, decline, or be unchanged? Why?
# 
# 4. True or False: Even if the Bayes decision boundary for a given problem is linear, we will probably achieve a superior test error rate using QDA rather than LDA because QDA is flexible enough to model a linear decision boundary. Justify your answer.

# ## Answer
# 
# 1. When the Bayes decision boundary is linear, we expect QDA to have lower training error as a result of its increased flexibility over LDA. However, we would expect LDA to have lower test error since it produces a linear decision boundary, is also the shape of the Bayes decision boundary in this case. The higher variance introduced by the flexibility of QDA wouldn't be offset by the lower bias, so QDA is an overly flexible model for this situation.
# 
# 2. When the Bayes decision boundary is non-linear, we expect QDA to have both lower training error (for the same reason noted above in Part 1) and lower test error. In this situation, the linear nature of LDA results means that it won't be able to accurately approximate the non-linear Bayes decision boundary, resulting in higher training error than QDA.
# 
# 3. As the sample size $n$ increases, the we generally expect the test prediction of QDA relative to LDA to improve. This is because as the size of the training set increases, reducing model variance becomes less crucial, since no change in a single data point is likely to greatly affect the model compared to when there are a small number of training observations.
# 
# 4. False. If we have few training observations, the increased flexibility of QDA over LDA will likely lead to QDA overfitting to the random noise in the training set. This would then result in a worse test error rate when using QDA compared to the test error rate for LDA. In other words, when there are few training observations, reducing model variance is crucial, so the increased variance in QDA over LDA will not be offset by the reduction in bias, resulting in worse test performance from QDA.

# # Conceptual Exercise 6
# 
# ## Question
# 
# Suppose we collect data for a group of students in a statistics class with variables $X_1 =$ hours studied, $X_2 =$ undergrad GPA, and $Y =$ receive an A. We fit a logistic regression and produce estimated coefficients $\hat{\beta}_0 = -6, \, \hat{\beta}_1 = 0.05, \, \hat{\beta}_2 = 1$.
# 
# 1. Estimate the probability that a student who studies for 40 hours and has an undergrad GPA of 3.5 gets an A in the class.
# 
# 2. How many hours would the student in part (1) need to study to have a 50% chance of getting an A in the class?

# ## Answer
# 
# 1. We plug $X_1 = 40$ and $X_2 = 3.5$ into the fitted logistic regression model, which gives us a probability of 0.3775, or 37.75%, that a student who studies for 40 hours and has an undergrad GPA of 3.5 will get an A in the class. 
# 
# \begin{equation}
#     p(\text{Receive an A} | X) = \frac{e^{-6 + 0.05X_1 + X_2}}{1 + e^{-6 + 0.05X_1 + X_2}}
# \end{equation}
# 
# 2. Given $X_2 = 3.5$ and the target of having a 50% chance of getting an A in the class, it is easier to plug those values into the logit representation of the logistic model to solve for the number of hours the student needs to study in order to have that predicted 50% chance at an A. When we do so, we find that $X_1 = 50$ hours will give the student a predicted 50% chance at getting an A in the class.
# 
# \begin{equation}
#     -6 + 0.05X_1 + 3.5 = \log \left( \frac{0.5}{1 - 0.5} \right) \Rightarrow
#     0.05X_1 - 2.5 = 0 \Rightarrow 
#     X_1 = 50
# \end{equation}

# # Conceptual Exercise 7
# 
# ## Question
# 
# Suppose that we wish to predict whether a given stock will issue a dividend this year ("Yes" or "No") based on $X$, last year's percent profit. We examine a large number of companies and discover that the mean value of $X$ for companies that issued a dividend was $\bar{X} = 10$, while the mean for those that didn't was $\bar{X} = 0$. In addition, the variance of $X$ for these two sets of companies was $\hat{\sigma}^2 = 36$. Finally, 80% of companies issued dividends. Assuming that $X$ follows a normal distribution, predict the probability that a company will issue a dividend this year given that its percentage profit was $X = 4$ last year.
# 
# *Hint: Recall that the density function for a normal random variable is*
# 
# \begin{equation}
#     f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x - \mu)^2/2\sigma^2}.
# \end{equation}
# 
# *You will need to use Bayes' theorem.*

# ## Answer
# 
# Let $k = 1$ be the class corresponding to companies which issue a dividend and let $k = 2$ be the class corresponding to companies which don't issue a dividend. Then by the assumptions the observations (percent profits) for class $k = 1$ follow an $N(\hat{\mu}_1 = 10, \, \hat{\sigma}^2 = 36)$ distribution while those for class $k = 2$ follow an $N(\hat{\mu}_2 = 0, \, \hat{\sigma}^2 = 36)$ distribution. In addition, we have prior probabilities $\pi_1 = 0.8$ and $\pi_2 = 0.2$, since 80% of companies issued dividends. Since we wish to predict the probability that a company will issue a dividend this year given that its percentage profit was $X = 4$ last year, we plug all of these, along with the density function for a normal random variable, into Bayes' theorem using $k = 1$.
# 
# \begin{equation}
#     p_k(x) = \frac{\pi_k \frac{1}{\sigma \sqrt{2\pi}}\exp \left( -\frac{1}{2\sigma^2} (x - \mu_k)^2 \right)}{\sum_{l = 1}^K \pi_l \frac{1}{\sigma \sqrt{2\pi}}\exp \left( -\frac{1}{2\sigma^2} (x - \mu_l)^2 \right)}
# \end{equation}
# 
# For notational convenience, we write
# 
# \begin{equation}
#     f_k(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x - \mu_k)^2/2\sigma^2}.
# \end{equation}
# 
# Thus, we have
# 
# \begin{equation}
#     p_1(x = 4) = \frac{\pi_1 f_1(x)}{\pi_1 f_1(x) + \pi_2 f_2(x)}
#     = \frac{0.8 f_1(4)}{0.8 f_1(4) + 0.2 f_2(4)}
#     \approx \frac{0.8 \times 0.04033}{0.8 \times 0.04033 + 0.2 \times 0.05324}
#     \approx .75185.
# \end{equation}
# 
# In other words, there is a 75.2% chance that a company will issue a dividend this year given that its percentage profit was $X = 4$ last year.

# # Conceptual Exercise 8
# 
# ## Question
# 
# Suppose we take a data set, divide it into equally-sized training and test sets, and then try out two different classification procedures. First we use logistic regression and get an error rate of 20% on the training data and 30% on the test data. Next we use 1-nearest neighbors (i.e. KNN with $K = 1$) and get an average error rate (averaged over both test and training data sets) of 18%. Based on these results, which method should we prefer to use for classification of new observations? Why?

# ## Answer
# 
# When performing KNN with $K = 1$, the training error is always going to be zero. This is because the KNN classifier assigns each point $x_0$ to the class for which the proportion of the $K$ points in the training set closest to $x_0$ is the greatest. If $x_0$ is already in training set, then its closest neighbor is itself, so it will always be assigned to its actual class. Thus, the average error rate of 18% for KNN with $K = 1$ actually translates to a test error rate of 36%. Since this is larger than the 30% test error rate from logistic regression, we should choose logistic regression over KNN with $K = 1$ to more accurately classify new observations.

# # Conceptual Exercise 9
# 
# ## Question
# 
# This problem has to do with *odds*.
# 
# 1. On average, what fraction of people with an odds of 0.37 of defaulting on their credit card payment will in fact default?
# 
# 2. Suppose that an individual has a 16% chance of defaulting on their credit card payment. What are the odds that they will default?

# ## Answer
# 
# Recall that the odds of an event $A$ with a probability $p(A)$ occurring is $p(A)/(1 - p(A))$.
# 
# 1. Plugging in 0.37 for the odds into the odds formula and solving for the probability an individual defaults on their credit card payment, we get 0.27. In other words, on average about 27% of people with an odds of 0.37 of defaulting on their credit card payment will in fact default.
# 
# 2. Plugging in 0.16 for the probability of default into the odds formula, we get an odds of $0.16/(1 - 0.16) = .1905$ that the individual will default.
