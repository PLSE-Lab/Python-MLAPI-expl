#!/usr/bin/env python
# coding: utf-8

# # Conceptual Exercise 1
# 
# ## Question
# 
# Describe the null hypotheses to which the p-values given in Table 3.4 correspond. Explain what conclusions you can draw based on these p-values. Your explanation should be phrased in terms of `sales`, `TV`, `radio`, and `newspaper`, rather than in terms of the coefficients of the model.
# 
# <table>
# <tr>
#     <th></th>
#     <th>Coefficient</th>
#     <th>Std. Error</th>
#     <th>t-statistic</th>
#     <th>p-value</th>
# </tr>
# <tr>
#     <td>Intercept</td>
#     <td>2.939</td>
#     <td>0.3119</td>
#     <td>9.42</td>
#     <td>&lt; 0.001</td>
# </tr>
# <tr>
#     <td>TV</td>
#     <td>0.046</td>
#     <td>0.0014</td>
#     <td>32.81</td>
#     <td>&lt; 0.001</td>
# </tr>
# <tr>
#     <td>Radio</td>
#     <td>0.189</td>
#     <td>0.0086</td>
#     <td>21.89</td>
#     <td>&lt; 0.001</td>
# </tr>
# <tr>
#     <td>Newspaper</td>
#     <td>-0.001</td>
#     <td>0.0059</td>
#     <td>-0.18</td>
#     <td>0.8599</td>
# </table>

# ## Answer
# 
# Generally speaking, the p-values given in the table report the partial effect of adding that variable to the model. In other words, they provide information about whether each individual predictor is related to the response, after adjusting for the other predictors. The p-value for `TV` corresponds to a null hypothesis that the coefficient for `TV` is zero, or in other words that TV advertising has no relationship with sales, in the presence of radio and newspaper advertising. Since the p-value for `TV` is essentially zero, there is strong evidence that TV advertising is related to sales. Things are similar for `radio`. Its p-value corresponds to a null hypothesis that the coefficient for `radio` is zero, or in other words that radio advertising has no relationship with sales, in the presence of TV and newspaper advertising. Once again, since the p-value is essentially zero, there is strong evidence that radio advertising is related to sales. Lastly, the p-value for `newspaper` corresponds to a null hypothesis that the coefficient for `newspaper` is zero, or in other words that newspaper advertising has no relationship with sales, in the presence of TV and radio advertising. Here, since the p-value for `newspaper` is quite large at 0.8599, there is no evidence that newspaper advertising is related to sales, in the presence of TV and radio advertising.

# # Conceptual Exercise 2
# 
# ## Question
# 
# Carefully explain the differences between the KNN classifier and KNN regression methods.

# ## Answer
# 
# The KNN classifier and KNN regression methods are similar in the fact that both use information from the $K$ neighbors which are closest to the prediction point $x_0$ in order to make a prediction. Where they differ is in types of responses that each is used for, with KNN classification being used with categorical response variables to assign the prediction point to a class, and KNN regression being used with quantitative response variables to estimate the numerical value of the response. This is illustrated in the formulas underlying each method. In KNN regression, $f(x_0)$ is estimated by taking the average of all the training responses in $\mathcal{N}_0$, the set of the $K$ training observations which are nearest to $x_0$.
# 
# \begin{equation}
# \hat{f}(x_0) = \frac{1}{K} \sum_{x_i \in \mathcal{N}_0} y_i
# \end{equation}
# 
# In KNN classification, on the other hand, $x_0$ is assigned to the class $Y = j$ which has the largest proportion of points in $\mathcal{N}_0$ being members of that class. 
# 
# \begin{equation}
# \text{Pr}(Y = j | X = x_0) = 
#     \frac{1}{K} \sum_{x_i \in \mathcal{N}_0} I(y_i = j)
# \end{equation}
# 
# Recall that here $I(y_i = j)$ is the indicator function which is equal to 1 if $x_i$ is in class $j$ and 0 otherwise.

# # Conceptual Exercise 3
# 
# ## Question
# 
# Suppose we have a data set with five predictors, $X_1 =$ GPA, $X_2 =$ IQ, $X_3 =$ Gender (1 for Female and 0 for Male), $X_4 =$ Interaction between GPA and IQ ($X_4 = X_1X_2$), and $X_5 =$ Interaction between GPA and Gender ($X_5 = X_1X_3$). The response is starting salary after graduation (in thousands of dollars). Suppose we use least squares to fit the model, and get $\hat{\beta}_0 = 50$, $\hat{\beta}_1 = 20$, $\hat{\beta}_2 = 0.07$, $\hat{\beta}_3 = 35$, $\hat{\beta}_4 = 0.01$, $\hat{\beta}_5 = -10$. In other words, the least squares regression model is
# 
# \begin{equation}
# \hat{y} = 50 + 20X_1 + 0.07X_2 + 35X_3 + 0.01X_4 - 10X_5
# \end{equation}
# 
# 1. Which answer is correct, and why?
#     1. For a fixed value of IQ and GPA, males earn more on average than females.
#     2. For a fixed value of IQ and GPA, females earn more on average than males.
#     3. For a fixed value of IQ and GPA, males earn more on average than females provided that the GPA is high enough.
#     4. For a fixed value of IQ and GPA, females earn more on average than males provided that the GPA is high enough.   
# 
# 2. Predict the salary of a female with IQ of 110 and a GPA of 4.0.
# 
# 3. True or false: Since the coefficient for the GPA/IQ interaction term is very small, there is very little evidence of an interaction effect. Justify your answer.

# ## Answer
# 
# 1. The third answer (C) is correct. To see why this is the case, suppose we have fixed values for GPA and IQ: $X_1 = a$ and $X_2 = b$. Plugging into the least squares regression model and rearranging, we have
# \begin{equation}
# \hat{y} = 50 + 20a + 0.07b + 0.01ab + (35 - 10a)X_3.
# \end{equation}
# For lower GPA values ($a < 3.5$), the coefficient of $X_3$ is positive, meaning that for a fixed value of IQ and a fixed GPA value less than 3.5, females earn more on average than males, as $X_3 = 1$ for females and is zero for males. For GPA values above 3.5, the coefficient of $X_3$ becomes negative, meaning that for a fixed value of IQ and a fixed GPA value greater than 3.5, males earn more on average. Hence, the model says that for a fixed value of IQ and GPA, males earn more on average than females, provided that the GPA is high enough.
# 
# 2. Plugging into the least squares regression model, we have 
# \begin{equation}
# \hat{y} = 50 + 20(4.0) + 0.07(110) + 35(1) + 0.01(4.0)(110) - 10(4.0)(1) = 137.1.
# \end{equation}
# This gives us a predicted salary of \\$137,100 for a female with an IQ of 110 and a GPA of 4.0.
# 
# 3. False. The value of the coefficient for an interaction term doesn't provide evidence for or against the possibility of an interaction effect. In order to actually make a statement about evidence of an interaction effect, or lack thereof, we would need to either be given or compute a p-value for the coefficient of the interaction term.

# # Conceptual Exercise 4
# 
# ## Question
# 
# I collect a set of data ($n$ = 100 observations) containing a single predictor and a quantitative response. I then fit a linear regression model to the data, as well as a separate cubic regression, i.e. $Y = \beta_0 + \beta_1X + \beta_2X^2 + \beta_3X^3 + \epsilon$.
# 
# 1. Suppose that the true relationship between $X$ and $Y$ is linear, i.e. $Y = \beta_0 + \beta_1X + \epsilon$. Consider the training residual sum of squares (RSS) for the linear regression, and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.
# 
# 2. Answer (1) using test rather than training RSS.
# 
# 3. Suppose that the true relationship between $X$ and $Y$ is not linear, but we don't know how far it is from linear. Consider the training RSS for the linear regression, and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.
# 
# 4. Answer (3) using test rather than training RSS.

# ## Answer
# 
# 1. I would expect the training RSS for the cubic regression to be lower than that of the linear regression, since cubic regression has a higher level of flexibility. As the flexibility of a model increases, the training mean squared error, which is equal to the product of the training RSS and the number of observations $n$, decreases monotonically. This is regardless of the nature of the true relationship between $X$ and $Y$.
# 
# 2. I would expect the test RSS for the linear regression to be lower than that of the cubic regression. Test RSS depends on the variance and squared bias of the model, as well as by the variance of $\epsilon$. Since the true relationship in this situation is linear, linear regression has lower bias than cubic regression. Moreover, linear regression, as a less-flexible model, always has less variance than cubic regression, since variance is a measure of how much the model would change if we estimated it using a different training set. Thus, since the variance of $\epsilon$ is a constant and the linear model has lower bias and variance compared to the cubic model, linear regression will result in a lower test RSS compared to cubic regression.
# 
# 3. Similar to part (1), I would expect the training RSS for the cubic regression to be lower than that of the linear regression. Again this is due to the fact that cubic regression has a higher level of flexibility, and increased flexibility results in lower training mean squared error, and hence lower training RSS.
# 
# 4. In this situation, cubic regression will likely have less bias than linear regression, since the true relationship is not linear. However, since we don't know how far it is from linear, we don't know the size of the difference in bias between the two models. Moreover, as noted in part (2), cubic regression has higher variance than linear regression, so the accuracy of a cubic regression model depends more on the training data. Since we don't know anything about the training data, we don't know how well it actually represents the relationshp between the predictor and the response. Because we don't know the size of the difference in bias between the two models, and also since cubic regression has higher variance and we don't know anything further about the training data, we don't have enough information to compare the values of the test RSS between the two models in this situation.

# # Conceptual Exercise 5
# 
# ## Question
# 
# Consider the fitted values that result from performing linear regression without an intercept. In this setting, the $i$th fitted value takes the form
# 
# \begin{equation}
# \hat{y}_i = x_i\hat{\beta},
# \end{equation}
# 
# where
# 
# \begin{equation}
# \hat{\beta} = \left( \sum_{j = 1}^n x_jy_j \right) / 
#               \left( \sum_{j = i}^n x_j^2 \right).
# \end{equation}
# 
# Show that we can write
# 
# \begin{equation}
# \hat{y}_i = \sum_{j = 1}^n a_jy_j.
# \end{equation}
# 
# What is $a_j$?
# 
# Note: We interpret this result by saying that the fitted values from linear regression are ***linear combinations*** of the response values.

# ## Answer
# 
# Combining the formula for the $i$th fitted value with the formula for $\hat{\beta}$, we get
# 
# \begin{equation}
# \hat{y}_i = \frac{x_i}{\sum_{j = 1}^n x_j^2} \sum_{j = 1}^n x_jy_j.
# \end{equation}
# 
# We can simplify this by setting
# 
# \begin{equation}
# c_i = \frac{x_i}{\sum_{j = 1}^n x_j^2}.
# \end{equation}
# 
# As $c_i$ does not vary with the value of $j$, we can move it inside the some and combine with the $x_j$ factor into a single factor $a_j = c_ix_j$.
# 
# \begin{equation}
# \hat{y}_i = c_i \sum_{j = 1}^n x_jy_j = \sum_{j = 1}^n a_jy_j.
# \end{equation}
# 
# To be extra concrete, we note the full formula for $a_j$ in the linear combination for the fitted value $\hat{y}_i$.
# 
# \begin{equation}
# a_j = \frac{x_ix_j}{\sum_{j = 1}^n x_j^2}.
# \end{equation}

# # Conceptual Exercise 6
# 
# ## Question
# 
# Using the formulas for the parameter estimates below, argue that in the case of simple linear regression, the least squares line always passes through the point $(\bar{x}, \bar{y})$. Recall that $\bar{x}$ is the average of the $x$ values and $\bar{y}$ is the average of the $y$ values.
# 
# \begin{align}
# \hat{\beta}_1 &= \frac{\sum_{i = 1}^n (x_i - \bar{x})(y_i - \bar{y})}
#                 {\sum_{i = 1}^n (x_i - \bar{x})^2}\\
# \hat{\beta}_0 &= \bar{y} - \hat{\beta}_1\bar{x}
# \end{align}

# ## Answer
# 
# In the case of simple linear regression, the model takes the form $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1x$, where $\hat{\beta_0}$ and $\hat{\beta}_1$ have the above formulas. Plugging in the formula for $\hat{\beta}_0$, we end up with
# 
# \begin{equation}
# \hat{y} = \bar{y} - \hat{\beta}_1(x - \bar{x}).
# \end{equation}
# 
# When we plug $x = \bar{x}$ into the regression equation, terms cancel and leaves us with $\hat{y} = \bar{y}$. In other words, the least squares line always passes through the point $(\bar{x}, \bar{y})$.

# # Conceptual Exercise 7
# 
# ## Question
# 
# It is claimed in the text that in the case of simple linear regression of $Y$ onto $X$, the $R^2$ statistic
# 
# \begin{equation}
# R^2 = 1 - \frac{\sum_{i = 1}^n (y_i - \hat{y}_i)^2}
#                {\sum_{i = 1}^n (y_i - \bar{y}_i)^2}
# \end{equation}
# 
# is equal to the square of the correlation between $X$ and $Y$
# 
# \begin{equation}
# \text{Cor}(X, Y) = \frac{\sum_{i = 1}^n (x_i - \bar{x})(y_i - \bar{y})}
#                         {\sqrt{\sum_{i = 1}^n (x_i - \bar{x})^2}
#                         \sqrt{\sum_{i = 1}^n (y_i - \bar{y})^2}}.
# \end{equation}
# 
# Prove that this is the case. For simplicity, you may assume that $\bar{x} = \bar{y} = 0$.

# ## Answer
# 
# We start with the simplifying assumption that $\bar{x} = \bar{y} = 0$. With this simplification, which in essence is a translation to put us in the situation of linear regression without an intercept, as discussed in Exercise 5 above. Thus, our regression formula simplifies to become
# 
# \begin{equation}
# \hat{y_i} = x_i\hat{\beta}, \,
# \hat{\beta} = \left( \sum_{j = 1}^n x_jy_j \right) / 
#               \left( \sum_{j = i}^n x_j^2 \right).
# \end{equation}
# 
# Our formulas for $R^2$ and $\text{Cor}(X, Y)$ simplify as well.
# 
# \begin{align}
# R^2 &= \frac{\sum_{i = 1}^n y_i^2 - \sum_{i = 1}^n (y_i - \hat{y}_i)^2}
#                {\sum_{i = 1}^n y_i^2} \\
# \text{Cor}(X, Y) &= \frac{\sum_{i = 1}^n x_iy_i}
#                         {\sqrt{\sum_{i = 1}^n x_i^2}
#                         \sqrt{\sum_{i = 1}^n y_i^2}}              
# \end{align}
# 
# To start with, we consider $\text{Cor}(X, Y)^2$.
# 
# \begin{equation}
# \text{Cor}(X, Y)^2 = \frac{\left( \sum_{i = 1}^n x_iy_i \right)^2}
#                     {\left( \sum_{i = i}^n x_i^2 \right) \left( \sum_{i = i}^n y_i^2 \right)}
# \end{equation}
# 
# Next, we focus on working with the numerator in the expression for $R^2$. First, observe that we can expand the terms in the second sum.
# 
# \begin{equation}
# (y_i - \hat{y}_i)^2 = y_i^2 - 2y_i\hat{y}_i + \hat{y}_i^2
# \end{equation}
# 
# This allows us to make a cancellation in the numerator.
# 
# \begin{equation}
# \sum_{i = 1}^n y_i^2 - \sum_{i = 1}^n (y_i - \hat{y}_i)^2 = 
# \sum_{i = 1}^n (2y_i\hat{y}_i - \hat{y}_i^2)
# \end{equation}
# 
# Now, we substitute the expression for $\hat{y}_i$ into the simplified numerator.
# 
# \begin{equation}
# \sum_{i = 1}^n (2y_i\hat{y}_i - \hat{y}_i^2) = 
# \sum_{i = 1}^n \left( 2x_iy_i \frac{a}{b} - \frac{x_i^2a^2}{b^2} \right)
# \end{equation}
# 
# For typing convenience (and hopefully improved readability), we are using
# \begin{equation}
# a = \sum_{j = 1}^n x_jy_j, \, b = \sum_{j = i}^n x_j^2.
# \end{equation}
# 
# After factoring out a $b$ from each term in the sum, we then split it into two pieces.
# 
# \begin{equation}
# \sum_{i = 1}^n \left( 2x_iy_i \frac{a}{b} - \frac{x_i^2a^2}{b^2} \right) = 
# \frac{1}{b} \sum_{i = 1}^n \left( 2x_iy_ia - \frac{x_i^2a^2}{b} \right) =
# \frac{1}{b} \sum_{i = 1}^n 2x_iy_ia - \frac{1}{b}\sum_{i = 1}^n \frac{x_i^2a^2}{b}
# \end{equation}
# 
# Notice that in the second sum there is a sum of $x_i^2$ in both the numerator and denominator (as $b$ is a sum of $x_i^2$), so those cancel out.
# 
# \begin{equation}
# \frac{1}{b} \sum_{i = 1}^n \frac{x_i^2a}{b} = 
# \frac{a^2}{b} = 
# \frac{1}{b} \left( \sum_{j = 1}^n x_jy_j \right)^2
# \end{equation}
# 
# Next we look at the first sum in the numerator.
# 
# \begin{equation}
# \frac{1}{b} \sum_{i = 1}^n 2x_iy_ia = 
# \frac{1}{b} \sum_{i = 1}^n \sum_{j = 1}^n 2x_iy_i(x_jy_j) = 
# \frac{2}{b} \left( \sum_{i = 1}^n x_iy_i \right)^2
# \end{equation}
# 
# Once we have worked with both sums in the numerator, we can recombine them.
# 
# \begin{equation}
# \frac{1}{b} \sum_{i = 1}^n 2x_iy_ia - \frac{1}{b}\sum_{i = 1}^n \frac{x_i^2a^2}{b} = 
# \frac{1}{b} \left( \sum_{i = 1}^n x_iy_i \right)^2
# \end{equation}
# 
# Putting this back into the expression for $R^2$, we complete the proof.
# 
# \begin{equation}
# R^2 = \frac{\sum_{i = 1}^n y_i^2 - \sum_{i = 1}^n (y_i - \hat{y}_i)^2}
#                {\sum_{i = 1}^n y_i^2} =  
# \frac{\frac{1}{b} \left( \sum_{i = 1}^n x_iy_i \right)^2}
#                {\sum_{i = 1}^n y_i^2} = 
# \frac{\left( \sum_{i = 1}^n x_iy_i \right)^2}
#                     {\left( \sum_{i = i}^n x_i^2 \right) \left( \sum_{i = i}^n y_i^2 \right)} =
# \text{Cor}(X, Y)^2
# \end{equation}
