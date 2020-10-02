#!/usr/bin/env python
# coding: utf-8

# # Conceptual Exercise 1
# 
# ## Question
# 
# For each of parts (1) through (4), indicate whether we would generally expect the performance of a flexible statistical learning method to be better or worse than an inflexible method. Justify your answer.
# 
# 1. The sample size $n$ is extremely large, and the number of predictors $p$ is small.
#     
# 2. The number of predictors $p$ is extremely large, and the number of observations $n$ is small.
#     
# 3. The relationship between the predictors and response is highly non-linear.
#     
# 4. The variance of the error terms, i.e. $\sigma^2 = \text{Var}(\varepsilon)$, is extremely high.
# 
# ## Answer
#     
# 1. Having a large sample size should reduce the error due to variance, since small changes in the data shouldn't greatly affect the set as a whole when it is used for training a model. Also, since the number of predictors is small, the computational cost of using a more flexible method isn't as much of an issue. These two factors combined mean that I would generally expect better performance from a more flexible method. However, I wouldn't want to use a method which is too flexible, since it then would pick up too much on whatever noise there is in the data coming from the irreducible error in the relationship between the predictors and the response.
#     
# 2. Having a large number of predictors makes flexible methods more computationally costly. In addition, the small number of observations means that small changes in the data could result in large changes in $\hat{f}$ if we use a flexible method, as flexible methods have a higher variance than inflexible ones. As such, in order to reduce the computational cost and also reduce the potential for error due to variance, I would expect an inflexible method to generally perform better than a flexible one in this situation.
#     
# 3. In this situation, a flexible statistical learning method would generally perform better than an inflexible one, as the increased flexibility would allow the method to better capture the non-linearity of the relationship and reduce the bias, or error that is introduced by approximating a real-life problem.
#     
# 4. Since the expected test mean-squared error can never lie below $\text{Var}(\varepsilon)$, which is the irreducible error in the relationship between the predictors and response, there isn't enough information to definitively say whether or not a flexible statistical learning method would perform better than an inflexible one without doing further analysis. In order to minimize the risk of initially overfitting the data, it would be better to start of with a less-flexible and computationally less-costly method if it would be too costly to do some kind of visualization to help with determining what kind of method to use as a starting point.

# # Conceptual Exercise 2
# 
# ## Question
# 
# Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested in inference or prediction. Finally, provide $n$ and $p$.
# 
# 1. We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry, and the CEO salary. We are interested in understanding which factors affect CEO salary.
#     
# 2. We are considering launching a new product and wish to know whether it will be a *success* or a *failure*. We collect data on 20 similar products that were previously launched. For each product, we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables.
#     
# 3. We are interested in predicting the % change in the USD/Euro exchange rate in relation to the weekly changes in the world stock markets. Hence, we collect weekly data for all of 2012. For each week we record the % change in the USD/Euro exchange rate, the % change in the US market, the % change in the British market, and the % change in the German market.
# 
# ## Answer
# 
# 1. This is a regression problem where the response variable is the CEO salary and the $p = 3$ predictors are profit, number of employees, and industry. Since we are interested in understanding which factors affect CEO salary, instead of trying to predict a CEO's salary given the the predictors, we are most interested in inference, or understanding the relationship between the predictors and the response. The data is for the top 500 firms in the US, meaning $n = 500$.
#     
# 2. This is a classification problem because we are putting products into one of two categories: *success* or *failure*. Since we want to know whether or not the new product will be a success or a failure, we are most interested in prediction. In this situation, $n = 20$ for the 20 similar products that were previously launched. There are $p = 13$ predictors: price charged for the product, marketing budget, competition price, and the ten other variables collected.
#     
# 3. This is a regression problem where the response variable is the % change in the USD/Euro exchange rate and the $p = 3$ predictors are the % change in the US market, the % change in the British market, and the % change in the German market. In this situation we are most interested in prediction. Lastly, since we collected weekly data for all of 2012, $n = 52$.

# # Conceptual Exercise 3
# 
# ## Question
# 
# We now revisit the bias-variance decomposition.
# 
# 1. Provide a sketch of typical (squared) bias, variance, training error, test error, and Bayes (or irreducible) error curves, on a single plot, as we go from less flexible statistical learning methods towards more flexible approaches. The $x$-axis should represent the amount of flexibility in the method, and the $y$ axis should represent the values for each curve. There should be five curves. Make sure to label each one.
#     
# 2. Explain why each of the five curves has the shape displayed in part (1).
# 
# ## Answer
# 
# 1. ![20191017_140739.jpg](attachment:20191017_140739.jpg)
# 
# 2. The bias generally starts off high for methods with low flexibility (unless the true $f$ is close to linear) and then decreases down to zero as flexibility increases. On the other hand, variance starts out at zero (since the least flexible approach is assuming $f$ takes a single constant value) and then increases as flexibility increases, since flexible methods are susceptible to changing considerably when any single data point changes. Training error starts out at some non-zero value, with magnitude depending on the linearity of the true $f$. As flexibility increases, it decreases down to zero. Similar to the training error, test error starts out at some non-zero value, with magnitude depending on the linearity of the true $f$. As flexibility increases, it then decreases down to a minimum (but never below the irreducible error) before increasing again. Lastly, the irreducible error is a constant non-zero value which is the absolute minimum possible test error (though that minimum need not be achieved).

# # Conceptual Exercise 4
# 
# ## Question
# 
# You will now think of some real-life applications for statistical learning.
# 
# 1. Describe three real-life approaches in which *classification* might be useful. Describe the response, as well as the predictors. Is the goal of each application inference or prediction? Explain your answer.
#     
# 2. Describe three real-life applications in which *regression* might be useful. Describe the response, as well as the predictors. Is the goal of each application inference or prediction? Explain your answer.
#     
# 3. Describe three real-life applications in which *cluster analysis* might be useful.
# 
# ## Answer
# 
# 1. These are three situations where *classification* might be useful.
#     
#     1. One situation where classification might be useful is classifying a song into a genre (e.g. pop, rock, hip-hop, country, etc.) based solely on analyzing things like the lyrics, the musical keys used, beat patterns, and other characteristics that could be determined only from analyzing the song's audio. With more data, such as info about the people who listen to the song and what other songs they listen to, there is room for further interesting analysis. In this situation, the goal is understanding the key components that would make a song fit into a given genre (what makes a song a pop song as opposed to a rock song, for example), and also what potential overlaps there might be between genres, making this a situation where the goal is inference.
#     
#     2. Another situation where classification might be useful is determining whether or not a person is at risk of defaulting on a loan. Here the response is whether or not a person will default on their loan. Some predictors could be their credit score and credit score history, their employment status, the amount of money they are borrowing, the purpose/type of loan, and whether or not the person has borrowed money in the past, and if so, whether or not they successfully paid back the loan. In this situation, we would be most interested in answering the question "Will this person be at risk of defaulting on their loan?", making this a situation where the main goal is prediction. Inference could still be a secondary goal, if we are also interested in understanding how the various predictors affect the risk of a person defaulting on their loan.
#     
#     3. A third situation where classification might be useful is determining whether or not a student at a university is at risk of failing a class or dropping out. Here the response is whether or not a student will fail a class (or drop out of school). Some predictors could be the number of units they are currently taking, their grades in the previous terms (and specifically grades in related classes if we are focusing on the "failing a class" situation), the grades and courses they took in high school, their race and their family's socioeconomic status, how often they use university-provided support services (e.g. tutoring, libraries, academic counseling, etc.), how much need-based and merit-based financial aid they receive, and whether or not they participate in work-study (and if so, how many hours they work). In this situation, the immediate goal would be prediction in order intervene for any student who is at risk of failing a class or dropping out of school. Inference would still be a secondary goal to help us understand the risk factors that lead a student to potentially fail or drop out, so then appropriate support measures or interventions could be enacted before students reach the point of requiring a more dramatic intervention.
#     
# 
# 2. These are three situations where *regression* might be useful.
#     
#     1. One situation where regression might be useful is predicting the time it would take a person to complete a marathon. Here, some predictors could be the person's recent race times, how many miles a week they run on average in the 12 weeks leading up to the marathon, the average pace at which they ran their easy miles, the weather (temperature and humidity) for the day of the marathon, and the elevation of the location for the marathon. Since most people would be interested in their predicted race time, prediction would be the main goal. Inference would still be a second goal for people who are also interested in things like how increasing or decreasing the mileage done in training might affect their performance.
#     
#     2. Another situation where regression might be useful is analyzing how long it takes a person to get their first job after graduating from college. Here, some predictors might be the person's major in college, their GPA (overall GPA and GPA in their major), the number of internships they completed before graduation, the type of degree they are graduating with (bachelor's, master's, doctoral, MD, etc), and data about the school they graduated from (ranking, public vs. private, number of students, etc.). In this situation, we'd be more interested in understanding how the different factors affect the time it takes the person to get their first job, rather than trying to make any predictions, so the main goal would be inference.
#     
#     3. A third situation where regression might be useful is predicting a student's SAT or ACT score and then understanding how some of those factors affect the score they receive. Some predictors could be the number of weeks they spent studying for the test, the number of hours per week they spent studying, the number of practice tests they took before the actual test, the scores they received on the most recent practice test, their grade level in school, their GPA, the types of study resources they used (test prep books, online prep, in-person prep classes, tutoring services/private tutoring, etc), their race, their family's socioeconomic status, and the date they are taking the test. Here prediction and inference are both interesting goals, since from a student perspective they would be interested in predicting what score they would get (though recent practice tests are generally a good predictor for that on their own), while from an educator perspective it would be helpful to understand which test prep strategies are can provide the biggest score boost. In addition, it would be important from the perspective of the test creators to understand whether or not the test is fair for students across the spectrum of racial and socioeconomic backgrounds. Thus, depending on the perspective from which you approach the situation, the main goal might change between inference and prediction
#     
# 
# 3. These are three situations where *cluster analysis* might be useful.
#     
#     1. One situation where cluster analysis might be useful would be in analyzing the different kinds of users on YouTube. Some possible variables that might be useful in the analysis include the number and types of channels they subscribe to, how long they watch each video from the channels they subscribe to, how long and how often they watch videos from channels they don't subscribe to, how much time on average they spend watching YouTube videos each day, how often they engage with videos (liking/disliking, leaving a comment, clicking on advertisements within videos). Cluster analysis could then reveal different types of YouTube users which then could be useful for YouTube in negotiating advertising deals and determining which advertisements to present to a given user. It would also be useful in providing users with suggestions for videos to watch and channels to consider looking at.
#     
#     2. Another situation where cluster analysis might be useful would be in online commerce and analyzing the differ kinds of shoppers on a site such as Amazon. Some possible variables that might be useful in the analysis include the items a shopper searches for, the items a shopper views, how often they make purchases and the monetary amount of each purchase along with the kinds of items they purchase, how often and the kinds of reviews they leave on items they purchase, how often the shopper returns items, and the shopper's shipping and billing locations.  
#     
#     3. A third situation where cluster analysis might be useful, or at least interesting, would be in analyzing the different kinds of users on a dating app, such as Tinder. Some possible variables that might be useful in the analysis might be the user's biographic data (age, gender identity, location, education, current employment), the contents of their profile, their listed dating preferences (age range, gender identity, location, education, current employment), how long they have been using the application and how often/how long they use it each day, how many profiles do they accept and reject in total and on a daily basis, the kinds of profiles they accept and reject, how often (if at all) they use premium features of the app, how often do they make a successful match, and (if available) where do those matches lead (e.g. just exchange some messages and nothing else, meet up in person, long-term relationship, etc.).

# # Conceptual Exercise 5
# 
# ## Question
# 
# What are the advantages and disadvantages of a very flexible (versus a less flexible) approach for regression or classification? Under what circumstances might a more flexible approach be preferred to a less flexible approach? When might a less flexible approach be preferred?
# 
# ## Answer
# 
# One main advantage to a very flexible statistical learning approach is the ability to generate a much wider range of possible shapes to estimate $f$. This is helpful if the relationship between the predictors and response is quite non-linear, as less-flexible methods have more restrictions to the shapes of $f$ they can generate, and therefore could generate models which are inherently far from the actual shape of $f$. Up to a point, higher flexibility also allows the model to make more accurate predictions, which is desirable if we are mainly interested in prediction. However, as flexibility increases, the interpretability of the models generated decreases. This is undesirable if we are mainly interested in inference and understanding the underlying relationship between the predictors and response. In that case, when inference is the main goal, a less flexible approach may be preferrable.

# # Conceptual Exercise 6
# 
# ## Question
# 
# Describe the differences between a parametric and a non-parametric statistical learning approach. What are the advantages of a parametric approach to regression or classification (as opposed to a non-parametric approach)? What are its disadvantages?
# 
# ## Answer
# 
# A parametric approach to statistical learning involves first making an assumption about the functional form or shape of the relationship between the predictors and the response. For example, one common starting assumption for a parametric approach is assuming a linear relationship
# $$
# f(X) = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_pX_p.
# $$
# The advantage of making this assumption in the parametric approach is that it reduces the problem of estimating $f$ to the problem of estimating parameters. This is usually much easier than trying to fit an entirely arbitrary function. However, the downside to this approach lies in choice made when choosing the initial model. The choice involves making simplifying assumptions which then results in a model which usually means the model will not match the true form of $f$. Moreover, if we choose a model that is too far from the true $f$, we'll have intrinsically bad estimates.
# 
# A non-parametric approach, on the other hand, doesn't involve making any initial assumptions about the shape of the relationship between the predictors and the response. Instead, the approach involves estimating $f$ in a way that gets as close to the data points as possible while maintaiing some minimum notion of smoothness. The main advantage to avoiding the initial assumption about the form of $f$ is that non-parametric methods can potentially more accurately fit a wider range of possible shapes of $f$. However, since no assumptions are made about the form of $f$, a very large number of observations is required to obtain an accurate estimate for $f$.

# # Conceptual Exercise 7
# 
# ## Question
# 
# The table below provides a training data set containing six observations, three predictors, and one qualitative response variable.
# 
# <table>
# <tr>
#     <th>Obs.</th>
#     <th>$X_1$</th>
#     <th>$X_2$</th>
#     <th>$X_3$</th>
#     <th>$Y$</th>
# </tr>
# <tr>
#     <td>1</td>
#     <td>0</td>
#     <td>3</td>
#     <td>0</td>
#     <td>Red</td>
# </tr>
# <tr>
#     <td>2</td>
#     <td>2</td>
#     <td>0</td>
#     <td>0</td>
#     <td>Red</td>
# </tr>
# <tr>
#     <td>3</td>
#     <td>0</td>
#     <td>1</td>
#     <td>3</td>
#     <td>Red</td>
# </tr>
# <tr>
#     <td>4</td>
#     <td>0</td>
#     <td>1</td>
#     <td>2</td>
#     <td>Green</td>
# </tr>
# <tr>
#     <td>5</td>
#     <td>-1</td>
#     <td>0</td>
#     <td>1</td>
#     <td>Green</td>
# </tr>
# <tr>
#     <td>6</td>
#     <td>1</td>
#     <td>1</td>
#     <td>1</td>
#     <td>Red</td>
# </tr>
# </table>
# 
# Suppose we wish to use this data to make a prediction for $Y$ when $X_1 = X_2 = X_3 = 0$ using $K$-nearest neighbors.
# 
# 1. Compute the Euclidean distance between each observation and the test point $X_1 = X_2 = X_3 = 0$.
#     
# 2. What is our prediction with $K = 1$? Why?
#     
# 3. What is our prediction with $K = 3$? Why?
#     
# 4. If the Bayes decision boundary in this problem is highly non-linear, then would we expect the *best* value for $K$ to be large or small? Why?
# 
# ## Answer
# 
# 1. First recall that the Euclidean distance between two points $X_i = (x_i, y_i, z_i)$ for $i = 1, 2$ is given by the formula
# $$
# d(X_1, X_2) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2}.
# $$
# In our case, since the test point is the origin $X_0 = (0, 0, 0)$, this formula simplifies to
# $$
# d_i = d(X_0, X_i) = \sqrt{x_i^2 + y_i^2 + z_i^2}.
# $$
# Therefore, we have the following Euclidean distances between each observation and the test point.
#     1. $d_1 = \sqrt{0^2 + 3^2 + 0^2} = 3$
#     1. $d_2 = \sqrt{2^2 + 0^2 + 0^2} = 2$
#     1. $d_3 = \sqrt{0^2 + 1^2 + 3^2} = \sqrt{10}$
#     1. $d_4 = \sqrt{0^2 + 1^2 + 2^2} = \sqrt{5}$
#     1. $d_5 = \sqrt{(-1)^2 + 0^2 + 1^2} = \sqrt{2}$
#     1. $d_6 = \sqrt{1^2 + 1^2 + 1^2} = \sqrt{3}$
# 
# 2. Using $K = 1$, we predict the color of the test point using only the closest neighbor, which is $X_5$ at a distance $\sqrt{2}$ away. Since that point is green, we predict that the test point will be green.
# 
# 3. Using $K = 3$, we predict the color of the test point using the three closest neighbors. They are $X_5$ (distance $\sqrt{2}$), $X_6$ (distance $\sqrt{3}$), and $X_4$ (distance $\sqrt{5}$). Since $X_4$ and $X_5$ are both green, while $X_6$ is red, we predict that the test point will be green. 
# 
# 4. If the Bayes decision boundary in this problem is highly non-linear, then we would expect the *best* value for $K$ to be small. This is because as $K$ increases, the method of $K$-nearest neighbors becomes less flexible and produces a decision boundary which is more linear. A small value for $K$, on the other hand, results in increased flexibility and a decision boundary which is more non-linear, which in this situation would be closer to the gold standard Bayes decision boundary.
