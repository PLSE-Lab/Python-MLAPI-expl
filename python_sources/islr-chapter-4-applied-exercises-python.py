#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import standard Python data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Import classes from scikit-learn for logistic regression, LDA, QDA, and KNN classification
# Import convenience function for computing confusion matrices 
# Import OneHotEncoder and StandardScaler for data pre-processing
# Import Pipeline, ColumnTransformer to encapsulate pre-processing heterogenous data and fitting
# into a single estimator
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load StatsModels API
# Note that if we wish to use R-style formulas, then we would use the StatsModels Formula API
import statsmodels.api as sm
import statsmodels.formula.api as smf


# # Applied Exercise 1
# 
# **This question should be answered using the `Weekly` data set, which is part of the `ISLR` package. This data is similar in nature to the `Smarket` data from this chapter's lab, except that it contains 1,089 weekly returns for 21 years, from the beginning of 1990 to the end of 2010.**

# In[ ]:


weekly_filepath = "../input/islr-weekly/Weekly.csv"
weekly = pd.read_csv(weekly_filepath)
weekly.head()


# Since this exercise will involve evaluating the statistical significance of predictors in the logistic regression model, we will use StatsModels for all computations that involve logistic regression. For all of the other models (LDA, QDA, KNN) we will use scikit-learn.

# ## Part 1
# **Produce some numerical and graphical summaries of the `Weekly` data. Do there appear to be any patterns?**

# In[ ]:


weekly.describe()


# In[ ]:


weekly["Direction"].value_counts()


# In[ ]:


sns.pairplot(weekly, hue = "Direction");


# From the initial summary statistics and scatterplot matrix, there don't appear to be any obvious patterns aside from the fact the volume of shares traded each week has grown quite a lot from 1990 to 2010. Looking more closely at the scatterplot of just volume over time, we can see that the number of shares traded each week has grown exponentially over the 21 years covered by the data.

# In[ ]:


fig = plt.figure(figsize = (10, 8))
ax = plt.axes()
ax.scatter(x = weekly.index, y = weekly["Volume"], alpha = 0.5)
ax.set(xlabel = "Week", ylabel = "Shares traded (in billions)");


# In[ ]:


weekly.corr()


# Lastly, we look at the matrix of correlations between the variables. Looking at the last row, we can see that each of the lag variables is only correlated very weakly with today's returns. The sole substantial value of 0.842, between `Volume` and `Year`, aligns with the strong correlation we saw in the above scatterplot.

# ## Part 2
# **Use the full data set to perform a logistic regression with `Direction` as the response and the five lag variables plus `Volume` as predictors. Use the summary function to print the results. Do any of the predictors appear to be statistically significant? If so, which ones?**

# In[ ]:


# Using the Logit class from StatsModels
# First encode response numerically
endog = (weekly["Direction"] == "Up").astype("int64")
exog = sm.add_constant(weekly.drop(columns = ["Direction", "Year", "Today"]))
logit_mod = sm.Logit(endog, exog)
logit_res = logit_mod.fit()
print(logit_res.summary())


# In[ ]:


pd.DataFrame({"Estimate": logit_res.params, "Std. Error": logit_res.bse, "z value": logit_res.tvalues,
             "Pr(>|z|)": logit_res.pvalues})


# The only statistically significant predictor is `Lag2`, with a p-value of 0.0296 providing evidence at the 5% significance level to reject the null hypothesis that it is not related to the response `Direction`. None of the other predictors are statistically significant, though `Lag1` is somewhat near the border of being significant at the 10% level, with a p-value of 0.1181.

# ## Part 3
# **Compute the confusion matrix and overall fraction of correct predictions. Explain what the confusion matrix is telling you about the types of mistakes made by logistic regression.**

# In order to produce a confusion matrix that helps us better visualize the incorrect predictions, we use seaborn to convert the confusion matrix into a [heatmap](http://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap).

# In[ ]:


mat = pd.DataFrame(logit_res.pred_table(), columns = ["Down", "Up"], index = ["Down", "Up"])
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"})
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


logit_preds = pd.Series(logit_res.predict()).apply(lambda x: "Up" if (x > 0.5) else "Down")
(logit_preds == weekly["Direction"]).mean()


# As we can see in the confusion matrix, the logistic regression model using the five lag variables along with `Volume` as the predictors, and a prediction threshold of 0.5, correctly predicted 54 down weeks out of a total of 484 actual down weeks and 557 up days out of a total of 605 actual up weeks. This means that the model correctly predicted the direction for 611 weeks out of the 1089 for an accuracy of 0.5611. While this seems to be better than random chance, it is important to note that the model was trained on the entire data set, so 0.5611 is the *training* accuracy. Moreover, a naive strategy of simply saying that every week will be an up week would have resulted in 605 correctly predicted weeks out of 1089, which is a very similar level of overall accuracy.
# 
# To look a little closer at the confusion matrix, let's assume that our goal is to correctly predict when the market will go up. In this case, up weeks will be considered as positive ($+$) and down weeks as negative ($-$). Having set this convention, we can now consider four important quantities associated with the confusion matrix: true positive rate (i.e. *sensitivity* or *recall*), false positive rate, positive predictive value (i.e. *precision*), and negative predictive value. The true positive rate is the number of correctly predicted positives divided by the overall number of positives -- the number of correctly predicted up weeks (557) over the total number of up weeks (605) for a value of $557/605 \approx 0.92$ for this model. While this is a pretty high value, which is good, the false positive rate -- the number of incorrectly predicted positives (weeks incorrectly predicted to be up weeks = 430 weeks) divided by the overall number of negatives (the total number of down weeks = 484 weeks) -- is comparably high at $430/484 \approx 0.888$, which might be quite bad depending on our sensitivity to losing money on an incorrectly predicted down week. Next is the positive predictive value, which is the number of true positives divided by the total number of predicted positives; in our case this is $557/987 \approx 0.564$. This is better than chance, but as already noted we would have a comparable positive predictive value if we just predicted that every week would be an up week. Lastly is the negative predictive value, which is the number of true negatives divided by the total number of predicted negatives; in our case this is $54/102 \approx 0.529$. 

# ## Part 4
# **Now fit the logistic regression model using a training data period from 1990 to 2008, with `Lag2` as the only predictor. Compute the confusion matrix and the overall fraction of correct predictions for the held out data (that is, the data from 2009 and 2010).**

# In[ ]:


train_mask = (weekly["Year"] < 2009)


# In[ ]:


# Using the Logit class from StatsModels
# First encode response numerically
train_endog = (weekly.loc[train_mask, "Direction"] == "Up").astype("int64")
train_exog = sm.add_constant(weekly.loc[train_mask, "Lag2"])
logit_mod = sm.Logit(train_endog, train_exog)
logit_res = logit_mod.fit()
print(logit_res.summary())


# In[ ]:


test_exog = sm.add_constant(weekly.loc[~train_mask, "Lag2"])
test_endog = weekly.loc[~train_mask, "Direction"]
logit_test_probs = logit_res.predict(test_exog)
logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Up" if (x > 0.5) else "Down")


# In[ ]:


mat = pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), columns = ["Down", "Up"], index = ["Down", "Up"])
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"})
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


(logit_test_preds == weekly.loc[~train_mask, "Direction"]).mean()


# In[ ]:


(weekly.loc[~train_mask, "Direction"] == "Up").mean()


# After fitting a logistic regression model on the data from 1990 through 2008 using only `Lag2` as the predictor, the model correctly predicted the market direction for 62.5% of the weeks in the held-out data (the data from 2009 and 2010). While this is better than chance, it still is less than a 10% improvement over naively predicting that every week will be an up week. Continuing with the convention from Part 3 that an up week is a positive result, the true positive rate is $56/61 \approx 0.918$, while the false positive rate is $34/43 \approx 0.791$. In addition, the positive predictive value is $56/90 \approx 0.622$ and the negative predictive value is $9/14 \approx 0.643$.

# ## Part 5
# **Repeat Part 4 using LDA.**

# In[ ]:


# First separate out the training and test sets
X_train = weekly.loc[train_mask, "Lag2"].to_frame()
y_train = weekly.loc[train_mask, "Direction"]
X_test = weekly.loc[~train_mask, "Lag2"].to_frame()
y_test = weekly.loc[~train_mask, "Direction"]

# Fit the LDA model using the training set
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)


# In[ ]:


y_pred = lda_clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = lda_clf.classes_, yticklabels = lda_clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


lda_clf.score(X_test, y_test)


# After performing linear discriminant analysis on the data from 1990 through 2008 using only `Lag2` as the predictor, we ended up with an identical confusion matrix to the one from Part 4 with the logistic regression model. As we saw in Part 4, the model correctly predicted the market direction for 62.5% of the weeks in the held-out data (the data from 2009 and 2010). While this is better than chance, it still is less than a 10% improvement over naively predicting that every week will be an up week. Continuing with the convention from Part 3 that an up week is a positive result, the true positive rate is $56/61 \approx 0.918$, while the false positive rate is $34/43 \approx 0.791$. In addition, the positive predictive value is $56/90 \approx 0.622$ and the negative predictive value is $9/14 \approx 0.643$.

# ## Part 6
# **Repeat Part 4 using QDA.**

# In[ ]:


# First separate out the training and test sets
X_train = weekly.loc[train_mask, "Lag2"].to_frame()
y_train = weekly.loc[train_mask, "Direction"]
X_test = weekly.loc[~train_mask, "Lag2"].to_frame()
y_test = weekly.loc[~train_mask, "Direction"]

# Fit the QDA model using the training set
qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)


# In[ ]:


y_pred = qda_clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


qda_clf.score(X_test, y_test)


# After performing quadratic discriminant analysis on the data from 1990 through 2008 using only `Lag2` as the predictor, we ended up with a model that ended up predicting that every week in 2009 and 2010 would be an up week. The model correctly predicted the market direction for 58.7% of the weeks in the held-out data (the data from 2009 and 2010). While this is better than chance, it is the same result one would get from naively predicting that every week will be an up week, so to better compare the performance of this model to the logistic regression and linear discriminant analysis models testing on a larger test set would be necessary. Continuing with the convention from Part 3 that an up week is a positive result, the true positive rate is $61/61 = 1$, while the false positive rate is $43/43 \approx 1$. In addition, the positive predictive value is $61/104 \approx 0.587$ and since there were no negative predictions discussing the negative predictive value is meaningless.

# ## Part 7
# **Repeat Part 4 using KNN with $K = 1$.**

# In[ ]:


# First separate out the training and test sets
X_train = weekly.loc[train_mask, "Lag2"].to_frame()
y_train = weekly.loc[train_mask, "Direction"]
X_test = weekly.loc[~train_mask, "Lag2"].to_frame()
y_test = weekly.loc[~train_mask, "Direction"]

# Set NumPy random seed for consistency and reproducibility of our results
np.random.seed(312)

# Fit the QDA model using the training set
knn_1_clf = KNeighborsClassifier(n_neighbors = 1)
knn_1_clf.fit(X_train, y_train)


# In[ ]:


y_pred = knn_1_clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


knn_1_clf.score(X_test, y_test)


# After performing $k$-nearest neighbors classification with $k = 1$ on the data from 1990 through 2008 using only `Lag2` as the predictor, the model correctly predicted the market direction for 49% of the weeks in the held-out data (the data from 2009 and 2010). While this essentially only as good as picking the direction randomly, it had worse performance than naively predicting that every week will be an up week. Continuing with the convention from Part 3 that an up week is a positive result, the true positive rate is $30/61 \approx 0.492$, while the false positive rate is $22/43 \approx 0.512$. In addition, the positive predictive value is $30/52 \approx 0.577$ and the negative predictive value is $21/52 \approx 0.404$.

# ## Part 8
# **Which of these methods appears to provide the best results on this data?**

# If we are only considering overall prediction accuracy, it appears that logistic regression and linear discriminant analysis were equally good as the models that performed the best on this data. Quadratic discriminant analysis came in third place, and $k$-nearest neighbors with $k = 1$ a distant fourth. As noted above in Part 7, $k$-nearest neighbors didn't perform any better than randomly guessing, and in fact performed worse than naively predicting every week would be an up week. One thing that I would be cautious of, though, is the fact that aside from KNN, the other three models very strongly preferred to predict up weeks, and therefore had high false positive rates. This might be okay for an investor who is comfortable with taking more risks and has can afford losing money to down weeks that were incorrectly predicted to be up weeks, but a risk-averse investor would probably want to use a less aggressive model, or at least require a higher probability threshold than 50% before accepting a prediction of an up week.

# ## Part 9
# **Experiment with different combinations of predictors, including possible transformations and interactions, for each of the methods. Report the variables, method, and associated confusion matrix that appears to provide the best results on the held out data. Note that you should also experiment with values for $K$ in the KNN classifier.**

# The combination of predictors I will try out is a weighted average of the lag variables where the recent lag values are weighted more heavily than the ones further in the past. More specifically, I will try out giving `Lag1` a weight of 40%, `Lag2` a weight of 35%, `Lag3` a weight of 15%, and `Lag4` and `Lag5` each weights of 5%. Note that I don't have a strong reason for these particular values of the weights aside from the intuitive assumption that recent weeks might be more-related to the direction of the current week than weeks that are further in the past.

# In[ ]:


weighted_lag_avg = 0.4*weekly["Lag1"] + 0.35*weekly["Lag2"] + 0.15*weekly["Lag3"] + 0.05*weekly["Lag4"] + 0.05*weekly["Lag5"]
weekly["weighted_lag_avg"] = weighted_lag_avg
weekly.head()


# In[ ]:


weekly[["Today", "weighted_lag_avg"]].corr()


# Computing the correlation between this weighted average and the value of the current week's return, we see that there only a very weak correlation between the two quantities. It is smaller in magnitude than the correlations between `Today` and the first the lag variables individually. That seems to suggest that this weighting might not be too useful, but I will still try out each of the classification methods using this transformation of the predictors. I will start out with logistic regression.

# In[ ]:


# Using the Logit class from StatsModels
# First encode response numerically
train_endog = (weekly.loc[train_mask, "Direction"] == "Up").astype("int64")
train_exog = sm.add_constant(weekly.loc[train_mask, "weighted_lag_avg"])
logit_mod = sm.Logit(train_endog, train_exog)
logit_res = logit_mod.fit()
print(logit_res.summary())


# In[ ]:


test_exog = sm.add_constant(weekly.loc[~train_mask, "weighted_lag_avg"])
test_endog = weekly.loc[~train_mask, "Direction"]
logit_test_probs = logit_res.predict(test_exog)
logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Up" if (x > 0.5) else "Down")


# In[ ]:


mat = pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), columns = ["Down", "Up"], index = ["Down", "Up"])
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"})
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


(logit_test_preds == weekly.loc[~train_mask, "Direction"]).mean()


# As we can see, the results with logistic regression aren't particularly encouraing, as with a prediction threshold of 50% this method was equivalent to always predicting that the market would go up when evaluated on the test set. In addition, the p-value for the coefficient of `weighted_lag_avg` is 0.599, which means that there isn't evidence to say that it is statistically significant. Thinking a little more about the idea of using a weighted average, it makes sense that the performance with logistic regression won't provide an improvement over what we did in Part 4, since the weighted average is still a linear combination of the variables. We already saw that in a logistic regression model, `Lag2` was the only statistically significant coefficient, and even then it is borderline at the 5% significance level, so the weighted average includes variables which we already had reason to believe weren't particularly helpful in making a strong model. I'll still try out the remaining methods with `weighted_lag_avg` before trying out one other combination of the predictors. Next up is linear discriminant analysis.

# In[ ]:


# First separate out the training and test sets
X_train = weekly.loc[train_mask, "weighted_lag_avg"].to_frame()
y_train = weekly.loc[train_mask, "Direction"]
X_test = weekly.loc[~train_mask, "weighted_lag_avg"].to_frame()
y_test = weekly.loc[~train_mask, "Direction"]

# Fit the LDA model using the training set
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)


# In[ ]:


y_pred = lda_clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = lda_clf.classes_, yticklabels = lda_clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


lda_clf.score(X_test, y_test)


# Linear discriminant analysis has the same performance as logistic regression. This is reasonable since the two methods often perform similarly. Now we'll consider quadratic discriminant analysis.

# In[ ]:


# First separate out the training and test sets
X_train = weekly.loc[train_mask, "weighted_lag_avg"].to_frame()
y_train = weekly.loc[train_mask, "Direction"]
X_test = weekly.loc[~train_mask, "weighted_lag_avg"].to_frame()
y_test = weekly.loc[~train_mask, "Direction"]

# Fit the QDA model using the training set
qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)


# In[ ]:


y_pred = qda_clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


qda_clf.score(X_test, y_test)


# In[ ]:


qda_clf.predict_proba(X_test).max()


# Quadratic discriminant analysis also performed the same, but when checking the maximum value in the n-dimensional array of posterior probabilities, I noticed that the maximum probability from QDA is 0.943. This means that it could be meaningful to try out a stricter threshold for predicting an up week. I will try out a threshold of 60%.

# In[ ]:


qda_predict_proba = qda_clf.predict_proba(X_test)
# Columns in the 2-dimensional array qda_predict_proba correspond to posterior probabilities
# for the classes, as found in qda_clf.classes_
# In this case, qda_clf.classes_ is the list ["Down", "Up"], so the index 1 column contains the 
# posterior probabilities for the class "Up"
y_pred_60 = pd.Series(qda_predict_proba[:, 1]).apply(lambda x: "Up" if (x > 0.6) else "Down")
mat = confusion_matrix(y_test, y_pred_60)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


(y_pred_60.values == y_test.values).mean()


# With a somewhat stricter probability threshold of 60% for predicting an up week, we see that our overall prediction accuracy goes down to 48.1%. However, our false positive rate is now $6/43 \approx 0.14$, which is a big improvement over the false positive rate of 1 from Part 6. In addition, our positive predictive value is $13/19 \approx 0.684$, which is also an improvement over the value of 0.587 from Part 6. This suggests that if we are risk-averse investors who only want to invest our money when there is a pretty good chance of the market going up, then QDA with our weighted average of the lag variables and a stricter probability threshold could be promising.  Last up is $k$-nearest neighbors.

# In[ ]:


# First separate out the training and test sets
X_train = weekly.loc[train_mask, "weighted_lag_avg"].to_frame()
y_train = weekly.loc[train_mask, "Direction"]
X_test = weekly.loc[~train_mask, "weighted_lag_avg"].to_frame()
y_test = weekly.loc[~train_mask, "Direction"]

# Set NumPy random seed for consistency and reproducibility of our results
np.random.seed(312)

# Fit the QDA model using the training set
knn_1_clf = KNeighborsClassifier(n_neighbors = 1)
knn_1_clf.fit(X_train, y_train)


# In[ ]:


y_pred = knn_1_clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


knn_1_clf.score(X_test, y_test)


# With $k = 1$, KNN performs even worse than random guessing when it comes to overall prediction accuracy, and the true positive rate (0.541), false positive rate (0.674), and positive predictive rate (0.532) aren't encouraging when compared to the result from Part 8. Before moving on, let's try out two more values for $k$: $k = 3$ and $k = 5$.

# In[ ]:


# Set NumPy random seed for consistency and reproducibility of our results
np.random.seed(312)

# Fit the QDA model using the training set
knn_3_clf = KNeighborsClassifier(n_neighbors = 3)
knn_3_clf.fit(X_train, y_train)


# In[ ]:


y_pred = knn_3_clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


knn_3_clf.score(X_test, y_test)


# In[ ]:


# Set NumPy random seed for consistency and reproducibility of our results
np.random.seed(312)

# Fit the QDA model using the training set
knn_5_clf = KNeighborsClassifier(n_neighbors = 5)
knn_5_clf.fit(X_train, y_train)


# In[ ]:


y_pred = knn_5_clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = qda_clf.classes_, yticklabels = qda_clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


knn_5_clf.score(X_test, y_test)


# Even when we increase the value of $k$, the results are largely the same. While it would be best to compare these different models using additional tools, such as computing ROC curves and AUC scores for each model, I'll save that for Chapter 9 when we discuss how to produce them in R more in depth. For now, it looks like as far as using `weighted_lag_avg` to make predictions goes, the most-promising model is QDA with a prediction threshold of 60%. 
# 
# If I have time to revisit this exercise in the future, I think it would be intersting to try out a different weighting strategy, where each lag variable is weighted based on the number of shares during the corresponding week of trading.

# # Applied Exercise 2
# 
# **In this problem, you will develop a model to predict whether a given car gets high or low gas mileage based on the `Auto` data set.**

# In[ ]:


auto_filepath = "../input/ISLR-Auto/Auto.csv"
auto = pd.read_csv(auto_filepath, na_values = ["?"]).dropna()
auto.head()


# Note that the `origin` column actually contains categorical data, even though it is coded using integers. In order to make my life a little easier for performing regression, I'm going replace the values in that column with their meanings.

# In[ ]:


auto["origin"] = auto["origin"].map({1: "American", 2: "European", 3: "Japanese"})
auto.head()


# For this exercise, I will use scikit-learn for all of the parts out of convenience.

# ## Part 1
# **Create a binary variable, `mpg_med`, that contains "Above" if `mpg` contains a value above its median, and "Below" if `mpg` contains a value below its median. You can compute the median using the `Series.median()` function.**

# In[ ]:


mpg_med = (auto["mpg"] > auto["mpg"].median()).map({False: "Below", True: "Above"})
auto["mpg_med"] = mpg_med
auto.head()


# ## Part 2
# 
# **Explore the data graphically in order to investigate the association between `mpg_med` and the other features. Which of the other features seem most likely to be useful in predicting `mpg_med`? Scatterplots and boxplots may be useful tools to answer this question. Describe your findings.**

# In[ ]:


fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 12))
sns.boxplot(x = "mpg_med", y = "cylinders", data = auto, ax = axes[0, 0])
sns.boxplot(x = "mpg_med", y = "displacement", data = auto, ax = axes[0, 1])
sns.boxplot(x = "mpg_med", y = "horsepower", data = auto, ax = axes[0, 2])
sns.boxplot(x = "mpg_med", y = "weight", data = auto, ax = axes[1, 0])
sns.boxplot(x = "mpg_med", y = "acceleration", data = auto, ax = axes[1, 1])
sns.boxplot(x = "mpg_med", y = "year", data = auto, ax = axes[1, 2])
fig.suptitle("Boxplots for cars with above and below median mpg", size = "xx-large", y = 0.925);


# First, I made boxplots to compare the overall distributions for each of the quantitative variables between cars with above-median mpg and those with below median-mpg. Starting in the upper-left corner, we see that aside from some outliers, the the majority of cars with above-median mpg have four-cylinder engines. Moving to the upper-middle pair of boxplots, at least 75% of the cars with above-median mpg have smaller engines than 75% of the cars with below-median mpg. This is also true for horsepower (upper-right pair of boxplots) and weight (lower-left pair of boxplots). We also see in the final two pairs of boxplots, for acceleration and manufacture year, that while there are definitely differences between cars with above-median mpg and those with below-median mpg, the differences aren't as strong compared to the first four predictors. In particular, there is a lot more overlap in both time to reach 60mpg and manufacture year between the two categories of cars, whereas for the other predictors there is almost no overlap for the boxplots between the two categories. This suggests that `cylinders`, `displacement`, `horsepower`, and `weight` will be the most useful in predicting `mpg_med`.

# In[ ]:


fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 12))
sns.distplot(auto.loc[auto["mpg_med"] == "Below", "cylinders"], ax = axes[0, 0])
sns.distplot(auto.loc[auto["mpg_med"] == "Above", "cylinders"], ax = axes[0, 0], color = "orange")
sns.distplot(auto.loc[auto["mpg_med"] == "Below", "displacement"], ax = axes[0, 1])
sns.distplot(auto.loc[auto["mpg_med"] == "Above", "displacement"], ax = axes[0, 1], color = "orange")
sns.distplot(auto.loc[auto["mpg_med"] == "Below", "horsepower"], ax = axes[0, 2])
sns.distplot(auto.loc[auto["mpg_med"] == "Above", "horsepower"], ax = axes[0, 2], color = "orange")
sns.distplot(auto.loc[auto["mpg_med"] == "Below", "weight"], ax = axes[1, 0])
sns.distplot(auto.loc[auto["mpg_med"] == "Above", "weight"], ax = axes[1, 0], color = "orange")
sns.distplot(auto.loc[auto["mpg_med"] == "Below", "acceleration"], ax = axes[1, 1])
sns.distplot(auto.loc[auto["mpg_med"] == "Above", "acceleration"], ax = axes[1, 1], color = "orange")
sns.distplot(auto.loc[auto["mpg_med"] == "Below", "year"], ax = axes[1, 2])
sns.distplot(auto.loc[auto["mpg_med"] == "Above", "year"], ax = axes[1, 2], color = "orange")
fig.suptitle("Histograms and KDE plots for cars with above (orange) and below (blue) median mpg",
             size = "xx-large", y = 0.925);


# Looking at histograms overlaid with kernel density estimates we are better able to visualize the extent to which the distributions of the cars with above-median fuel economy overlap with the distributions of the cars with below-median fuel economy for each quantitative variable. This provides further evidence to suggest that `horsepower` and `weight` will be useful in predicting `mpg_med`, with decently clear separations between the distribution peaks between above-median and below-median cars. The plot involving `cylinders` seems to indicate that that variable wouldn't be as helpful on on its own. The small number of possibilities for the number of engine cylinders (3, 4, 5, 6, or 8) results in a lot of overlap, though there is a very prononunced peak that indicates a large number of cars with above-median fuel efficiency have four-cylinder engines. The significant overlap between groups for the acceleration plot is further evidence to suggest that `acceleration` will not be a variable that is helpful in predicting `mpg_med`.

# In[ ]:


fig, ax = plt.subplots(figsize = (10, 8))
sns.scatterplot(x = "year", y = "mpg", data = auto, ax = ax)
ax.axhline(y = auto["mpg"].median(), color = "orange", linewidth = 3);


# Next, the above scatterplot of `mpg` vs `year` also shows that the newer cars in the data set tend to be more fuel efficient. Therefore, while manufacture year might not be as useful as the other four quantitative variables, that stood out (`cylinders`, `displacement`, `horsepower` and `weight`), it still seems worth including.

# In[ ]:


fig, ax = plt.subplots(figsize = (10, 8))
sns.violinplot(x = "origin", y = "mpg", data = auto, ax = ax)
sns.swarmplot(x = "origin", y = "mpg", data = auto, ax = ax, color = ".25")
ax.axhline(y = auto["mpg"].median(), color = "red", linewidth = 3);


# Lastly, when looking at a violinplot overlaid with a swarmplot that compares the mpg values for each car, categorized by country of origin, we see that there is a clear difference between American cars, which tend to have below-median fuel efficiency, and European and Japanese cars, which tend to have above-median fuel efficiency. Thus, it seems that `origin` will also be useful in predicting `mpg_med`. 
# 
# In conclusion, for the remaining parts I will use all of the predictors except for `acceleration` and `name` to in my models for trying to predict `mpg_med`. Also, I will exclude `mpg` because that was directly used to create the classification label.

# ## Part 3
# **Split the data into a training set and a test set.**

# To split my data into a training set and a test set, I will use the `train_test_split()` function to do a 75%-25% split. In other words, I will use 75% of the data (294 observations) for the training set and the remaining 25% (98 observations) for the test set.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(auto, auto["mpg_med"], test_size = 0.25, random_state = 312)


# ## Part 4
# 
# **Perform LDA on the training data in order to predict `mpg_med` using the variables that seemed most associated with `mpg_med` in Part 2. What is the test error of the model obtained?**

# Before we start fitting models and estimating test errors, we make a note about the overall process we will use for model preprocessing and fitting. In order to encapsulate the preprocessing and fitting into a single estimator, we will make use of [pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline). The steps in our pipeline will be as follows.
# 
# 1. Preprocessing
#     1. Use the [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder) to encode the categorical column `origin`.
#     2. When performing KNN or logistic regression with regularization, use the [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) to standardize the numerical columns to have mean zero and variance one.
#     3. Since we have heterogenous data types, we then combine the preprocessing for the categorical columns and numeric columns into a [ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#column-transformer). This page is an [example of how to use the ColumnTransformer class](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html).
#     4. If we are not standardizing the numerical columns, we pass them through without doing anything.
# 2. Combine the preprocessor with our choice of model estimator class (e.g. LogisticRegression, LinearDiscrminantAnalysis, etc.) into a single pipeline.

# In[ ]:


categorical_features = ["origin"]
categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])
numerical_features = ["cylinders", "displacement", "horsepower", "weight", "year"]
preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),
                                 ("num", "passthrough", numerical_features)])
clf = Pipeline([("preprocessor", preprocessor), ("classifier", LinearDiscriminantAnalysis())])


# In[ ]:


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test, y_test)


# When using linear discriminant analysis to predict `mpg_med` using `cylinders`, `displacement`, `horsepower`, `weight`, `year`, and `origin`, we had an overall test error of 10.2%. One thing of note is that this method only mis-classified one of the cars that actually had above-median fuel economy.

# In[ ]:


# Exclude the origin and year columns
categorical_features = []
categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])
numerical_features = ["cylinders", "displacement", "horsepower", "weight"]
preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),
                                 ("num", "passthrough", numerical_features)])
clf = Pipeline([("preprocessor", preprocessor), ("classifier", LinearDiscriminantAnalysis())])


# In[ ]:


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test, y_test)


# If we exclude the `origin` and `year` variables, our overall test error increases very slightly to 11.2%.

# ## Part 5
# 
# **Perform QDA on the training data in order to predict `mpg_med` using the variables that seemed most associated with `mpg_med` in Part 2. What is the test error of the model obtained?**

# In[ ]:


categorical_features = ["origin"]
categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])
numerical_features = ["cylinders", "displacement", "horsepower", "weight", "year"]
preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),
                                 ("num", "passthrough", numerical_features)])
clf = Pipeline([("preprocessor", preprocessor), ("classifier", QuadraticDiscriminantAnalysis())])


# In[ ]:


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test, y_test)


# When using quadratic discriminant analysis to predict `mpg_med` using `cylinders`, `displacement`, `horsepower`, `weight`, `year`, and `origin`, we had an overall test error of 9.18%, which is slight decrease compared to linear discriminant analysis using all of those predictors.

# In[ ]:


# Exclude the origin and year columns
categorical_features = []
categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])
numerical_features = ["cylinders", "displacement", "horsepower", "weight"]
preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),
                                 ("num", "passthrough", numerical_features)])
clf = Pipeline([("preprocessor", preprocessor), ("classifier", QuadraticDiscriminantAnalysis())])


# In[ ]:


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test, y_test)


# If we exclude the `origin` and `year` variables, our overall test error slightly increases up to 10.2%. This is the same performance as linear discriminant analysis using all of the original predictors, though this model has a different confusion matrix. In particular, using the convention that above-median mpg is the positive class label, this model has a lower false positive rate ($8/46 \approx 0.174$ versus $9/46 \approx 0.196$), a lower true positive rate ($50/52 \approx 0.962$ versus $51/52 \approx 0.981$), a higher positive predictive value ($50/58 \approx 0.862$ versus $51/60 = 0.85$), and a lower negative predictive value ($38/40 = 0.95$ versus $37/38 \approx 0.974$).

# ## Part 6
# **Perform logistic regression on the training data in order to predict `mpg_med` using the variables that seemed most associated with `mpg_med` in Part 2. What is the test error of the model obtained?**

# In[ ]:


categorical_features = ["origin"]
categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])
numerical_features = ["cylinders", "displacement", "horsepower", "weight", "year"]
preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),
                                 ("num", "passthrough", numerical_features)])
clf = Pipeline([("preprocessor", preprocessor), 
                ("classifier", LogisticRegression(penalty = "none", solver = "lbfgs"))])


# In[ ]:


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test, y_test)


# When using logistic regression to predict `mpg_med` using `cylinders`, `displacement`, `horsepower`, `weight`, `year`, and `origin`, we had an overall test error of 10.2%, which is comparable to the test errors we have obtained thus far with the other methods.

# In[ ]:


# Exclude the origin and year columns
categorical_features = []
categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])
numerical_features = ["cylinders", "displacement", "horsepower", "weight"]
preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),
                                 ("num", "passthrough", numerical_features)])
clf = Pipeline([("preprocessor", preprocessor), 
                ("classifier", LogisticRegression(penalty = "none", solver = "lbfgs", max_iter = 500))])


# In[ ]:


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test, y_test)


# If we exclude the `origin` and `year` variables, our overall test error increases very slightly, with two additional mis-classified predictions, to 12.24%.

# ## Part 7
# **Perform KNN on the training data, with several values of $K$, in order to predict `mpg_med`. Use only the variables that seemed most associated with `mpg_med` in Part 2. What test errors did you obtain? Which value of $K$ seems to perform the best on this data set?**

# I will try out $k = 1, 3, 5, \dots, 19$ when using $k$-nearest neighbors. As I did in the previous parts, I will first use `cylinders`, `displacement`, `horsepower`, `weight`, and `year` as the predictors. I will, however, exclude `origin` as it is a qualitative variable. I could convert it back to a numerical encoding, but then I would need to be careful to make sure each value is equidistant from the others. I also need to be careful of the different scales used to measure the various predictors, since variables such as `weight` are on the order of thousands of pounds while `displacement` is on the order of tens or hundreds of cubic inches. In order to put all of the quantitative variables on a comparable scale, I will standardize them so that all of them have mean zero and standard deviation one. As mentioned already in Part 4, I will perform the standardization using StandardScaler as part of a pipeline.

# In[ ]:


numerical_features = ["cylinders", "displacement", "horsepower", "weight", "year"]
numerical_transformer = Pipeline([("standardize", StandardScaler())])
preprocessor = ColumnTransformer([("num", numerical_transformer, numerical_features)])

np.random.seed(312)
k_vals = list(range(1, 21, 2))
knn_errors = {}
confusion_matrices = {}
for k in k_vals:
    clf = Pipeline([("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors = k))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrices[k] = confusion_matrix(y_test, y_pred)
    knn_errors[k] = 1 - clf.score(X_test, y_test)
pd.Series(knn_errors)


# In[ ]:


mat = confusion_matrices[pd.Series(knn_errors).idxmin()]
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# Here we see that $k = 1$ has the lowest test error, with a value of 8.16%, though $k = 3, 5, 13, 15, 17, 19$ all had the same test error value as well. Inspecting the confusion matrix for $k = 1$, which I stored in the dictionary `confusion_matrices`, we also observe that most of the errors were false positives -- that is cars the model classified as having above-median fuel efficiency which actually were in the below-median group.
# 
# Next, I will exclude the `year` variable as I did in the previous parts to see how the test errors change.

# In[ ]:


numerical_features = ["cylinders", "displacement", "horsepower", "weight"]
numerical_transformer = Pipeline([("standardize", StandardScaler())])
preprocessor = ColumnTransformer([("num", numerical_transformer, numerical_features)])

np.random.seed(312)
k_vals = list(range(1, 21, 2))
knn_errors = {}
confusion_matrices = {}
for k in k_vals:
    clf = Pipeline([("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors = k))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrices[k] = confusion_matrix(y_test, y_pred)
    knn_errors[k] = 1 - clf.score(X_test, y_test)
pd.Series(knn_errors)


# In[ ]:


mat = confusion_matrices[pd.Series(knn_errors).idxmin()]
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf["classifier"].classes_, yticklabels = clf["classifier"].classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# When I excluded the `year` variable, the model generally performed a little worse, especially for lower values of $k$. In this case, the lowest test error was with $k = 9$, where one additional false positive brought the error up to 9.18%.
# 
# In conclusion, using $k$-nearest neighbors with $k = 1$ and `cylinders`, `displacement`, `horsepower`, `weight`, and `year` as the predictors provided the lowest overall error on our held-out test set. We should be careful and gather more evidence by using cross-validation in order to better evaluate this model's accuracy.

# # Applied Exercise 3
# 
# **This problem involves writing functions.**

# ## Part 1
# **Write a function, `Power()`, that prints out the result of raising 2 to the 3rd power. In other words, your function should compute $2^3$ and print out the results.**
# 
# ***Hint: Recall that `x**a` raises `x` to the power `a`. Use the `print()` function to output the result.***

# In[ ]:


def Power():
    print(2**3)


# In[ ]:


Power()


# ## Part 2
# **Create a new function, `Power2()` that allows you to pass *any* two numbers, `x` and `a`, and prints out the value of `x^a`. You can do this by beginning your function with the line**
# 
# ```
# > def Power2(x, a):
# ```
# 
# **You should be able to call your function by entering, for instance,**
# 
# ```
# > Power2(3, 8)
# ```
# 
# **in a notebook cell. This should output the value $3^8$, namely 6,561.**

# In[ ]:


def Power2(x, a):
    print(x**a)


# In[ ]:


Power2(3, 8)


# ## Part 3
# **Using the `Power2()` function that you just wrote, compute $10^3$, $8^{17}$, and $131^3$.**

# In[ ]:


Power2(10, 3)


# In[ ]:


Power2(8, 17)


# In[ ]:


Power2(131, 3)


# ## Part 4
# **Now create a new function, `Power3()`, that actually *returns* the result `x^a` as an `R` object, rather than simply printing it to the screen. That is, if you store the value `x^a` in an object called `result` within your function, then you can simply `return()` this result using the following line:**
# 
# ```
# return result
# ```
# 
# **The line above should be the last line in your function, before the `}` symbol.**

# In[ ]:


def Power3(x, a):
    return x**a


# In[ ]:


result = Power3(5, 3)


# In[ ]:


result


# ## Part 5
# **Now using the `Power3()` function, create a plot of $f(x) = x^2$. The $x$-axis should display a range of integers from 1 to 10, and the $y$-axis should display $x^2$. Label the axes appropriately, and use an appropriate title for the figure. Consider displaying either the $x$-axis, the $y$-axis, or both axes on the log-scale. You can do this by using `Axes.semilogx()`, `Axes.semilogy()`, or `Axes.loglog()` instead of the usual `Axes.plot()` function in matplotlib.**

# In[ ]:


x = list(range(1, 11))
y = [Power3(i, 2) for i in x]
fig, ax = plt.subplots(figsize = (10, 8))
ax.plot(x, y, "bo")
ax.set(xlabel = "x", ylabel = "f(x)", title = "Plot of x vs x^2");


# In[ ]:


x = list(range(1, 11))
y = [Power3(i, 2) for i in x]
fig, ax = plt.subplots(figsize = (10, 8))
ax.semilogy(x, y, "bo")
ax.set(xlabel = "x", ylabel = "f(x)", title = "Log-scale plot of x vs x^2");


# ## Part 6
# **Create a function, `PlotPower()` that allows you to create a plot of `x` against `x^a` for a fixed `a` and for a range of values of `x`. For instance, if you call**
# 
# ```
# > PlotPower(np.arange(1, 11), 3)
# ```
# 
# **then a plot should be created with an $x$-axis taking on values $1, 2, \dots, 10$, and a $y$-axis taking on values $1^3, 2^3, \dots, 10^3$.**

# In[ ]:


def PlotPower(x, a, x_scale = None, y_scale = None):
    """
    Assumes x is array-like, a is a float
    If given, assumes x_scale, y_scale are strings that can be passed to the
    matplotlib Axes.set_xscale() and Axes.set_yscale() functions
    """
    y = [Power3(i, a) for i in x]
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.plot(x, y, "bo")
    if x_scale is not None:
        ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale(y_scale)


# In[ ]:


PlotPower(np.arange(1, 11), 3)


# I added the `x_scale` and`y_scale` arguments so the user can specify if they want to use an alternate scale on either or both axes.

# In[ ]:


PlotPower(np.arange(1, 11), 3, x_scale = "log", y_scale = "log")


# # Applied Exercise 4
# **Using the `Boston` data set, fit classification models in order to predict whether a given suburb has a crime rate above or below the median. Explore logistic regression, LDA, and KNN models using various subsets of the predictors. Describe your findings.**

# To have consistency with my results for when I do these exercises in Python, I'll use the corrected Boston housing data set instead of the one that is part of the `MASS` library.

# In[ ]:


boston_filepath = "../input/corrected-boston-housing/boston_corrected.csv"
index_cols = ["TOWN", "TRACT"]
data_cols = ["TOWN", "TRACT", "CMEDV", "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
            "PTRATIO", "B", "LSTAT"]
boston = pd.read_csv(boston_filepath, index_col = index_cols, usecols = data_cols)
boston.head()


# To start, we create a binary variable, `crim_med`, which is "Above" if `CRIM` contains a value above the median and "Below" otherwise.

# In[ ]:


crim_med = (boston["CRIM"] > boston["CRIM"].median()).map({False: "Below", True: "Above"})
boston["crim_med"] = crim_med
boston.head()


# Next we explore the data both numerically, by looking at the matrix of correlations, and graphically, by looking at boxplots to investigate the association between `crim_med` and the other features. First, we look at the matrix of correlations.

# In[ ]:


boston.corr()["CRIM"]


# We see that there are some moderate to moderately strong correlations between crime rate and the other variables. The most correlated variables are `RAD` (correlation of 0.626), a measure of accessibility to radial highways, and `TAX` (correlation of 0.583), the property tax rate in USD per \$10,000. In fact, all of the variables aside from `CHAS`, an indicator variable with value 1 if a town borders the Charles River, have correlation values with a magnitude of at least 0.2. In addition, as we saw in Applied Exercise 8 from Chapter 3, every predictor aside from `CHAS` had a statistically significant association with `CRIM` when we performed univariate linear regressions. While we should first explore the data graphically, there are two possible subsets of predictors that immeidately jump out as candidates for use in producing models to predict whether or not a given suburb has a per capita crime rate above the median. First would be to use all of the predictors aside from `CHAS` and second would be to use the predictors with a correlation magnitude of at least 0.3 (`CMEDV`, `INDUS`, `NOX`, `AGE`, `DIS`, `RAD`, `TAX`, `B`, and `LSTAT`). We will make models using both subsets of predictors to compare the predictive strength of the various models. Before making further choices regarding which predictors to use when fitting models, we will look at boxplots to explore the data graphically. We start with boxplots of each predictor, aside from `CHAS`, comparing the suburbs with above-median crime rate and those with below-median crime rate.

# In[ ]:


fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 12))
sns.boxplot(x = "crim_med", y = "CMEDV", data = boston, ax = axes[0, 0])
axes[0, 0].set_ylabel("Median home value ($1000)")
sns.boxplot(x = "crim_med", y = "ZN", data = boston, ax = axes[0, 1])
axes[0, 1].set_ylabel("Proportion of land zoned for 25000+ sq ft lots")
sns.boxplot(x = "crim_med", y = "INDUS", data = boston, ax = axes[0, 2])
axes[0, 2].set_ylabel("Proportion of non-retail business acres")
sns.boxplot(x = "crim_med", y = "NOX", data = boston, ax = axes[1, 0])
axes[1, 0].set_ylabel("Nitric oxides concentration (parts per 10 million)")
sns.boxplot(x = "crim_med", y = "RM", data = boston, ax = axes[1, 1])
axes[1, 1].set_ylabel("Average rooms per home")
sns.boxplot(x = "crim_med", y = "AGE", data = boston, ax = axes[1, 2])
axes[1, 2].set_ylabel("Proportion of homes built before 1940")
fig.suptitle("Boxplots for towns with above and below median crime rate", size = "xx-large", y = 0.925);


# In[ ]:


fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 12))
sns.boxplot(x = "crim_med", y = "DIS", data = boston, ax = axes[0, 0])
axes[0, 0].set_ylabel("Weighted distance to Boston employment centers")
sns.boxplot(x = "crim_med", y = "RAD", data = boston, ax = axes[0, 1])
axes[0, 1].set_ylabel("Index of accessibility to radial highways")
sns.boxplot(x = "crim_med", y = "TAX", data = boston, ax = axes[0, 2])
axes[0, 2].set_ylabel("Property tax rate (USD per $10000)")
sns.boxplot(x = "crim_med", y = "PTRATIO", data = boston, ax = axes[1, 0])
axes[1, 0].set_ylabel("Pupil-teacher ratio")
sns.boxplot(x = "crim_med", y = "B", data = boston, ax = axes[1, 1])
axes[1, 1].set_ylabel("1000*(Proportion of black residents - 0.63)^2")
sns.boxplot(x = "crim_med", y = "LSTAT", data = boston, ax = axes[1, 2])
axes[1, 2].set_ylabel("Proportion lower socioeconomic status population")
fig.suptitle("Boxplots for towns with above and below median crime rate", size = "xx-large", y = 0.925);


# The twelve boxplots we produced provide further evidence to suggest that the predictors which are likely to be most useful in predicting `crim.med` are those having a correlation magnitude of at least 0.3 with `CRIM` (`CMEDV`, `INDUS`, `NOX`, `AGE`, `DIS`, `RAD`, `TAX`, `B`, and `LSTAT`). Those are the variables which show the most separation in distributions between the suburbs with above-median crime rates and those with below-median crime rates. In particular, for all of those variables except `CMEDV`, it appears that the medians for the suburbs with above-median crime rates are well clear of the upper or lower quartile (depending on the particular variable) of the values for suburbs with below-median crime rates. Therefore, we will proceed and produce models using two different subsets of predictors: all of them aside from `CHAS`, and only the ones which are at least moderately correlated with `CRIM` (`CMEDV`, `INDUS`, `NOX`, `AGE`, `DIS`, `RAD`, `TAX`, `B`, and `LSTAT`).
# 
# Before training any models, we split our data in to a training set and a test set using the 75%-25% split.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(boston, boston["crim_med"], test_size = 0.25, 
                                                    random_state = 312)


# We also make two arrays of columns to help with selecting the particular subsets of columns we wish to use: all of the predictors aside from `CHAS`, and those which are at least moderately correlated with `CRIM` (absolute value of correlation at least 0.3).

# In[ ]:


not_chas = boston.columns.drop(["CHAS", "CRIM", "crim_med"])
moderate_corr = boston.corr().loc[(boston.corr()["CRIM"].abs() > 0.3), "CRIM"].index.drop("CRIM")


# Now that we have split the data in to a training set and a test set, we fit classification models to predict whether or not a given suburb has an above-median crime rate. Note that for this exercise we will stick with the default prediction threshold of 50% for convenience. If we were trying to make predictions for an application, then we should consider other prediction thresholds depending on our tolerance for false positives versus false negatives. Here we use the convention that an above-median crime rate is the positive category and below-median the negative category. For each model, we will first use all of the predictors before using the ones which are at least moderately correlated with `CRIM`. The first classification model we use is logistic regression.

# In[ ]:


clf = LogisticRegression(penalty = "none", solver = "lbfgs", max_iter = 10000)
clf.fit(X_train[not_chas], y_train)
y_pred = clf.predict(X_test[not_chas])
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf.classes_, yticklabels = clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test[not_chas], y_test)


# When using logistic regression to predict `crim_med` using all of the predictors aside from `CHAS`, we had an overall test error of 7.87%. Our positive predictive value was $50/52 \approx 0.962$ and our true positive rate was $50/58 \approx 0.862$. So far that seems like a pretty good error rate for the first model we tried, so lets continue with the smaller subset of predictors and then move on to other models to see if we can improve the performance at all.

# In[ ]:


clf = LogisticRegression(penalty = "none", solver = "lbfgs", max_iter = 10000)
clf.fit(X_train[moderate_corr], y_train)
y_pred = clf.predict(X_test[moderate_corr])
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf.classes_, yticklabels = clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test[moderate_corr], y_test)


# The logistic regression model using fewer predictors (`CMEDV`, `INDUS`, `NOX`, `AGE`, `DIS`, `RAD`, `TAX`, `B`, and `LSTAT`) performed slightly worse, with an overall test error of 8.66%. It had a positive predictive value of $51/55 \approx 0.927$, which is a slight decrease from the model with more predictors. However, the true positive rate is slightly better, at $51/58 \approx 0.879$. This is due to the fact that source of error for the model with fewer predictors is mis-classifying suburbs which actually have below-median crime rates as being in the above-median category. We now move on to linear discriminant analysis.

# In[ ]:


clf = LinearDiscriminantAnalysis()
clf.fit(X_train[not_chas], y_train)
y_pred = clf.predict(X_test[not_chas])
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf.classes_, yticklabels = clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test[not_chas], y_test)


# Compared to the logistic regression model using all of the predictors aside from `CHAS`, linear discriminant analysis performed much worse, with an overal test error of 15.7%. That is almost double the overall test error from logistic regression. This performance discrepancy suggests that the assumption underlying LDA, that the observations are drawn from a Gaussian distribution where both classes share a common covariance matrix, are not met for at least some of these predictors. Here, the main source of the error comes from mis-classifying suburbs which actually have above-median crime rates as being in the below-median category.

# In[ ]:


clf = LinearDiscriminantAnalysis()
clf.fit(X_train[moderate_corr], y_train)
y_pred = clf.predict(X_test[moderate_corr])
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf.classes_, yticklabels = clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test[moderate_corr], y_test)


# Similar to the model which used all of the predictors aside from `CHAS`, linear discriminant analysis using fewer predictors (`CMEDV`, `INDUS`, `NOX`, `AGE`, `DIS`, `RAD`, `TAX`, `B`, and `LSTAT`) performed worse than the logistic regression model using the same predictors. It did, however, perform a bit better than the LDA model which used more predictors. So far logistic regression appears to give the best test error rate. We continue on to quadratic discriminant analysis.

# In[ ]:


clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train[not_chas], y_train)
y_pred = clf.predict(X_test[not_chas])
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf.classes_, yticklabels = clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test[not_chas], y_test)


# Quadratic discriminant analysis with all of the predictors aside from `CHAS` had an overall test error of 8.66%, which is comparable to that of logistic regression using the predictors `CMEDV`, `INDUS`, `NOX`, `AGE`, `DIS`, `RAD`, `TAX`, `B`, and `LSTAT`. This is a solid improvement over linear discriminant analysis using the same predictors, and suggests that relaxing the assuption of a common covariance matrix for both classes to assuming each class has its own covariance matrix is a more accurate assumption about the data. One thing of note is that the QDA model has a high positive predictive value of $48/49 \approx 0.979$, which is highest value out of all of the models we have seen up to this point. This model would be very good if our priority is to minimize the number of suburbs with below-median crime rates that are predicted to be in the above-median category.

# In[ ]:


clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train[moderate_corr], y_train)
y_pred = clf.predict(X_test[moderate_corr])
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf.classes_, yticklabels = clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


1 - clf.score(X_test[moderate_corr], y_test)


# Reducing to the smaller subset of predictors (`CMEDV`, `INDUS`, `NOX`, `AGE`, `DIS`, `RAD`, `TAX`, `B`, and `LSTAT`) resulted in a QDA model that performs slightly worse in terms of overall test error rate (11.8%) than the QDA model with more predictors. Even so, this QDA model still has a better overall test error rate than both of the LDA models.
# 
# For $k$-nearest neighbors classification, we make sure to standardize all of the variables to have mean zero and standard deviation one by using a pipeline with the StandardScaler as we did in Part 7 of Exercise 2 above.

# In[ ]:


np.random.seed(312)
k_vals = list(range(1, 21, 2))
knn_errors = {}
confusion_matrices = {}
for k in k_vals:
    clf = Pipeline([("standardize", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors = k))])
    clf.fit(X_train[not_chas], y_train)
    y_pred = clf.predict(X_test[not_chas])
    confusion_matrices[k] = confusion_matrix(y_test, y_pred)
    knn_errors[k] = 1 - clf.score(X_test[not_chas], y_test)
pd.Series(knn_errors)


# In[ ]:


mat = confusion_matrices[1]
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf.classes_, yticklabels = clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# In[ ]:


mat = confusion_matrices[3]
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf.classes_, yticklabels = clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# We see that when using all of the predictors aside from `CHAS`, the values of $k$ with the lowest overall test error for $k$-nearest neighbors classification are $k = 1$ and $k = 3$, with an overal test error of 6.3%. This is two fewer mis-classified test observations compared to logistic regression using the same predictors and provides the best overall test error rate of the model/predictor combinations we have investigated thus far. Looking at the confusion matrix, we see that KNN with $k = 3$ has the best positive predictive value out of all of the classification models, $51/52 \approx 0.981$, and compared to logistic regression using the same predictors, this model has a slightly better true positive rate ($51/58 \approx 0.879$ vs 0.954) as well.

# In[ ]:


np.random.seed(312)
k_vals = list(range(1, 21, 2))
knn_errors = {}
confusion_matrices = {}
for k in k_vals:
    clf = Pipeline([("standardize", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors = k))])
    clf.fit(X_train[moderate_corr], y_train)
    y_pred = clf.predict(X_test[moderate_corr])
    confusion_matrices[k] = confusion_matrix(y_test, y_pred)
    knn_errors[k] = 1 - clf.score(X_test[moderate_corr], y_test)
pd.Series(knn_errors)


# In[ ]:


mat = confusion_matrices[1]
fig, ax = plt.subplots()
sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"},
           xticklabels = clf.classes_, yticklabels = clf.classes_)
ax.set(xlabel = "predicted label", ylabel = "true label");


# Reducing to the smaller subset of predictors (`CMEDV`, `INDUS`, `NOX`, `AGE`, `DIS`, `RAD`, `TAX`, `B`, and `LSTAT`), we see that again the value of $k$ with the lowest overall test error rate is $k = 1$. In this case, however, KNN with $k = 1$ and this subset of predictors performed slightly worse than the previous KNN classifier which used more predictors. Looking at the confusion matrix compared to $k = 3$ above, we see that all of the additional errors come from false positives.
# 
# While we would need more precisely determine the metrics by which we would choose our preferred model and then perform further testing, such as cross validation, to more confidently choose a model, it seems as if there are two likely frontrunners to choose: logistic regression using all of the predictors aside from `CHAS`, and $k$-nearest neighbors using $k = 3$ the same set of predictors. The strength of the logistic regression model lies in relative simplicity and interpretability compared to KNN, as well as the fact that it had the second lowest overall test error out of the models we explored. The KNN classifier had slightly lower overall test error, but had a much better positive predictive value than logistic regression, at least when logistic regression used a prediction threshold of 0.5. One graphic we could use to better compare the performance of these two models would be to produce ROC curves for each model and compare the two area-under-curve values.
