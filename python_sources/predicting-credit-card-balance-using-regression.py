#!/usr/bin/env python
# coding: utf-8

# > ## Project Goal
# 
# The goal of this exercise is to analyze a dataset consisting of information from 400 credit card holders and to comprehend which factors influence the Credit Card Balance of a cardholder and to predict the average Balance of a given individual. Such an exercise could be conducted as part of a customer analysis within a credit card company. The results of the analysis could determine which customers present a risk of credit default, or what the expected consumer behavior of prospective customers will be. In addition, combining the credit Balance data with information such as credit Limit can assist in calculating the credit utilization of a card, information which feeds into a cardholder's credit Rating.   
# 
# For this goal, a multivariable regression analysis will be undertaken. The exercise will begin with an exploratory data analysis of the dataset, followed by feature selection and regression analysis, including linear and logistic regression. Lastly, the regression model created will be employed to simulate a new dataset and predict the credit Balance of cardholders given their demographic information. 

# > ## Dataset Description
# 
# This dataset is part of "An Introduction to Statistical Learning with Applications in R" available at http://www-bcf.usc.edu/~gareth/ISL/index.html 
# 
# The present exercise will study the Credit Card Balance Data. This is a data frame with 400 observations on the following variables:
# * ID - Identification
# * Income - Income in \$10,0000
# * Limit - Credit limit
# * Rating - Credit rating
# * Age - Age in years
# * Education - number of years of education
# * Gender - Male or Female
# * Student - Yes or No
# * Married - Yes or No
# * Ethnicity - African American, Asian or Caucasian
# * Balance - Average credit card balance in $
# 
# The aim is to determine which factors influence the credit card Balance of any given individual.
# 

# > ## Assumptions
# 
# The following assumptions about the dataset have been made:
# * Credit card Balance refers to the average monthly balance across all of the cards owned by a cardholder. This assumption was made as a result of the Cards variable which refers to the number of credit cards owned by a person and has only one associated Balance figure.
# * The Balance is calculated as the highest amount incurred on a credit card in a given month. For example if a cardholder spends \$400, \$500, and \$600 over the course of three months, and each month pays the balance in full, the average balance will be recorded as \$500 (i.e. any preliminary balances before the maximum are not taken into account, neither is the final balance of zero).

# > ## Project set-up

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')

col_list = ['#005f9a', '#00CDCD', '#f1bdbf']
sns.set_palette(col_list)


# In[ ]:


credit_df = pd.read_csv("../input/Credit.csv", index_col=0)


# The categorical variables should be converted into the appropriate data type. 

# In[ ]:


credit_df.Gender = credit_df.Gender.astype('category')
credit_df.Student = credit_df.Student.astype('category')
credit_df.Married = credit_df.Married.astype('category')
credit_df.Ethnicity = credit_df.Ethnicity.astype('category')


# > # Exploratory Data Analysis

# In[ ]:


credit_df.describe()


# In[ ]:


credit_df.head()


# In[ ]:


credit_df.describe(include=['category'])


# > ## Describing the target of inference

# One of the first steps of this Exploratory Data Analysis is to examine the target of inference ( Balance ) in isolation.

# In[ ]:


sns.distplot(credit_df.Balance)


# It can be noticed that a large portion of the sample consists of Zero Balance Cards. Therefore there are 2 potential questions that can be answered:
# * Does an individual have a positive average credit card Balance? 
# * If yes, what is the magnitude of the average Balance?
# 
# This is an important distinction because, if the average credit card Balance for a given individual is zero, we conclude that the person does not make use of that credit card. As a credit card company, we may be interested in knowing the average balance across our frequent users (for instance, to identify those at risk of default), and the zero Balances may skew our results. 
# 
# We can create an additional data frame which only contains the observations with a positive Balance. 

# In[ ]:


active_credit_df = credit_df.loc[credit_df.Balance>0,].copy()
active_credit_df.Balance.describe() 


# In[ ]:


sns.distplot(active_credit_df.Balance)


# Without the zero Balances, the curve resembles a normal distribution. The analysis will make use of both data frames to fit models and explore the differences in those models.
# 
# In addition, a new variable can be added to the original data frame which specifies whether an individual is a regular user of their credit card (i.e. whether they have a positive Balance).

# In[ ]:


credit_df['Active'] = np.where(credit_df['Balance']>0, 'Yes', 'No')  
credit_df.Active.describe()


# > ## Pairwise correlations

# A correlation matrix will be created in order to visualize the relationships among the numerical predictors and the target of inference. 

# In[ ]:


numeric_credit_df = credit_df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(8,8))
plt.matshow(credit_df.corr(), cmap=plt.cm.Blues, fignum=1)
plt.colorbar()
tick_marks = [i for i in range(len(numeric_credit_df.columns))]
plt.xticks(tick_marks, numeric_credit_df.columns)
plt.yticks(tick_marks, numeric_credit_df.columns)


# Based on the correlation matrix, Balance appears correlated with Limit, Rating, and moderately correlated with Income. In addition, Limit and Rating are highly correlated with each other, and they both have a relationship with Income. 
# 
# However, this matrix does not confirm whether the correlation coefficients are statistically significant, hence further investigation is necessary.

# In[ ]:


from scipy.stats import pearsonr
r1, p1 = pearsonr(credit_df.Balance, credit_df.Limit)
msg = "Correlation coefficient Balance-Limit: {}\n p-value: {}\n"
print(msg.format(r1, p1))
r2, p2 = pearsonr(credit_df.Balance, credit_df.Rating)
msg = "Correlation coefficient Balance-Rating: {}\n p-value: {}\n"
print(msg.format(r2, p2))
r3, p3 = pearsonr(credit_df.Balance, credit_df.Income)
msg = "Correlation coefficient Balance-Income: {}\n p-value: {}\n"
print(msg.format(r3, p3))
r4, p4 = pearsonr(credit_df.Limit, credit_df.Rating)
msg = "Correlation coefficient Limit-Rating: {}\n p-value: {}\n"
print(msg.format(r4, p4))
r5, p5 = pearsonr(credit_df.Limit, credit_df.Income)
msg = "Correlation coefficient Limit-Income: {}\n p-value: {}\n"
print(msg.format(r5, p5))
r6, p6 = pearsonr(credit_df.Rating, credit_df.Income)
msg = "Correlation coefficient Rating-Income: {}\n p-value: {}\n"
print(msg.format(r6, p6))


# All the relationships are significant. Additionally Limit and Rating have a remarkably high correlation coefficient.

# In[ ]:


sns.regplot(x='Limit',
           y='Rating',
           data=credit_df,
           scatter_kws={'alpha':0.2},
           line_kws={'color':'black'})


# Limit and Rating are highly correlated, introducing multi-colinearity in the model. More specifically, Rating as an antecedent of Limit is more meaningful for the model because it also drives Limit levels for card owners. Therefore, if one of the two should be removed to fix collinearity issues, Limit has a lower priority for the model. Finally, it is pertinent to note that both Limit and Rating are complex measures, subsuming a range of other factors, among which several already present in the model, such as Education or Income. 
# 
# For clarity, the analysis will focus less on Rating and Limit, and more on those measures representing more direct indicators of credit profiling. 
# 
# We will now examine the categorical variables and their relationship to balance. 

# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(15, 6))
f.subplots_adjust(hspace=.3, wspace=.25)
credit_df.groupby('Gender').Balance.plot(kind='kde', ax=axes[0][0], legend=True, title='Balance by Gender')
credit_df.groupby('Student').Balance.plot(kind='kde', ax=axes[0][1], legend=True, title='Balance by Student')
credit_df.groupby('Married').Balance.plot(kind='kde', ax=axes[1][0], legend=True, title='Balance by Married')
credit_df.groupby('Ethnicity').Balance.plot(kind='kde', ax=axes[1][1], legend=True, title='Balance by Ethnicity')


# Student appears to be the only predictor to influence the distribution of Balance. To verify, the same relationships can be analyzed on the active-only sample of the population. 

# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(15, 6))
f.subplots_adjust(hspace=.3, wspace=.25)
active_credit_df.groupby('Gender').Balance.plot(kind='kde', ax=axes[0][0], legend=True, title='Balance by Gender')
active_credit_df.groupby('Student').Balance.plot(kind='kde', ax=axes[0][1], legend=True, title='Balance by Student')
active_credit_df.groupby('Married').Balance.plot(kind='kde', ax=axes[1][0], legend=True, title='Balance by Married')
active_credit_df.groupby('Ethnicity').Balance.plot(kind='kde', ax=axes[1][1], legend=True, title='Balance by Ethnicity')


# Although the variables Gender, Married, and Ethnicity do not appear associated with Balance when observed in isolation, their interaction with one another might make them valuable. This will be investigated at a later point in the analysis. 
# 
# The Student variable should be closely examined at this point.

# In[ ]:


sns.boxplot(x='Student', y='Balance', data = credit_df)


# Students display on average higher credit card Balances. We infer that Students have a higher need for financing due to student loans and have generally lower personal income (which will be seen later in the analysis). Moreover, we could presume that students tend to have less control over their finances and their financial flexibility may be lower. 

# > # Multivariable Regression Modelling

# After visually examining the predictors and their relationship with Balance, regression models can be fit using those predictors. feature selection will be used to determine which combination of variables best predicts credit card Balance. 
# 
# Given the choice of 9 predictors, the initial model could include all of them. Yet, as previously noted, due to potential collinearity issues, Limit will be eliminated. 
# 
# The Ordinary Least Square algorithm is used to create the linear regression model.

# In[ ]:


mod0 = smf.ols('Balance ~ Income + Rating + Cards + Age + Education + Gender + Student + Married + Ethnicity', data = credit_df).fit()
mod0.summary()


# Income is negatively related to balance which could be interpreted in the sense that the higher the Income, the lower the need to access loans. 
# 
# Student has an expected positive impact. Similarly, these results are consistent with Age. This variable reveals that with Age, the credit card Balance decreases which corroborate the notion that people become more in charge of their finances over time. 
# 
# Surprisingly, the years of Education do not have a significant impact on the credit card Balance. Other variables with no impact are Cards, Ethnicity, Gender and Married. 
# 
# We fit the same model on our reduced dataset of only active customers.

# In[ ]:


active_mod0 = smf.ols('Balance ~ Income + Rating + Cards + Age + Education + Gender + Student + Married + Ethnicity', data = active_credit_df).fit()
active_mod0.summary()


# The model has a superior fit for the Active customers dataset. This points to the hypothesis that perhaps the non-active customers are driven by some reason other than the ones present in our variables. Since they rarely make use of their credit card, it is difficult to draw conclusions. Perhaps they, in fact, have credit card debt, yet we do not have the relevant data, potentially their credit card is provided by a different company. An alternative explanation could be that the card owners maintain a zero-balance card in order to decrease their credit utilization and boost their credit rating, assuming they also own a positive balance credit card elsewhere. 
# 
# We could exclude the variables which yield a high p-value from the model. Nevertheless, these variables could have an interaction with other variables and might still be significant.
# 
# We continue by fitting a model with only the variables that proved significant, yet as part of the analysis, we will return to examining the other variables. 

# In[ ]:


mod1 = smf.ols('Balance ~ Income + Rating + Age + Student', data = credit_df).fit()
mod1.summary()


# Now all the variables are significant, but the R-squared has, in fact, decreased. 
# 
# The potential relationships among these 4 variables should be analyzed.

# In[ ]:


f, axes = plt.subplots(3, 2, figsize=(12, 10))
f.subplots_adjust(hspace=.5, wspace=.25)
credit_df.groupby('Student').Income.plot(kind='kde', ax=axes[0][0], title='Income by Student')
credit_df.groupby('Student').Rating.plot(kind='kde', ax=axes[0][1], title='Rating by Student')
credit_df.plot(kind='scatter', x='Age' , y='Income' , ax=axes[1][0], title='Income and Age')
credit_df.plot(kind='scatter', x='Age' , y='Rating' , ax=axes[1][1], color='orange', title='Rating and Age')
credit_df.plot(kind='scatter', x='Rating' , y='Income' , ax=axes[2][0], color='orange', title='Income and Rating')
credit_df.groupby('Student').Age.plot(kind='kde', ax=axes[2][1], legend=True, title='Age by Student')


# As expected, we notice a positive relationship between Income and Rating. This could be explained by the fact that credit Rating is a score assigned to individuals based on their creditworthiness, including the level of personal Income. 
# 
# Furthermore, as we predicted earlier, Students display lower values of Income compared to non-Students. 
# 
# Surprisingly, in this dataset, the Income does not Increase with Age, and the Age of Students compared to non-Students does not differ significantly. This observation could lead us to be concerned about the data quality or to seek further attributes of this population to conduct additional investigation. Nonetheless, given the simulated nature of this dataset additional inquiries are not feasible, yet it is worth noting that on a different population, we expect a stronger relationship among Age, Student, and Income. 
# 
# We will now closer examine Balance and Income for Students and non-Students. Since we are interested in how Balance behaves, we will only take into account non-zero Balances.

# In[ ]:


sns.lmplot(x='Income',
          y='Balance',
          data=active_credit_df,
          line_kws={'color':'black'},
          lowess=True,
          col='Student')


# We observe a positive relationship between Balance and Income for non-Students, however, in the case of Students, changes in Income do not impact their average credit card Balance. A further step would be to fit a regression model on non-Students only and observe Balance. For Students, the line appears to behave differently in 3 sections, so perhaps a Spline model might be appropriate. However, the sample size of Students is not large enough (no. of observations = 40) for the results to add significant value to the overall analysis.
# 
# An interesting observation is that the regression analysis showed that Income was negatively related to Balance, yet this figure appears to show a positive relationship. We should verify whether the relationship is in fact non-linear.

# > ## Non-linear Relationships

# In[ ]:


mod2 = smf.ols('Balance ~ Income + I(Income**2) + Age + Student + Rating', data = credit_df).fit()
mod2.summary()


# The model has improved and the non-linear term is marginally significant. Additionally, while Income has a negative impact on Balance, Income squared has a slightly positive impact. We expect Balance to have a negative slope initially, and a positive one at higher levels of Income. We can plot this relationship.

# In[ ]:


sns.regplot('Balance', 'Income',
           data = active_credit_df,
           ci=None,
           order=2,
           line_kws={'color':'black'})


# As anticipated, at lower levels of Income, increases in personal Income cause a decrease in credit card Balance, which can be interpreted as individuals requiring less financing as they make use of personal finances instead of credit debt. 
# 
# However, at high levels of income, Balance increases, meaning that those individuals are in higher need of loans, potentially due to increased investment activities and a greater risk tolerance. 

# > ## Interactions Between Predictors
# 
# An interaction term could be considered between Income and Rating. This stems from the assumption that there could be a synergy between Income and Rating; that individuals with increased levels of Income and high credit Rating will have a higher level of financial control and stability.

# In[ ]:


mod3 = smf.ols('Balance ~ Income + I(Income**2) + Age + Student + Income*Rating', data = credit_df).fit()
mod3.summary()


# The Adjusted R-squared increased and the interaction term is statistically significant.
# 
# We could examine how the variables that were originally removed interact with other variables, such as for example years spent in Education and personal Income. 

# In[ ]:


mod4 = smf.ols('Balance ~ Income + I(Income**2) + Rating + Age + Student + Education*Income', data = credit_df).fit()
mod4.summary()


# Interestingly, while education did not appear significant at first, it does interact with Income, with a positive effect. Moreover, in this model Education itself seems marginally significant, meaning that the longer the time spent in education, the lower the Balance and the more financially cautious an individual is. Yet the effect reverses at high levels of Income and Education were highly educated high earning individuals appear to have higher needs for financing.  
# 
# Let us consider further interactions, such as the one between Married and Age.

# In[ ]:


mod5 = smf.ols('Balance ~ Income + I(Income**2) + Rating + Age + Student + Married*Age', data = credit_df).fit()
mod5.summary()


# While neither Married nor Age by themselves are significant, the interaction term is. This reveals the fact that individuals with higher values for Age who are also Married have lower credit card Balances pointing to higher financial prudence or risk aversion. We can visualize this relationship.

# In[ ]:


sns.lmplot(x="Age", 
           y="Balance", 
           hue="Married", 
           ci=None,
           data=active_credit_df);


# We continue by examining Gender and number of credit Cards.

# In[ ]:


mod6 = smf.ols('Balance ~ Income + I(Income**2) + Rating + Age + Student + Gender*Cards', data = credit_df).fit()
mod6.summary()


# In[ ]:


sns.lmplot(x="Cards", 
           y="Balance", 
           hue="Gender", 
           ci=None,
           data = credit_df);


# The interaction between Gender and Cards is significant with a high coefficient, which demonstrates that Females who own more Cards have on average higher Balance. In this dataset we make the assumption that the Balance is recorded per person and not per credit card, implying that the balance recorded is calculated as an average across all the cards belonging to the same individual. While owning more credit Cards suggests in itself higher financing needs (perhaps the multiple cards offer different benefits), the relation with Gender is worth noting. We should also mention the fact that Gender in isolation has a negative impact on Balance, suggesting that females, in general, have less credit card debt, except when that individual also owns multiple Cards. 

# > # Finding the Best Model

# After investigating polynomial relationships and interaction among terms, the last part of the analysis will focus on finding the best model and using it to predict credit card Balance for a generated sample of individuals. The purpose of this exercise is to employ the model and make inferences on the target variable, given a set of predictors. In other words, if an individual applies for a credit card, this model will predict an average credit card balance given the demographic factors (i.e. Income, Age, Rating). The information can be further utilized to estimate the risk of credit default or other customer behavior indicators of the given card owner. 
# 
# For this part of the analysis, we will focus separately on the active and non-active samples of the population. 
# 
# The best performing model so far on the entire sample included a polynomial relationship between Income and Balance and an interaction term between Income and Rating. 

# In[ ]:


active_mod7 = smf.ols('Balance ~ Income + I(Income**2) + Rating + Age + Student + Income*Rating', 
                      data = active_credit_df).fit()
active_mod7.summary()


# In the case of the active user population, neither non-linear relationships nor interaction among variables displayed low p-values. Nevertheless, other variables might explain the variance in Balance, such as Limit or Cards.

# In[ ]:


active_mod8 = smf.ols('Balance ~ Limit + Rating + Income + Age + Student + Cards', data = active_credit_df).fit()
active_mod8.summary()


# This is the best model so far, with an R-squared of 99%. Adding the Limit term seems to have had a strong impact. This is a relationship which can be further analyzed. 
# 

# In[ ]:


sns.regplot(x='Limit',
          y='Balance',
          data=active_credit_df,
          line_kws={'color':'black'},
          lowess=True)


# Credit Limit appears to be a strong predictor for credit card Balance. This is an expected finding since we assume that a card owner is not allowed to have a Balance that exceeds their Limit. Consequently, this relationship simply articulates that, the higher the credit Limit, the greater the credit card expenses are for a given individual, and hence the insightfulness of this relationship is debatable. 

# In[ ]:


mod9 = smf.ols('Balance ~ Rating', data = credit_df).fit()
mod9.summary()


# Similarly, credit Rating is a highly accurate predictor of Balance. This could suggest that individuals with high Rating are more willing to incur credit debt as they are confident that they will be able to pay off the balance. Nevertheless, Rating is a complex variable, related to other predictors such as Income. 

# For the entire dataset, the best model predicted 96% of the variance, while the model fit on the active-only population predicted 99%. The difference suggests that there are other factors influencing non-active cardholders which are not present in our data, or their spending behaviour is reflected on other lending platforms.

# > ## Logistic Regression 
# 
# In order to determine the factors that influence whether a card owner is an active credit card user, a logistic regression can be fit. 

# In[ ]:


log_mod = smf.glm('Active ~ Limit + Rating + Income + Age + Cards + Education', 
                   data = credit_df,
                   family=sm.families.Binomial()).fit()
log_mod.summary()


# The results unveil that Income is the only statistically significant predictor of whether the card owner is active or not, high earners having a greater probability of being active. This could be explained by low earners maintaining a zero-balance card in order to boost their creditworthiness. 
# 
# Naturally, a more in-depth investigation could follow and more complex models can be fit. However, the Active variable could be influenced by factors outside of our predictors. As previously stated, non-active users could be individuals who make use of other credit cards, yet we do not have access to such data. 

# > ## Making Predictions

# Lastly, the model can be utilized in order to predict the credit card Balance of future customers given their demographic information. For this purpose, a dataset has been generated.

# In[ ]:


df_new=pd.DataFrame({'Income':np.random.normal(45, 20, 40),
                    'Rating':np.random.normal(355, 55, 40),
                    'Limit':np.random.normal(4735, 200, 40),
                    'Age':np.random.normal(56, 17, 40),
                    'Cards':list(range(0,10))*4,
                    'Student':['Yes']*20+['No']*20})
df_new.Cards[df_new.Cards == 0] = 3
df_new.Income[df_new.Income <= 0] = df_new.Income.mean()
df_new.Rating[df_new.Rating <= 0] = df_new.Rating.mean()
df_new.Limit[df_new.Limit <= 0] = df_new.Limit.mean()
df_new['Balance']= active_mod8.predict(df_new)
df_new.describe()


# In[ ]:


mod8 = smf.ols('Balance ~ Income + I(Income**2) + Age + Student + Income*Rating + Limit + Cards', data = credit_df).fit()
mod8.summary()

