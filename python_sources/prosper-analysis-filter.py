#!/usr/bin/env python
# coding: utf-8

# # Prosper Loan Data - Default Analysis
# 
# * Forked from: https://www.kaggle.com/jschnessl/prosper-analysis

# ### Table Of Contents
# 
# * [Introduction](#intro)
# * [Preprocessing: imputation of missing values](#na)
# * [Dimensionality reduction: part 1](#reduce1)
# * [Focusing in on default](#default)
# * [Exploring the data: categorical information](#explore1)
# * [Exploring the data: credit scores](#explore2)
# * [Exploring the data: financial information](#explore3)
# * [Exploring the data: credit history](#explore4)
# * [Exploring the data: loan characteristics](#explore5)
# * [Dimensionality reduction: part 2](#reduce2)
# * [Preprocessing: fixing data types](#dtypes)
# * [Train/test split](#split)
# * [Preprocessing: scaling features to a range](#scale)
# * [Dimensionality reduction: part 3](#reduce3)
# * [Choosing a classifier](#class)
# * [Parameter tuning](#tuning)
# * [Predicting defaults](#predict)
# * [Conclusion](#end)

# ### Introduction <a class="anchor" id="intro"></a>
# 
# This notebook will document my efforts to investigate an interesting dataset from the Prosper peer-to-peer lending platform, and to then apply some machine learning classifiers. There are many questions one could ask of this data, but in this exercise I will focus on one question specifically: whether one can successfully predict which loans will default. Prosper loans pay pretty hefty interest rates to their creditors. There is thus a significant financial incentive to accurately predicting which of the loans would eventually default or not.
# 
# The original data can be found here: https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv. A variable dictionary concisely explaining the data can be found here: https://docs.google.com/spreadsheets/d/1gDyi_L4UvIrLTEC6Wri5nbaMmkGmLQBk-Yx3z0XDEtI/edit?usp=sharing.
# 
# I will clean the data, separate out historical data, explore the data with some visualisations, further prepare the data, and finally choose and tune a classification algorithm before a brief conclusion to wrap everything up summarizing the hypothetical financial result of using the trained classifier to predict loan default.

# In[ ]:


#Imports

#Data analysis and math
import math
import datetime
import numpy as np
import pandas as pd
from scipy import stats as st

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
sns.set_context({"figure.figsize": (15, 7.5)})

#Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile

#Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.grid_search import GridSearchCV

#Metrics
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict

from sklearn.metrics import recall_score, precision_score, f1_score, make_scorer


# In[ ]:


#Input (preserve original data in case we need to refer back to it at any point)
df = original_data = pd.read_csv("../input/prosperLoanData.csv")

pd.set_option("display.max_columns", len(df.columns))
df.head()


# In[ ]:


#Examine columns, missing values, and data types
df.info()


# Taking a quick look at the data, there are a few potential processing tasks that jump out right away. 
# 
#     1) Some variables have a lot of null values. TotalProsperLoans, for instance, and a lot of other variables associated with the Prosper history of the debtor.
#     2) A lot of these variables seem to be administrative identifiers (e.g. ListingKey and ListingNumber), and redundant ones at that. They won't be of much use to us. 
#     3) There are 17 variables of type object. Those will probably need some attention before they're ready for some of our classification models.
#     4) Variables have a wide variety of ranges. We will probably need to re-scale them to make them more amenable to some of our classifiers.
#     
# Let's start by filling in these NaN values as best we can.

# ### Preprocessing: imputation of missing values <a class="anchor" id="na"></a>

# Beginning with categorical data, for now let's just fill the NaN values with "Unknown".

# In[ ]:


# categorical = df.select_dtypes(include=["object"]).columns.values
# df[categorical] = df[categorical].fillna("Unknown")

# df.select_dtypes(exclude=[np.number]).isnull().sum()


# Next, about 20 loans are missing an APR value. Because APR is equal to the borrower rate + fees, let's calculate the median difference between the two, and add that value to the borrower rate of our data points missing an APR.

# In[ ]:


borrower_fees = df["BorrowerAPR"] - df["BorrowerRate"]
borrower_fees.median()


# In[ ]:


df["BorrowerAPR"].fillna(df["BorrowerRate"] + borrower_fees.median(), inplace=True)

df["BorrowerAPR"].isnull().sum()


# EstimatedEffectiveYield will always be the borrower rate minus some expected loss from interest charge-offs and fees.

# In[ ]:


estimated_loss_from_fees = df["BorrowerRate"] - df["EstimatedEffectiveYield"]
estimated_loss_from_fees.median()


# In[ ]:


df["EstimatedEffectiveYield"].fillna(df["BorrowerRate"] - estimated_loss_from_fees.median(), inplace=True)

df["EstimatedEffectiveYield"].isnull().sum()


# EstimatedLoss is harder to gauge. Let's just take the median.

# In[ ]:


df["EstimatedLoss"].fillna(df["EstimatedLoss"].median(), inplace=True)

df["EstimatedLoss"].isnull().sum()


# EstimatedReturn is defined as EstimatedEffectiveYield - EstimatedLoss.

# In[ ]:


df["EstimatedReturn"].fillna(df["EstimatedEffectiveYield"] - df["EstimatedLoss"], inplace=True)

df["EstimatedReturn"].isnull().sum()


# The numeric ProsperRating and the ProsperScore NaNs can both be replaced with median values.

# In[ ]:


# df["ProsperRating (numeric)"].fillna(df["ProsperRating (numeric)"].median(), inplace=True)
# df["ProsperScore"].fillna(df["ProsperScore"].median(), inplace=True)


# df["ProsperRating (numeric)"].isnull().sum(), df["ProsperScore"].isnull().sum()


# There are a host of variables which hold a lot of promise for our eventual classification algorithm, and are only missing a relatively small number of values. I'm going to drop all the rows missing values for these, rather than filling them out with arbitrary median values or 0s.

# In[ ]:


df.shape


# In[ ]:


df.dropna(subset=["EmploymentStatusDuration", "CreditScoreRangeLower", "FirstRecordedCreditLine", "CurrentCreditLines",
                  "TotalCreditLinespast7years"], inplace=True)
df.shape


# Let's see how we're doing so far.

# In[ ]:


df.info()


# It seems that most of our remaining Null values fall into four groups: 
# 
#     1) DebtToIncomeRatio, which strikes me as a potentially very useful feature. Let's take a look at what's going on there and do our best to reconstruct or substitute the missing values.
#     2) ScorexChangeAtTimeOfListing, which is the difference between the borrower's credit score when it was reviewed for this loan, versus the score last time they took a Prosper loan. We'll have to think about how we can deal with that, because it's an interesting potential feature.
#     2) Data dealing with the debtor's Prosper history, which we can fill with 0s to represent a lack of such history.
#     3) LoanFirstDefaultedCycle, which we are actually going to drop entirely very shortly for reasons to be explained. 

# In[ ]:


df_debt_income_null = df[df["DebtToIncomeRatio"].isnull()]

df_debt_income_null[:5]


# In[ ]:


df.loc[40]


# In[ ]:


df.loc[40, "MonthlyLoanPayment"], df.loc[40, "StatedMonthlyIncome"]


# This is bizarre, because this debtor has both debt and income data available. Let's take a look at some other cases.

# In[ ]:


df_debt_income_null = df[df["DebtToIncomeRatio"].isnull()]

df_debt_income_null["MonthlyLoanPayment"].isnull().sum(), df_debt_income_null["StatedMonthlyIncome"].isnull().sum()


# All of these data points have a monthly income and monthly loan payment. Maybe it has to do with income being verifiable?

# In[ ]:


df_debt_income_null["IncomeVerifiable"][:10]


# There we have it. The question is thus how to treat these variables. I'm inclined to calculate the Debt to Income ratio based on the stated monthly income, and to allow verifiable income to work as a feature that quantifies the risk that that income is overstated.

# In[ ]:


# #Calculate DebtToIncomeRatio for unverifiable incomes, adding $1 to account for $0/month incomes
# df["DebtToIncomeRatio"].fillna(df["MonthlyLoanPayment"] / (df["StatedMonthlyIncome"] + 1), inplace = True)

# df["DebtToIncomeRatio"].isnull().sum()


# Now let's think about how to treat the change in credit score over time. It would be very interesting to see whether a rising or a falling credit score correlates with default. Unfortunately, because it relies on a history of borrowing with Prosper, most loans are missing a value here. Unlike the other Prosper history variables, it doesn't really make sense to replace the Null values with a 0, as that would assert that the score has been constant and mislead our investigation or our models. Similarly, replacing with a measure of central tendency could throw things off. I think that, unfortunately, the most conservative approach will be to drop the column entirely.

# In[ ]:


# df.drop("ScorexChangeAtTimeOfListing", axis=1, inplace=True)


# Now let's fill the missing Prosper histories with 0s.

# In[ ]:


# prosper_vars = ["TotalProsperLoans","TotalProsperPaymentsBilled", "OnTimeProsperPayments", "ProsperPaymentsLessThanOneMonthLate",
#                 "ProsperPaymentsOneMonthPlusLate", "ProsperPrincipalBorrowed", "ProsperPrincipalOutstanding"]

# df[prosper_vars] = df[prosper_vars].fillna(0)

# df.isnull().sum()


# ### Dimensionality reduction: part 1 <a class="anchor" id="reduce1"></a>

# Now that we've dealt with Null values, at least for the time being, let's see if we can't quickly get rid of some extraneous information by dropping redundant or wholly irrelevant columns from the dataset. Let's begin with redundant administrative variables.

# In[ ]:


# df.drop(["ListingKey", "ListingNumber", "LoanKey", "LoanNumber"], axis=1, inplace=True)

df.drop([ "ListingNumber", "LoanNumber"], axis=1, inplace=True)


# There are a few other variables which could have useful information, but would necessitate a different analysis, so we will drop them too. For example, MemberKey could be used in an interesting inquiry into whether certain debtors are consistently mis-classed as risky or high-interest when in fact they consistently pay their loans back. Likewise, a time-series analysis would be very interesting: did defaults spike around the financial crisis, or do defaults become more common around Christmas? These are fascinating questions, but I think it will be better to stay focused and to drop variables like this which increase complexity unnecessarily.

# In[ ]:


# df.drop(["ListingCreationDate", "ClosedDate", "DateCreditPulled", "LoanOriginationDate", "LoanOriginationQuarter", "MemberKey"],
#         axis=1, inplace=True)


# Similarly, there is a whole class of variables that describe the status of a loan at present, or the history of the loan. 
# These are beyond the scope of this analysis. We will shortly lump delinquent loans in with defaults, and drop all loans that are current, and so, given our focus, certain columns (e.g. LoanCurrentDaysDelinquent) are irrelevant to us.

# In[ ]:


# df.drop(["LoanCurrentDaysDelinquent", "LoanFirstDefaultedCycleNumber", "LoanMonthsSinceOrigination", "LP_CustomerPayments",
#          "LP_CustomerPrincipalPayments", "LP_InterestandFees", "LP_ServiceFees", "LP_CollectionFees", "LP_GrossPrincipalLoss",
#          "LP_NetPrincipalLoss", "LP_NonPrincipalRecoverypayments"], axis=1, inplace=True)


# Alright, now that we have gotten rid of a couple variables which won't be helpful to the analysis, let's make sure all of our missing values have been dealt with before turning our attention to historical loans exclusively, which will be most suitable to our purposes.

# In[ ]:


df.info()


# ### Focusing in on default <a class="anchor" id="default"></a>

# Let's investigate the data in relation to the real question: default. Let's first convert our LoanStatus variable into a binary variable excluding current loans, and visualize some of the relationships between default and other variables. 

# In[ ]:


df["LoanStatus"].value_counts()


# What is really of interest to us is what distinguishes completed loans from defaulted loans. Because there is no way to tell whether "current" loans will eventually default or not, we can't use them for our analysis. Nearly half of the dataset is not useful to us, as the loans are still outstanding. In order to be conservative in our eventual estimates, to simplify the problem, and to retain data, let's assume all the "past due" and "chargedoff" loans (and that 1 cancellation) will default. Thus we'll be left with two classes: "completed" and "defaulted". Let's encode those binary outcomes as 1 and 0, respectively.

# In[ ]:


#Remove outstanding loans

df_historical = df[df["LoanStatus"] != "Current"]

df_historical["LoanStatus"].value_counts()


# In[ ]:


#Encode all completed loans as 1, and all delinquent, chargedoff, cancelled and defaulted loans as 0

df_historical["LoanStatus"] = (df_historical["LoanStatus"] == "Completed").astype(int)

df_historical["LoanStatus"][:10]


# In[ ]:


df_historical["LoanStatus"].mean(), 1 - df_historical["LoanStatus"].mean()


# So, in the historical data as a whole, 67.43% of loans are completed. 32.57% of loans "defaulted". (During analysis, one should always bear in mind that default as we've defined it includes charge-offs, cancellations, and even any current loans with late payments. It might be more constructive to think of the loans labelled with a 0 as "bad" loans rather than outright "defaults").

# Let's take a look at our new historical dataframe and then start exploring variables' relationship with default, starting with some potentially useful categorical information.

# In[ ]:


df_historical.describe()


# In[ ]:


df_historical.Lis


# Let's take a look at another categorical variable: ListingCategory.
# 
# First though, it's a bit annoying having to constantly refer to our variable definitions to understand the listing category, so before we interpret this, let's change our numeric values to the actual category names. This will also be useful, because the numeric values imply some sort of false ordinality, and we should really handle this like a categorical variable.

# In[ ]:


df_historical.replace(to_replace={"ListingCategory (numeric)": {0: "Unknown", 1: "Debt", 2: "Reno", 3: "Business", 4: "Personal",
                                                                5: "Student", 6: "Auto", 7: "Other", 8: "Baby", 9: "Boat", 
                                                                10: "Cosmetic", 11: "Engagement", 12: "Green", 13: "Household",
                                                                14: "LargePurchase", 15: "Medical", 16: "Motorcycle", 17: "RV",
                                                                18: "Taxes", 19: "Vacation", 20: "Wedding"}}, inplace=True)

df_historical.rename(index=str, columns={"ListingCategory (numeric)": "ListingCategory"}, inplace=True)

df_historical["ListingCategory"][:10]


# In[ ]:


sns.barplot(x="ListingCategory", y="LoanStatus", data=df_historical)


# In[ ]:


# rv, green = df_historical[df_historical["ListingCategory"] == "RV"], df_historical[df_historical["ListingCategory"] == "Green"]

# 1 - rv["LoanStatus"].mean(), 1 - green["LoanStatus"].mean()


# So, once again, certain types of loans seem to be outperforming others, with RV loans only defaulting 11.11% of the time, and green loans defaulting 56.52% of the time. Other frequent defaulters are loans for household expenses and for medical and dental work, while people seem to completely pay boat and motorcycle loans quite frequently. 

# Let's examine some credit scoring metrics.

# ### Exploring the data: credit scores <a class="anchor" id="explore2"></a>

# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(221)
sns.barplot(x="ProsperRating (numeric)", y="LoanStatus", data=df_historical)

ax2 = fig.add_subplot(222)
sns.barplot(x="ProsperScore", y="LoanStatus", data=df_historical)

ax3 = fig.add_subplot(223)
sns.barplot(x="CreditScoreRangeLower", y="LoanStatus", data=df_historical)

ax4 = fig.add_subplot(224)
sns.barplot(x="CreditScoreRangeUpper", y="LoanStatus", data=df_historical)


# Both the Prosper scores and the credit scores seem to be doing a good job of predicting default, with higher ratings defaulting less frequently. Interestingly, the loans with the highest ProsperScores (11) default more frequently than loans rated at a 9 or a 10, and there is a high degree of variance in default rate for those highly rated loans.
# 
# It should be noted here that the credit score "range" seems to be constant. So let's quickly double-check that, and remove the redundancy by dropping CreditScoreRangeUpper and renaming the lower bound "CreditScore".

# In[ ]:


credit_score_range = df_historical["CreditScoreRangeUpper"] - df_historical["CreditScoreRangeLower"]

credit_score_range.value_counts()


# In[ ]:


df_historical.drop("CreditScoreRangeUpper", axis=1, inplace=True)

df_historical.rename(index=str, columns={"CreditScoreRangeLower": "CreditScore"}, inplace=True)


# Let's visualize some financial variables.

# ### Exploring the data: financial information <a class="anchor" id="explore3"></a>

# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(221)
sns.barplot(x="EmploymentStatus", y="LoanStatus", data=df_historical)

ax2 = fig.add_subplot(222)
sns.boxplot(x="LoanStatus", y="EmploymentStatusDuration", data=df_historical).set_ylim([0,400])


# There doesn't seem to be a relationship between EmploymentStatusDuration and loan default. We'll take a look at the correlation in a moment. EmploymentStatus does, however, seem to have a relationship with default. Interestingly, part-time workers defaulted less often than full-time workers. People who listed their employment status as "Other" defaulted even more often than those who weren't employed at all.

# In[ ]:


x = df_historical["EmploymentStatusDuration"]
y = df_historical["LoanStatus"]

r, p = st.pearsonr(x, y)

print("The correlation between employment status duration and loan default is {}, with a p-value of {}".format(r, p))


# The very weak correlation between employment status duration and loan default, as well as the lack of a statistically significant p-value, lead me to believe that we can safely drop this variable from our dataset.

# In[ ]:


# df_historical.drop("EmploymentStatusDuration", axis=1, inplace=True)


# Both current and historical delinquency correlate significantly (p < 0.05) with loan status. Current delinquencies in particular should be a very strong feature for us.

# The distribution of the loan amount seems to be consistent between both classes. 

# In[ ]:


df_historical["Term"].value_counts()


# Prosper loans can only have a term of 12, 36, or 60 months. By far the most common is the term of 3 years. Term probably won't be of much use to us to classify our loans either then. Let's wrap up by examining interest rates.

# The mean interest rate (without any fees) for all Prosper loans is a fairly substantial 20.35%. From my experience and a quick google, that is roughly equivalent to that of a credit card. The standard deviation is 8.25% however, meaning that the lowest 25% of borrowers are paying a much more reasonable 13.64%, but that the top 25% of borrowers are paying 27.00% and more, before fees!

# Generally, loans that default have a higher interest rate than loans that are paid completely, even when controlling for credit score. That makes sense, but it's still really cool to see in the chart above.

# Right, so we've had the chance to take a look at some variables and their relationship to default. It seems there are a lot of good potential features available to us, although some of them are categorical and will need some work before we can use them in all of our classifiers.

# ### Dimensionality reduction: part 2 <a class="anchor" id="reduce2"></a>

# In[ ]:


df_historical.describe()


# * Drop  further columns  that I don't think will help our model too much. 
# Thinking: 
# The wide majority of loans were fully funded, and there are other variables (like Investors) capturing similar information, so I think it's safe to drop the PercentFunded variable. 
# BorrowerAPR and LenderYield are both versions of BorrowerRate, just with fees included, so they're unnecessary.
# CreditGrade and ProsperRating (Alpha) are annoying categorical variables with many proxies already.
# IncomeRange is also a tricky data type and basically redundant. 
# Occupation could be very interesting, but being such a broad categorical variable it would be incredibly difficult to use. 
# Group affiliation could be an interesting line of inquiry, but I feel like that is probably best left to another investigation into Prosper's grouping practices. 
# Finally, the estimated variables could be used as features, but I feel that since the entire point of this exercise is to outperform Prosper's estimates of risk, it is somewhat contradictory to use their precise estimates in our classifications. 

# In[ ]:


# df_historical.drop(["CreditGrade", "BorrowerAPR", "LenderYield", "EstimatedEffectiveYield", "EstimatedLoss", "EstimatedReturn",
#                  "ProsperRating (Alpha)", "Occupation", "CurrentlyInGroup", "GroupKey", "IncomeRange", "PercentFunded"], axis=1,
#                 inplace=True)

# df_historical.info()


# ### Preprocessing: fixing data types <a class="anchor" id="dtypes"></a>

# Let's quickly convert our boolean values into 0s and 1s for consistency's sake.

# In[ ]:


# df_historical["IsBorrowerHomeowner"] = df_historical["IsBorrowerHomeowner"].astype(int)
# df_historical["IncomeVerifiable"] = df_historical["IncomeVerifiable"].astype(int)

# df_historical["IsBorrowerHomeowner"][:10], df_historical["IncomeVerifiable"][:10]


# Lovely. Now, let's turn our attention to those powerful categorical variables. Of the 4 remaining variables of type "object", one stands out: FirstRecordedCreditLine. It could be a datetime object, let's take a look.

# In[ ]:


df_historical["FirstRecordedCreditLine"][:10]


# The length of credit history could potentially be a very powerful feature so I'd like to keep this if possible. Let's try turning this variable into a "YearsWithCredit" variable that will take continuous integer values rather than being a datetime object. This data comes from 2014, so let's compare the earliest recorded credit lines to 2014.

# In[ ]:


first_credit_year = df_historical["FirstRecordedCreditLine"].str[:4]

df_historical["YearsWithCredit"] = 2014 - pd.to_numeric(first_credit_year)

df_historical.drop("FirstRecordedCreditLine", axis=1, inplace=True)

df_historical["YearsWithCredit"][:10]


# We're left with 3 variables of type "object" that need to be cleaned up: ListingCategory, BorrowerState and EmploymentStatus. Let's create dummy variables for each of these and join them to the high_yield dataframe. BorrowerState has, of course, 50 possible values (52 actually, counting DC and "Unknown"), so with that one I'm first going to group states into a few bins depending on their rates of default.

# In[ ]:


# category = pd.get_dummies(df_historical["ListingCategory"])

# df_historical = df_historical.join(category, rsuffix="_category")
# df_historical.drop("ListingCategory", axis=1, inplace=True)

# df_historical.info()


# In[ ]:


# employment = pd.get_dummies(df_historical["EmploymentStatus"])

# df_historical = df_historical.join(employment, rsuffix="_employmentstatus")
# df_historical.drop("EmploymentStatus", axis=1, inplace=True)

# df_historical.info()


# In[ ]:


# state_defaults = df_historical.groupby("BorrowerState")["LoanStatus"].mean()

# vlow_risk = sorted(state_defaults)[51]
# low_risk = sorted(state_defaults)[40]
# mid_risk = sorted(state_defaults)[29]
# high_risk = sorted(state_defaults)[19]
# vhigh_risk = sorted(state_defaults)[9]

# new_geography = {}

# for state in state_defaults.index:
#     if high_risk > state_defaults[state]:
#         v = "StateVeryHighRisk"
#     elif mid_risk > state_defaults[state] >= high_risk:
#         v = "StateHighRisk"
#     elif low_risk > state_defaults[state] >= mid_risk:
#         v = "StateMidRisk"
#     elif vlow_risk > state_defaults[state] >= low_risk:
#         v = "StateLowRisk"
#     else:
#         v = "StateVeryLowRisk"
#     new_geography[state] = v

# df_historical.replace(to_replace={"BorrowerState": new_geography}, inplace=True)
                               
# df_historical["BorrowerState"][:10]


# In[ ]:


# state = pd.get_dummies(df_historical["BorrowerState"])

# df_historical = df_historical.join(state, rsuffix="_state")
# df_historical.drop("BorrowerState", axis=1, inplace=True)

# df_historical.info()


# ## Export data

# In[ ]:


df_historical.drop(["LoanOriginationQuarter","ListingKey"],axis=1,inplace=True)


# In[ ]:


df_historical.shape


# In[ ]:


df_historical.head()


# In[ ]:


df_historical.to_csv("prosperLoans_default_filtV1.csv.gz",index=False,compression="gzip")

