#!/usr/bin/env python
# coding: utf-8

# Scenario: You are a Data Scientist working for a consulting firm. One of your colleagues from the Auditing department has asked you to help them assess the financial statement of organisation X.
# 
# You have been supplied with two lists of data: monthly revenue and montly expenses for the financial year in question. Your task is to calculate the following financial metrics:
# 
# profit for each month
# profit after tax for each month (the tax rate is 30%)
# profit margin for each month - equals to profit after tax divided by revenue
# good months - where the profit after tax was greater than the mean for the year
# bad months - where the profit after tax was less than the mean for the year
# the best month - where the profit after tax was max for the year
# the worst month - where the profit after tax was min for the year
# All results need to be presented as lists
# 
# Results for dollar values need to be calculated with \$0.01 precision, but need to be presented in Units of $1,000 (i.e. lk) with no decimal points.
# 
# Results for the profit margin ratio need to be presents in units of % with no decimal points.
# 
# Note: Your colleague has warned you that it is okay for tax for a given month to be negative (in accounting terms, negative tax translates into a deferred tax asset).
# 
# *Citation & Reference: www.superdatascience.com*

# In[ ]:


#Data 
revenue = [14574.49, 7606.46, 8611.41, 9175.41, 8058.65, 8105.44, 11496.28, 9766.09, 10305.32, 14379.96, 10713.97, 15433.50]
expenses = [12051.82, 5695.07, 12319.20, 12089.72, 8658.57, 840.20, 3285.73, 5821.12, 6976.93, 16618.61, 10054.37, 3803.96]

#importing libraries required
import numpy as np

#Solution
profit = []
netProfit = []
profitMargin = []
goodMonths = []
badMonths = []

for i in range(len(revenue)):
    profit.append(round(revenue[i] - expenses[i],2))
    netProfit.append(round((profit[i] - profit[i]*0.30),2))
    profitMargin.append(round((netProfit[i]/revenue[i])*100))
    
meanOfYear = np.mean(netProfit)
maxOfYear = np.max(netProfit)
minOfYear = np.min(netProfit)
bestMonth = []
worstMonth = []

for i in range(len(netProfit)):
    if netProfit[i] > meanOfYear:
        goodMonths.append(i+1)
    elif netProfit[i] < meanOfYear:
        badMonths.append(i+1)
    if netProfit[i] == np.max(netProfit) :
        bestMonth = i+1
    elif netProfit[i] == np.min(netProfit) :
        worstMonth = i+1

print("profits for each month:\n", profit, "\n")
print("profits after tax for each month:\n", netProfit, "\n")
print("profit margin for each month:\n", profitMargin, "\n")
print("good months:\n", goodMonths,"\n")
print("bad months:\n", badMonths,"\n")
print("best month:\n", bestMonth,"\n")
print("worst month:\n", worstMonth,"\n")


# In[ ]:




