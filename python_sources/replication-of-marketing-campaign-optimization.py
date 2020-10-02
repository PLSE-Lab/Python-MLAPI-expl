#!/usr/bin/env python
# coding: utf-8

# # Promoting financial products to bank customers
# 
# This kernel is a replication of [CPLEX solution](https://github.com/IBMDecisionOptimization/DOforDSX-MarketingCampaigns-example/blob/master/jupyter/MarketingCampaigns.ipynb) with an open-source optimization package OR-Tools.
# 
# In 2016, a retail bank sold several products (mortgage account, savings account, and pension account) to its customers. It kept a record of all historical data, and this data is available for analysis and reuse. Following a merger in 2017, the bank has new customers and wants to start some marketing campaigns.
# 
# The budget for the campaigns is limited. The bank wants to contact a customer and propose only one product.
# 
# The marketing department needs to decide:
# 
# * Who should be contacted?
# * Which product should be proposed? Proposing too many products is counter productive, so only one product per customer contact.
# * How will a customer be contacted? There are different ways, with different costs and efficiency.
# * How can they optimally use the limited budget?
# * Will such campaigns be profitable?
# 
# ## Predictive and prescriptive workflow
# 
# From the historical data, we can train a machine learning product-based classifier on customer profile (age, income, account level, ...) to predict whether a customer would subscribe to a mortgage, savings, or pension account.
# 
# We can apply this predictive model to the new customers data to predict for each new customer what they will buy.
# On this new data, we decide which offers are proposed. Which product is offered to which customer through which channel:
# a. with a greedy algorithm that reproduces what a human being would do
# b. using an optimization model wih IBM Decision Optimization.
# The solutions can be displayed, compared, and analyzed.
# 
# Table of contents:
# 
# * Understand the historical data
# * Predict the 2017 customer behavior
# * Get business decisions on the 2017 data
# * Conclusion on the decision making
# 
# This notebook takes some time to run because multiple optimization models are solved and compared in the part dedicated to what-if analysis. The time it takes depends on your subscription type, which determines what optimization service configuration is used.

# # How decision optimization can help
# 
# Prescriptive analytics (decision optimization) technology recommends actions that are based on desired outcomes. It takes into account specific scenarios, resources, and knowledge of past and current events. With this insight, your organization can make better decisions and have greater control of business outcomes.
# 
# Prescriptive analytics is the next step on the path to insight-based actions. It creates value through synergy with predictive analytics, which analyzes data to predict future outcomes.
# 
# Prescriptive analytics takes that prediction to the next level by suggesting the optimal way to handle that future situation. Organizations gain a strong competitive advantage by acting quickly in dynamic conditions and making superior decisions in uncertain environments.
# 
# * Automate complex decisions and trade-offs to better manage your limited resources.
# * Take advantage of a future opportunity or mitigate a future risk.
# * Proactively update recommendations based on changing events.
# * Meet operational goals, increase customer loyalty, prevent threats and fraud, and optimize business processes.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

get_ipython().run_line_magic('matplotlib', 'inline')


# # Understand the historical data
# 
# Load 2016 historical data, analyze it visually, and train a classifier to predict 2017 sales.
# 
# ***Load the historical customer data with their purchases (Mortgage, Savings, and Pension).***

# In[ ]:


known_behaviors = pd.read_csv("https://raw.githubusercontent.com/vberaudi/utwt/master/known_behaviors2.csv")
known_behaviors.head()


# ***Check the 2016 customers***

# In[ ]:


a = known_behaviors[known_behaviors.Mortgage == 1]
b = known_behaviors[known_behaviors.Pension == 1]
c = known_behaviors[known_behaviors.Savings == 1]
print("Number of clients: %d" %len(known_behaviors))
print("Number of clients predicted to buy mortgage accounts: %d" %len(a))
print("Number of clients predicted to buy pension accounts: %d" %len(b))
print("Number of clients predicted to buy savings accounts: %d" %len(c))


# In[ ]:


known_behaviors["nb_products"] = known_behaviors.Mortgage + known_behaviors.Pension + known_behaviors.Savings


# In[ ]:


abc = known_behaviors[known_behaviors.nb_products > 1]
print("We have %d clients who bought several products" %len(abc))
abc = known_behaviors[known_behaviors.nb_products == 3]
print("We have %d clients who bought all the products" %len(abc))


# In[ ]:


products = ["Savings", "Mortgage", "Pension"]


# ***Do some visual analysis of the historical data***
# 
# It's possible to use pandas plotting capabilities, but it would require a new version of it. This Notebook relies on matplotlib as it is present everywhere.

# In[ ]:


def plot_cloud_points(df):
    figure = plt.figure(figsize=(20, 5))
    my_cm  = ListedColormap(['#bb0000', '#00FF00'])
    axes = {p : ('age', 'income') if p != "Mortgage"else ('members_in_household', 'loan_accounts') for p in products}
    for product in products:
        ax = plt.subplot(1, len(products), products.index(product)+1)
        ax.set_title(product)
        axe = axes[product]
        plt.xlabel(axe[0])
        plt.ylabel(axe[1])
        ax.scatter(df[axe[0]], df[axe[1]], c=df[product], cmap=my_cm, alpha=0.5)


# In the following visualization, you can see the behavior of the 2016 customers for the three products. The green color indicates that a customer bought a product; red indicates a customer did not buy a product. The depth of the color indicates the number of purchases or non-purchases.

# In[ ]:


plot_cloud_points(known_behaviors)


# ## Understanding the 2016 customers
# 
# We can see that:
# 
# * The greater a customer's income, the more likely it is s/he will buy a savings account.
# * The older a customer is, the more likely it is s/he will buy a pension account.
# * There is a correlation between the number of people in a customer's household, the number of loan accounts held by the customer, and the likelihood a customer buys a mortgage account. To see the correlation, look at the upper right and lower left corners of the mortgage chart.

# # Predict the 2017 customer behavior
# 
# **Create and train a simple machine-learning algorithm to predict what the new clients will buy.**

# In[ ]:


known_behaviors.columns


# In[ ]:


cols = ['age', 'income', 'members_in_household', 'loan_accounts']


# In[ ]:


X = known_behaviors[cols]
ys = [known_behaviors[p] for p in products]


# In[ ]:


X.head()


# We use a standard basic support gradient boosting algorithm to predict whether a customer might by product A, B, or C.

# In[ ]:


from sklearn import svm
from sklearn import ensemble


# In[ ]:


classifiers = []
for i,p in enumerate(products):
    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X, ys[i])
    classifiers.append(clf)


# ## New customer data and predictions
# 
# Load new customer data, predict behaviors using trained classifier, and do some visual analysis. We have all the characteristics of the new customers, as for the 2016 clients, but the new customers did not buy any product yet.
# 
# ***Load new customer data***

# In[ ]:


unknown_behaviors = pd.read_csv("https://raw.githubusercontent.com/vberaudi/utwt/master/unknown_behaviors.csv")


# In[ ]:


for c in unknown_behaviors.columns:
    assert c in known_behaviors.columns


# In[ ]:


to_predict = unknown_behaviors[cols]


# In[ ]:


print("Number of new customers: %d" %len(unknown_behaviors))


# ***Predict behaviors of the new customers***

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


predicted = [classifiers[i].predict(to_predict) for i in range(len(products))]
for i,p in enumerate(products):
    to_predict[p] = predicted[i]
to_predict["id"] = unknown_behaviors["customer_id"]


# ***Package new data with predictions for optimization***

# In[ ]:


offers = to_predict
offers.head()


# We will reset index, we will use index number.

# In[ ]:


offers = offers.rename_axis('index_nb').reset_index()


# ***Do some visual analysis of the predicted data***

# The predicted data has the same semantic as the base data, with even more clear frontiers:
# 
# * for savings, there is a clear frontier at $50K revenue.
# * for pension, there is a clear frontier at 55 years old customers.
# 
# The training data contains customers who bought more than one product, let's see our prediction

# In[ ]:


a = offers[offers.Mortgage == 1]
b = offers[offers.Pension == 1]
c = offers[offers.Savings == 1]
print("Number of new customers: %d" %len(offers))
print("Number of customers predicted to buy mortgages: %d" %len(a))
print("Number of customers predicted to buy pensions: %d" %len(b))
print("Number of customers predicted to buy savings: %d" %len(c))


# In[ ]:


to_predict["nb_products"] = to_predict.Mortgage + to_predict.Pension + to_predict.Savings

abc = to_predict[to_predict.nb_products > 1]
print("We predicted that %d clients would buy more than one product" %len(abc))
abc = to_predict[to_predict.nb_products == 3]
print("We predicted that %d clients would buy all three products" %len(abc))


# # Remarks on the prediction
# 
# The goal is to contact the customers to sell them only one product, so we cannot select all of them. This increases the complexity of the problem: we need to determine the best contact channel, but also need to select which product will be sold to a given customer.
# 
# It may be hard to compute this. In order to check, we will use two techniques:
# 
# * a greedy algorithm
# * OR-Tools, an open-source alternative for optimization(we have replicated CPLEX solution)

# # Get business decisions on the 2017 data
# 
# ## Assign campaigns to customers
# * We have predicted who will buy what in the list of new customers.
# * However, we do not have the budget to contact all of them. We have various contact channels with different costs and effectiveness.
# * Furthermore, if we contact somebody, we don't want to frustrate them by proposing multiple products; we want to propose only one product per customer.
# 
# ***Some input data for optimization***

# In[ ]:


# How much revenue is earned when selling each product
productValue = [200, 300, 400]
value_per_product = {products[i] : productValue[i] for i in range(len(products))}

# Total available budget
availableBudget = 25000

# For each channel, cost of making a marketing action and success factor
channels =  pd.DataFrame(data=[("gift", 20.0, 0.20), 
                               ("newsletter", 15.0, 0.05), 
                               ("seminar", 23.0, 0.30)], columns=["name", "cost", "factor"])

offersR = range(0, len(offers))
productsR = range(0, len(products))
channelsR = range(0, len(channels))


# **Using a greedy algorithm**
# 
# * We create a custom algorithm that ensures 10% of offers are made per channel by choosing the most promising per channel. The algorithm then continues to add offers until the budget is reached.

# In[ ]:


gsol = pd.DataFrame()
gsol['id'] = offers['id']

budget = 0
revenue = 0

for product in products:
    gsol[product] = 0

noffers = len(offers)

# ensure the 10% per channel by choosing the most promising per channel
for c in channelsR: #, channel in channels.iterrows():
    i = 0;
    while (i< ( noffers // 10 ) ):
        # find a possible offer in this channel for a customer not yet done
        added = False
        for o  in offersR:
            already = False
            for product in products:   
                if gsol.get_value(index=o, col=product) == 1:
                    already = True
                    break
            if already:
                continue
            possible = False
            possibleProduct = None
            for product in products:
                if offers.get_value(index=o, col=product) == 1:
                    possible = True
                    possibleProduct = product
                    break
            if not possible:
                continue
            #print "Assigning customer ", offers.get_value(index=o, col="id"), " with product ", product, " and channel ", channel['name']
            gsol.set_value(index=o, col=possibleProduct, value=1)
            i = i+1
            added = True
            budget = budget + channels.get_value(index=c, col="cost")
            revenue = revenue + channels.get_value(index=c, col="factor")*value_per_product[product]            
            break
        if not added:
            print("NOT FEASIBLE")
            break


# In[ ]:


# add more to complete budget       
while (True):
    added = False
    for c, channel in channels.iterrows():
        if (budget + channel.cost > availableBudget):
            continue
        # find a possible offer in this channel for a customer not yet done
        for o  in offersR:
            already = False
            for product in products:   
                if gsol.get_value(index=o, col=product) == 1:
                    already = True
                    break
            if already:
                continue
            possible = False
            possibleProduct = None
            for product in products:
                if offers.get_value(index=o, col=product) == 1:
                    possible = True
                    possibleProduct = product
                    break
            if not possible:
                continue
            #print "Assigning customer ", offers.get_value(index=o, col="id"), " with product ", product, " and channel ", channel['name']
            gsol.set_value(index=o, col=possibleProduct, value=1)
            i = i+1
            added = True
            budget = budget + channel.cost
            revenue = revenue + channel.factor*value_per_product[product]            
            break
    if not added:
        print("FINISH BUDGET")
        break
    
print(gsol.head())


# In[ ]:


a = gsol[gsol.Mortgage == 1]
b = gsol[gsol.Pension == 1]
c = gsol[gsol.Savings == 1]

abc = gsol[(gsol.Mortgage == 1) | (gsol.Pension == 1) | (gsol.Savings == 1)]

print("Number of clients: %d" %len(abc))
print("Numbers of Mortgage offers: %d" %len(a))
print("Numbers of Pension offers: %d" %len(b))
print("Numbers of Savings offers: %d" %len(c))
print("Total Budget Spent: %d" %budget)
print("Total revenue: %d" %revenue)


comp1_df = pd.DataFrame(data=[["Greedy", revenue, len(abc), len(a), len(b), len(c), budget]], columns=["Algorithm","Revenue","Number of clients","Mortgage offers","Pension offers","Savings offers","Budget Spent"])


# The greedy algorithm only gives a revenue of \$50.8K.
# 
# ## Replicating IBM Decision Optimization CPLEX Modeling with OR-Tools
# 
# Let's create the optimization model to select the best ways to contact customers and stay within the limited budget.

# In[ ]:


from __future__ import print_function
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver('SolveMCProblemMIP',
                           pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)


# ***Define the decision variables***
# 
# * The integer decision variables channelVars, represent whether or not a customer will be made an offer for a particular product via a particular channel.
# * The integer decision variable totaloffers represents the total number of offers made.
# * The continuous variable budgetSpent represents the total cost of the offers made.

# In[ ]:


channelVars = {}

# variables
for o in offersR:
    for p in productsR:
        for c in channelsR:
            channelVars[o,p,c] = solver.BoolVar('channelVars[%i,%i,%i]' % (o,p,c))


# ***Set up the constraints***
# 
# * Offer only one product per customer.
# * Compute the budget and set a maximum on it.
# * Compute the number of offers to be made.
# * Ensure at least 10% of offers are made via each channel.

# In[ ]:


# constraints
# At most 1 product is offered to each customer
for o in offersR:
    solver.Add(solver.Sum(channelVars[o,p,c] for p in productsR for c in channelsR) <=1)

# Do not exceed the budget
solver.Add(solver.Sum(channelVars[o,p,c]*channels.get_value(index=c, col="cost") 
                                           for o in offersR 
                                           for p in productsR 
                                           for c in channelsR)  <= availableBudget)

# At least 10% offers per channel
for c in channelsR:
    solver.Add(solver.Sum(channelVars[o,p,c] for p in productsR for o in offersR) >= len(offers) // 10)


# In[ ]:


print(f'Number of constraints : {solver.NumConstraints()}' )
print(f'Number of variables   : {solver.NumVariables()}')


# ***Express the objective***
# 
# We want to maximize expected revenue, so we take into account the predicted behavior of each customer for each product.

# In[ ]:


# objective 
obj = 0

for c in channelsR:
    for p in productsR:
        product=products[p]
        coef = channels.get_value(index=c, col="factor") * value_per_product[product]
        obj += solver.Sum(channelVars[o,p,c] * coef * offers.get_value(index=o, col=product) for o in offersR)

solver.Maximize(obj)


# ***Solve the Decision Optimization***

# In[ ]:


# time limit
#solver.set_time_limit = 100.0


# In[ ]:


sol = solver.Solve()


# In[ ]:


totaloffers = solver.Sum(channelVars[o,p,c] for o in offersR for p in productsR for c in channelsR)

budgetSpent = solver.Sum(channelVars[o,p,c]*channels.get_value(index=c, col="cost") 
                                           for o in offersR 
                                           for p in productsR 
                                           for c in channelsR)

print(f'Total offers : {totaloffers.solution_value()}')
print(f'Budget Spent : {budgetSpent.solution_value()}')

for c, n in zip(channelsR, list(channels.name)):
    channel_kpi = solver.Sum(channelVars[o,p,c] for p in productsR for o in offersR)
    print(f'{n} : {channel_kpi.solution_value()}')

for p, n in zip(productsR, products):
    product = products[p]
    product_kpi = solver.Sum(channelVars[o,p,c] for c in channelsR for o in offersR)
    print(f'{n} : {product_kpi.solution_value()}')

print(f'It has taken {solver.WallTime()} milliseconds to solve the optimization problem.')


# ***Allocation Products to Customers***

# In[ ]:


from itertools import product as prod

results = []

for o, p, c in prod(list(range(len(offers))),list(range(len(products))), list(range(len(channels)))):
    if channelVars[(o, p, c)].solution_value() > 0:
        #print(f'{o} : {products[p]}')
        results.append([o, products[p]])

results = pd.DataFrame(results, columns=['index_nb', 'product'])


# In[ ]:


results.head()


# In[ ]:


results['product'].value_counts()


# Whole dataset:

# In[ ]:


all_results = offers.merge(results, on='index_nb', how='inner')
all_results.head()

