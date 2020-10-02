#!/usr/bin/env python
# coding: utf-8

# A business organisation developed a new product and decided to market this to existing customers of the organization. At the first stage, they decided to contact a sample of the customers. This promotional program is completed and the response information along with profile of the customers is available in the dataset. The organization is interested in using this information to select the best customers from the rest of the customer pool so that promotion cost can be reduced. 
# 
# The Analytics approach is to build a model using the available information from the completed campaign. This model will be applied to the target datset (the customers not contacted) to select the customers so that profit is maximised. 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/promoted.csv")
df.head(3)


# Following are the variable definitions:-
# 
# <br> resp:	Responded or not (1=yes, 0=no)
# <br> card_tenure:	Card Tenure in months
# <br> risk_score:	Risk Score
# <br> num_promoted:	Number of times customer was contacted earlier for the same product
# <br> avg_bal:	Average balance
# <br> geo_group:	Geographical Group (NW W, S, E, SE)
# <br> res_type:	Residence Type (SI=single family, CO=cooperative, CN=condominium, RE=rental, TO=townhouse)

# In[ ]:


df.info()


# In[ ]:


df.describe()


# Looks like missing values are the main issue.  lets start with cleaning the dataset. 
# 
# In the case of numerical variiables, let's replace missings with mean since the proportion of missing is less.

# In[ ]:


df['card_tenure'].fillna(df['card_tenure'].mean(), inplace=True)
df['avg_bal'].fillna(df['avg_bal'].mean(), inplace=True)
df.describe()


# let' now tackle categorical variables

# In[ ]:


df['geo_group'].value_counts(dropna=False)


# In[ ]:


df['res_type'].value_counts(dropna=False)


# Since the proportion of missing is not large, we will merge missing with largest category

# In[ ]:


df['geo_group'].fillna('E', inplace=True)
df['res_type'].fillna('CO', inplace=True)


# In[ ]:


df.info()


# Let's create dummy varables to handle categorical variables.

# In[ ]:


dfg=pd.get_dummies(df['geo_group'], prefix='geo', drop_first=True)
dfr=pd.get_dummies(df['res_type'], prefix='res', drop_first=True)
dfrg=df.join([dfg,dfr])
dfrg.drop(['geo_group', 'res_type'], axis=1, inplace=True)
df=dfrg
df.head(3)


# In[ ]:


y=df['resp']
X=df.iloc[:, 2:]


# In[ ]:


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=11)
#X_train.head(3)


# Let's calculate the proportion of responders

# In[ ]:


y_train.mean()


# Multicollinearity is checked using VIF

# In[ ]:


import statsmodels.api as sm
from pandas.core import datetools

xc = sm.add_constant(X_train)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() 
vif["vif"] = [variance_inflation_factor(xc.values, i) for i in range(xc.shape[1])]
vif["features"] = xc.columns
vif


# The result indicate existence of multicollinearity. However, we will not drop variables with large VIF straightaway. Variables with lower significance level will be removed first and if VIF is still present it will be dropped.

# In[ ]:


import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())


# Variables are dropped one by one by testing for VIF and significance level (the code below shows the final look at the end of the process)

# In[ ]:


y=df['resp']
X=df.iloc[:, 2:]

X.drop(['res_CO','res_RE', 'res_TO','geo_N', 'geo_W','num_promoted', 'geo_SE', 'res_SI'], axis=1, inplace=True)

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=11)


import statsmodels.api as sm
from pandas.core import datetools

xc = sm.add_constant(X_train)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() 
vif["vif"] = [variance_inflation_factor(xc.values, i) for i in range(xc.shape[1])]
vif["features"] = xc.columns
print(vif)

import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())


# Let's test the model using the Test dataset

# In[ ]:


import statsmodels.api as sm
logit_model=sm.Logit(y_test,X_test)
result=logit_model.fit()
print(result.summary())


# Comparing the coefficients of the Test and Train model, we can conclude that the model is validated.
# 
# Following conditions were checked.
# 		- coeffiecints should be significant
# 		- sign of the coefficients should be same
# 		- absolute difference between the coefficients should be less than 25%

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# The confusion matrix shows that the model is not predicting responders at all. This is due to the choice of cutoff probability (0.5). Since the model is highly unbalanced, estimated probability is less than 0.5 for all cases resuting in zero correct prediction for responders. 

# ### KS chart is one of the most commonly used performance measure. let's draw it

# Let's calculate the predicted probability of default

# In[ ]:


prb=logreg.predict_proba(X_train)[:,1]
type(prb)


# We will bring predicted probability and response information together.

# In[ ]:


#pd.DataFrame(prb, y_train, columns=[('prob'), ('y')])
df = pd.DataFrame({'prob':prb, 'y':y_train})
#df


# In[ ]:


df=df.sort_values('prob', ascending=False)
df = df.reset_index()
df['New_ID'] = df.index
df.head()


# In[ ]:


df['cum_y'] = df.y.cumsum()
df['cumy_perc'] = 100*df.cum_y/df.y.sum()
df['pop_perc'] = 100*df.New_ID/df.New_ID.max()
df['ks_value']=df['cumy_perc'] - df['pop_perc']
ksmax=df['ks_value'].max()
df.head(3)
print(ksmax)
max_pop =df.loc[df['ks_value'] == df['ks_value'].max(), 'pop_perc'].item()
print(max_pop)


# In[ ]:


df.tail(3)


# In[ ]:


plt.plot(df.pop_perc, df.cumy_perc, color='g')
plt.plot(df.pop_perc, df.pop_perc, color='orange')
plt.xlabel('Percent Population')
plt.ylabel('Percent Responded')
#plt.title('KS Lift Chart')
plt.text(40, 30, 'KSmax value is %d at %d percent' %(ksmax, max_pop))
plt.title('KS Lift Chart')
plt.show()


# The chart shows that maximum separation between model curve and no model is 21 and it is at 47 percent of the population.

# The Model is built and evaluated. Now let's apply this for selecting the best customers from the target group. this would involve following steps.
# 
#     - decide the cutoff for selecting customers (this is based on the profitability analysis. We will choose the cutoff such that profit is maximised.
#     - Read target data
#     - apply same data cleaning approach adopted in the model building stage
#     - score the data (use the developed model to estimate probability of response for each customer in the target set)
#     - apply the cutoff and create the contact list.
#     

# Let's assume that the revenue from a responded customer is 3200 and cost of contacting is 200. 
# We will calculate profit and plot it.

# In[ ]:


revenue=3200
cost=200
df['profit'] = df['cum_y']*3200 - df['New_ID']*200
max_pop =df.loc[df['profit'] == df['profit'].max(), 'pop_perc'].item()
prmax=df['profit'].max()
df.head(3)


# In[ ]:


print(max_pop)


# Let's look at the maximum profit that can be achieved

# In[ ]:


df['profit'].max()


# Let's plot the profit against the proportion contacted.

# In[ ]:


plt.plot(df.pop_perc, df.profit, color='g')
#plt.plot(df.pop_perc, df.pop_perc, color='orange')
plt.xlabel('Percent Population')
plt.ylabel('Profit')
#plt.title('KS Lift Chart')
plt.text(40, 30, 'Max profit is %d at %d percent' %(prmax, max_pop))
plt.title('Profit Curve')
plt.show()


# This analysis suggest that we should select top 47% of the target group to maximise the profit. Now let's read taget group, clean and score it.

# In[ ]:


dft = pd.read_excel("D:/REGI/1. Analytic Text Book/zSolutions/Chapter-8 Building Binary Models/Data/DSCH08LOGIRESW.xlsx", sheet_name="target")


# In[ ]:


dft.head(3)


# In[ ]:


dft.describe()


# In[ ]:


dft.info()


# We will adopt the same claening approach of the modeling data.

# In[ ]:


dft['card_tenure'].fillna(dft['card_tenure'].mean(), inplace=True)
dft['avg_bal'].fillna(dft['avg_bal'].mean(), inplace=True)
dft.describe()


# let' now tackle categorical variables

# In[ ]:


dft['geo_group'].value_counts(dropna=False)


# In[ ]:


dft['res_type'].value_counts(dropna=False)


# Since the proportion of missing is not too large, we will merge missing with largest category

# In[ ]:


dft['geo_group'].fillna('E', inplace=True)
dft['res_type'].fillna('CO', inplace=True)


# In[ ]:


dft.info()


# Let's create dummy variables to model categorical variables

# In[ ]:


dfg=pd.get_dummies(dft['geo_group'], prefix='geo', drop_first=True)
dfr=pd.get_dummies(dft['res_type'], prefix='res', drop_first=True)
dfrg=dft.join([dfg,dfr])
dfrg.drop(['geo_group', 'res_type'], axis=1, inplace=True)
dft=dfrg
dft.head(3)


# In[ ]:


#X=dft.iloc[:, 1:]
Xt=dft.iloc[:, np.r_[1:3,4]]
Xt.head(3)


# In[ ]:


Xt.describe()


# Let's apply the model and calculate the response probability of all customers in the Target dataset

# In[ ]:


cid=dft['customer_id']
prb=logreg.predict_proba(Xt)[:,1]
tgt=pd.DataFrame({'resp_prob':prb, 'customer_id':cid })
tgt.sort_values('resp_prob', ascending=False, inplace=True)
tgt['tgt_num'] = range(len(tgt))
tgt['tgt_perc'] = 100*tgt.tgt_num/(tgt.tgt_num.max())
tgt.head(3)


# We will apply the optimal cutoff estimated (max_pop).

# In[ ]:


target=tgt[tgt['tgt_perc']<=max_pop]
target.head()
target.info()


# In[ ]:




