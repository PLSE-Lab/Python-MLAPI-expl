#!/usr/bin/env python
# coding: utf-8

# 
# # Customer Attrition
# 
# Customer attrition, also known as customer churn, customer turnover, or customer defection, is the loss of clients or customers.
# 
# Telephone service companies, Internet service providers, pay TV companies, insurance firms, and alarm monitoring services, often use customer attrition analysis and customer attrition rates as one of their key business metrics because the cost of retaining an existing customer is far less than acquiring a new one. Companies from these sectors often have customer service branches which attempt to win back defecting clients, because recovered long-term customers can be worth much more to a company than newly recruited clients.
# 
# Companies usually make a distinction between voluntary churn and involuntary churn. Voluntary churn occurs due to a decision by the customer to switch to another company or service provider, involuntary churn occurs due to circumstances such as a customer's relocation to a long-term care facility, death, or the relocation to a distant location. In most applications, involuntary reasons for churn are excluded from the analytical models. Analysts tend to concentrate on voluntary churn, because it typically occurs due to factors of the company-customer relationship which companies control, such as how billing interactions are handled or how after-sales help is provided.
# 
# predictive analytics use churn prediction models that predict customer churn by assessing their propensity of risk to churn. Since these models generate a small prioritized list of potential defectors, they are effective at focusing customer retention marketing programs on the subset of the customer base who are most vulnerable to churn.
# 

# ### Importing the data

# In[ ]:


# Library for DataFrame and Data manipulation
import pandas as pd
# Matrix operations and statistical functions
import numpy as np
# Plotting and dependency for seaborn
import matplotlib.pyplot as plt
# Graphs and chart used in this notebook
import seaborn as sns
# to convert categorical values into numerical value
from sklearn.preprocessing import LabelEncoder
# these imports are self explanatory
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ### Understanding the data

# In[ ]:


# Reading Data from csv file
data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[ ]:


data.head()


# In[ ]:


# Describe some stats of numerical features apperantly there should be  numerical features
data.describe()


# In[ ]:


# Describe data type of the features , or the type of value they contain
data.info()


# In[ ]:


# There are some missing values in 'TotalCharges' but insted of representing it by 'NAN' it represents it by blank space
data[data['TotalCharges']==' ']


# ### Data Manipulation

# In[ ]:


# extracting index of rows where TotalCharges has a white space 
ws = data[data['TotalCharges']==' '].index
# removing all those rows whose index have been extracted just now
data.drop(ws,axis=0,inplace=True)
# converting 'TotalCharges' data type from object to float, because it contains real values
data['TotalCharges'] = data.TotalCharges.astype('float')


# In[ ]:


# standardization , all values will me subtracted by column's mean and divided by cloumn's standard deviation
def standardized(data,col):
    mean = data[col].mean()
    std = data[col].std()
    data[col] = (data[col]-mean)/std
    


# In[ ]:


# this function is to plot countplot, since majority of features are categorical, i would stick to countplot most of the time
def plot_comparison(col,val,some=None):
    sub_data = data[data[col]==val]
    value = sub_data[sub_data['Churn']=='Yes']
    print('In '+str(col)+' = '+str(val)+' , it is {:.2f} % likely that customer will leave'.format((len(value)/len(sub_data)*100)))
    sns.countplot(sub_data['Churn'],palette="bright",ax=some)


# ### Data Visualization

# In[ ]:


data


# # Let's discuss the effect of some of the variables, that may help in churn prediction 

# ## 0. Churn 
# **Target variable or something we have to predict about unkown customers**

# In[ ]:


sns.countplot(data['Churn'])
print(len(data[data['Churn']=='Yes'])/len(data)*100)


# **Almost 26% of customers churn or leave the telecome company as a result of cut-throat competition**

# ## 1. CustomerID
# **This variable doesn't give any extra information about the customer or about churning, so the best option is to get rid off this variable**

# In[ ]:


data = data.drop('customerID',axis=1)


# ## 2. Gender
# **we can check whether males are more potential churner than females**

# In[ ]:



fig,ax1 = plt.subplots(figsize=(12,6))
sns.countplot(x='gender',data=data,hue='Churn')


# **As we can observe that the rate of churn is almost equal in Male and Female so gender doesn't seems to give any singnificant information about target variable**

# ## 3. SeniorCitizen
# **a. How many senior citizen dataset have**

# In[ ]:


sns.countplot(data['SeniorCitizen'])


# **b. Among Senior and non-senior citezen who is churning more frequently**

# In[ ]:


plot_comparison('SeniorCitizen',1)


# In[ ]:


plot_comparison('SeniorCitizen',0)


# In[ ]:


# no. of non-senior citizen churning
data['SeniorCitizen'].value_counts()[0]*23/100


# In[ ]:


# no. of senior citizen churning
data['SeniorCitizen'].value_counts()[1]*45/100


# **It is quite clear from above 2 countplots that Senior citzen are more likely to leave company,but a non senior citizen's 23% is higher in no.s.**

# ## 4. Partner
# **Pie plot is to describe overall count**

# In[ ]:


fig,ax1 = plt.subplots(1,3,figsize=(12,4))
# A pie plot to show the share of Yes and No 
data['Partner'].value_counts().plot(kind='pie')
# first plot from right shows no. of Potential Churners among Partner = 'Yes' category 
plot_comparison('Partner','Yes',ax1[0])
# plot in middle shows no. of Potential Churners among Partner = 'No' category 
plot_comparison('Partner','No',ax1[1])


# **Since, no. of Partners and non- partners are almost equal, We can conclude that Non-partners are more frequent churners**

# ## 5. Dependent

# In[ ]:


fig,ax1 = plt.subplots(1,3,figsize=(12,4))
data['Dependents'].value_counts().plot(kind='pie')
plot_comparison('Dependents','Yes',ax1[0])
plot_comparison('Dependents','No',ax1[1])


# In[ ]:


# No. of Non-dependent churning
data['Dependents'].value_counts()[0]*31.28/100


# **Need less to say Non-dependent is a huge no. of churners**

# In[ ]:


sns.catplot(hue='Partner',x='Dependents',col='Churn',kind="count",data=data)


# **So, far Non-dependent non-partners are dangerous category**

# ## 6. tenure
# **a. it is numerical variable and if i am not wrong it represents, no. of years an individual using company's service**

# In[ ]:


#distribution plot
sns.distplot(data['tenure'],rug=False,hist=False)


# **This gives us some info about how data is spreaded**

# In[ ]:


# to check mean,standard deviation,Qurtile and minimum maximum value in this feature 
data['tenure'].describe()


# In[ ]:


sns.boxplot(y=data['tenure'],x=data['Churn'])


# **If tenure is less or extremely high the chances of churning is more and also
# As you can see in the above boxplot, there are few outliers in Churn = yes category.they can be considered Potential Outliers** 

# In[ ]:


# all rows where Churn=='Yes'
churn = data[data['Churn']=='Yes'] 
# Finding first and third quartile
Q1,Q3 = churn['tenure'].quantile([.25,.75])
# Inter Quartile range
IQR = Q3-Q1
# all the values greater than Q3+1.5*IQR and less than Q1-1.5*IQR are considered outliers 
outliers = Q3+1.5*IQR
outliers


# In[ ]:


data['tenure'].max()


# **Outlier is turned out to be greater than 69.5 but tenure has maximum value as 72. I guess, we can afford to have this value so we'll  not remove them.**

# In[ ]:


# A scatter plot between every Numerical variable in the dataset
sns.pairplot(data)


# **There are 4 numerical features, where SeniorCitizen will be considered as Categorical feature becuase it consists of 0 and 1 only which represents 'No' and 'Yes' respectively.There is not quite visible relation but there is some correlation between TotalCharges and tenure and between MonthlyCharges and TotalCharges.** 

# In[ ]:


data


# ## 7. PhoneService

# In[ ]:


fig,ax1 = plt.subplots(1,3,figsize=(12,4))
plot_comparison('PhoneService','Yes',ax1[1])
plot_comparison('PhoneService','No',ax1[2])
sns.countplot(data['PhoneService'],ax=ax1[0])


# **However, No. of customers who have phone service are more, this fact doesn't seems to help either**

# ## 8. MultipleLines

# In[ ]:


data['MultipleLines'] = data['MultipleLines'].map(lambda x: 'Yes' if x=='Yes' else 'No')

fig,ax1 = plt.subplots(1,3,figsize=(12,4))
plot_comparison('MultipleLines','Yes',ax1[1])
plot_comparison('MultipleLines','No',ax1[2])

sns.countplot(data['MultipleLines'],ax=ax1[0])


# **Even after having multiple Lines of phones, people quiting more. since difference between two classes is less we won't consider this feature for model building**

# **Chi square test between MultipleLines and PhoneService**

# In[ ]:


# Preparing Contingency table for Chi_square test 
yes = data[data['MultipleLines']=='Yes']
no = data[data['MultipleLines']=='No']
yyes = len(yes[yes['PhoneService']=='Yes'])
yno = len(yes[yes['PhoneService']=='No'])
nyes = len(no[no['PhoneService']=='Yes'])
nno = len(no[no['PhoneService']=='No'])
table = [[yyes,yno],[nyes,nno]]


# In[ ]:


# H0 : two features are dependent
# H1 : two features are Independent
from scipy.stats import chi2_contingency
from scipy.stats import chi2
# Observed Frequency table
print('Table :',table)

stat, p, dof, expected = chi2_contingency(table)
# (nrows-1)*(ncols-1) in our case (2-1)*(2-1) = 1
print('degree of Freedom=%d' % dof)
# Expeceted Frequency table
print("Expected :",expected)

prob = 0.95
# tabulated value of chi-square at 5% of significance 
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# **We know Chi-Square is not valid becuase every entery into contingency table must be greater than 5 but we also know that MultipleLines are highly dependent on PhoneService.**

# ## 9. InternetService

# In[ ]:


data['InternetService'].value_counts()


# In[ ]:


fig,ax1 = plt.subplots(2,2,figsize=(12,10))
sns.countplot(data['InternetService'],ax=ax1[0,0])
plot_comparison('InternetService','DSL',ax1[0,1])
plot_comparison('InternetService','Fiber optic',ax1[1,0])
plot_comparison('InternetService','No',ax1[1,1])


# **DSL is an outdated technology, customer of this category might consider switching company but fibre optic is latest technology and if customer of this category churning more than anyone else this could be an alarming situation for telocome company.**

# In[ ]:


sns.catplot(hue='InternetService',x='PhoneService',col='Churn',kind="count",data=data)


# **Company's DSL service is fine It's Phone Service and Fibre optic service that company needs to work on**
# 

# ## 10. OnlineSecurity

# In[ ]:


fig,ax1 = plt.subplots(2,2,figsize=(12,10))
sns.countplot(data['OnlineSecurity'],ax=ax1[0,0])
plot_comparison('OnlineSecurity','Yes',ax1[0,1])
plot_comparison('OnlineSecurity','No',ax1[1,0])
plot_comparison('OnlineSecurity','No internet service',ax1[1,1])


# **'No internet service' can also be viewed as a 'No' which adds upto 49% in other words customers are using Phone services only**

# ## 11. OnlineBackup

# In[ ]:


fig,ax1 = plt.subplots(2,2,figsize=(12,10))
sns.countplot(data['OnlineBackup'],ax=ax1[0,0])
plot_comparison('OnlineBackup','Yes',ax1[0,1])
plot_comparison('OnlineBackup','No',ax1[1,0])
plot_comparison('OnlineBackup','No internet service',ax1[1,1])


# **Online Backup service is worse than security service**

# ## 12. DeviceProtection

# In[ ]:


fig,ax1 = plt.subplots(2,2,figsize=(12,10))
sns.countplot(data['DeviceProtection'],ax=ax1[0,0])
plot_comparison('DeviceProtection','Yes',ax1[0,1])
plot_comparison('DeviceProtection','No',ax1[1,0])
plot_comparison('DeviceProtection','No internet service',ax1[1,1])


# **we can see the similar trend in all internet related services**

# ## 13. Contract

# In[ ]:



fig,ax1 = plt.subplots(2,2,figsize=(12,10))
sns.countplot(data['Contract'],ax=ax1[0,0])
plot_comparison('Contract','Month-to-month',ax1[0,1])
plot_comparison('Contract','Two year',ax1[1,0])
plot_comparison('Contract','One year',ax1[1,1])


# **No need to worry about 2 years contractors, they are going to stay, which is pretty obvious**  

# ## 14. PaperlessBilling

# In[ ]:



fig,ax1 = plt.subplots(1,3,figsize=(12,4))
plot_comparison('PaperlessBilling','Yes',ax1[1])
plot_comparison('PaperlessBilling','No',ax1[2])

sns.countplot(data['PaperlessBilling'],ax=ax1[0])


# In[ ]:


fig,ax1 = plt.subplots(1,2,figsize=(12,5))
sns.boxplot(x='PaperlessBilling',y='MonthlyCharges',data=data,ax=ax1[0])
sns.boxplot(x='PaperlessBilling',y='TotalCharges',data=data,ax=ax1[1])


# **It may be just a coincidence but those who are opting PaperlessBilling their MonthlyCharges and TotalCharges are high**

# ## 15. PaymentMethod

# In[ ]:


f,ax = plt.subplots(figsize=(12,6))
sns.countplot(data['PaymentMethod'])


# In[ ]:



fig,ax1 = plt.subplots(2,2,figsize=(12,10))
plot_comparison('PaymentMethod','Electronic check',ax1[0,0])
plot_comparison('PaymentMethod','Mailed check',ax1[0,1])
plot_comparison('PaymentMethod','Bank transfer (automatic)',ax1[1,0])
plot_comparison('PaymentMethod','Credit card (automatic)',ax1[1,1])


# ## 16. MonthlyCharges

# In[ ]:


sns.distplot(data['MonthlyCharges'])


# In[ ]:


sns.boxplot(data['MonthlyCharges'],data['Churn'])


# **Needless to say MonthlyCharges are high for churners**

# In[ ]:


sns.distplot(data['TotalCharges'])


# In[ ]:


sns.boxplot(y=data['TotalCharges'],x=data['Churn'])


# **Surprisingly, Total charges shows a very different behaviour than monthly charges probably because it is right skewed**

# ### Conclusion: 
# 
# 1. Customers are not gender bias
# 2. Senior Citizens, non-partner, Non-dependent are more frequent Churners
# 3. If tenure of customer is less or extremely high, chances of their opting out are higer
# 4. Phone Service and having Multiple lines has nothing to do with Churn prediction(acording to data)
# 5. Online Backup service is worse than Online security
# 6. Long term contract can help retaining customers
# 7. those who are opting paperless Billing they might have having some problem
# 8. payment method == Electronic check has high rate of churners 

# ### Implement Machine Learning Models

# In[ ]:


# don't want to alter original dataset
copy = data.copy(deep=True)


# In[ ]:


# label encoder to change categorical values into Numerical values
le = LabelEncoder()
for col in copy.columns:
    #if data type of column is object then and only apply label encoder
    if copy[col].dtypes =='object':
        copy[col] = le.fit_transform(copy[col])
        
# seperating Independent and dependent features
X = copy.drop('Churn',axis=1)
y = copy['Churn']


# In[ ]:


# to select k no. of best features from the available list of features 
from sklearn.feature_selection import SelectKBest
#using chi-square test
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=12)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[ ]:


#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns


# In[ ]:


print(featureScores.nlargest(15,'Score'))  #print 10 best features


# In[ ]:


# will build model based on these 12 features becuase there chi-square is very high
features = ['TotalCharges','tenure','MonthlyCharges','Contract','OnlineSecurity','TechSupport','OnlineBackup','DeviceProtection',
           'SeniorCitizen','Dependents','PaperlessBilling','Partner']


# In[ ]:


# Standardization 
X = copy[features]
y = copy['Churn']
standardized(X,'tenure')
standardized(X,'TotalCharges')
standardized(X,'MonthlyCharges')


# In[ ]:





# In[ ]:


X_train, X_test, Y_train, Y_test= train_test_split(X, y, test_size=0.30, random_state=1, shuffle=True)


# In[ ]:





# ### Model Evaluation

# In[ ]:



# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
#models.append(())
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[ ]:


#Logistic Regression on test data
lr = LogisticRegression()
lr.fit(X_train,Y_train)
pred = lr.predict(X_test)
acc = accuracy_score(Y_test,pred)
acc


# In[ ]:


# LinearDiscriminantAnalysis on test data
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)
pred = lda.predict(X_test)
acc = accuracy_score(Y_test,pred)
acc


# In[ ]:


# Support Vector machine on test data
svc = SVC(gamma='auto')
svc.fit(X_train,Y_train)
pred = svc.predict(X_test)
acc = accuracy_score(Y_test,pred)
acc


# **Linear regression, Linear Discriminant Analysis and Support Vector Machine all three are giving almost same accuracy for test data**

# # If you enjoy this notebook, then upvote and feel free to provide me with your feedback !!!

# In[ ]:





# In[ ]:




