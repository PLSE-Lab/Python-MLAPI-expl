#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


# # Statistics
# Descriptive Statistics and Inferential Statistics

# # Descriptive statistics:
# Mean, Variance, Standard deviation, quartiles, covariance, correlation

# ## Mean, Variance and Standard deviation:

# In[ ]:


a=np.random.randint(0,20,6)
a


# In[ ]:


pop_mean=a.sum()/len(a)
print('Population Mean:',pop_mean)
sam_mean=pop_mean
print('Sample Mean:',sam_mean)
sum_of_squares_of_deviations=0
for i in a:
    sum_of_squares_of_deviations=sum_of_squares_of_deviations+((i-pop_mean)**2)
pop_var=sum_of_squares_of_deviations/len(a)
pop_std=(pop_var)**0.5
print('Population Variance:',pop_var)
print('Population Standard deviation:',pop_std)
sam_var=sum_of_squares_of_deviations/(len(a)-1)
sam_std=(sam_var)**0.5
print('Sample Variance:',sam_var)
print('Sample Standard deviation:',sam_std)


# In[ ]:


#np.var() and np.std() give population variance and std
#use ddof=1 to get sample var and sample std.
np.var(a),np.std(a),np.var(a,ddof=1),np.std(a,ddof=1)


# ## 5 Point summary:
# min, Q1, Q2, Q3, max
# 
# iqr

# In[ ]:


arr1=np.random.randint(0,100,30)
print(arr1)
print('5 point summary:')
print('min:',np.percentile(arr1,0))
print('Q1:',np.percentile(arr1,25))
print('Q2:',np.percentile(arr1,50))
print('Q3:',np.percentile(arr1,75))
print('max:',np.percentile(arr1,100))
print()
print('iqr:',np.percentile(arr1,75)-np.percentile(arr1,25))
plt.boxplot(arr1)
plt.show()


# ## Covariance:

# In[ ]:


arr2=np.random.randint(0,100,30)
np.cov(arr1,arr2)


# In[ ]:


s=0
for i in range(len(arr1)):
    s=s+((arr1[i]-np.mean(arr1))*(arr2[i]-np.mean(arr2)))
s/(len(arr1)-1)


# ## Correlation:

# In[ ]:


print(np.corrcoef(arr1,arr2))
np.cov(arr1,arr2)[0,1]/(np.std(arr1,ddof=1)*np.std(arr2,ddof=1))


# # Discrete Probability Distributions:
# Binomial Distribution, Poisson Distribution and Hypergeometric Distribution

# ## Binomial Distribution:

# In[ ]:


binom=stats.binom.pmf(np.arange(10),10,0.05)
binom


# In[ ]:


binom_cum=stats.binom.cdf(np.arange(10),10,0.05)
binom_cum


# In[ ]:


plt.plot(np.arange(10),binom,'o-')
plt.plot(np.arange(10),binom_cum,'o-')
plt.show()


# ## Poission Distribution:

# In[ ]:


poiss=stats.poisson.pmf(np.arange(10),2.5)
poiss


# In[ ]:


poiss_cum=stats.poisson.cdf(np.arange(10),2.5)
poiss_cum


# In[ ]:


plt.plot(np.arange(10),poiss,'o-')
plt.plot(np.arange(10),poiss_cum,'o-')
plt.show()


# ## Hypergoemetric Distribution:

# In[ ]:


hgd=stats.hypergeom(100,10,20)
plt.plot(np.arange(20),hgd.pmf(np.arange(20)),'o-')
plt.plot(np.arange(20),hgd.cdf(np.arange(20)),'o-')
plt.show()


# # Continuous Probability Distributions:
# Normal Distribution, Geometric Distribution, Exponential Distribution

# ## Normal Distrinution:

# In[ ]:


# can give the mena and std as arguments, or can calculate z and pass z alone. work same either way.
print(stats.norm.cdf(90,68,12))
z=(90-68)/12
print(stats.norm.cdf(z))


# In[ ]:


plt.plot(np.arange(0,100,4),stats.norm.pdf(np.arange(0,100,4),68,12),'o-')


# In[ ]:


stats.norm.cdf(100,68,12)-stats.norm.cdf(50,68,12)


# In[ ]:


#confidence interval using stats.norm.interval
n=100
xavg=170
sigma=0.02
s= sigma / np.sqrt(n)
cil,ciu=stats.norm.interval(0.95,loc=xavg, scale=s)
print(cil,ciu)


# In[ ]:


#returns zscore value for the given alpha/2 value
stats.norm.ppf(0.025)


# In[ ]:


stats.norm.isf(0.9,100,20)


# In[ ]:


h=np.random.randint(0,100,30)
mean=np.mean(h)
sd=np.std(h,ddof=1)
print(mean,sd)


# In[ ]:


stats.norm.isf(0.9,mean,sd/np.sqrt(30))


# # One sample Test of Means:
# 

# ## t-test using norm.cdf:

# In[ ]:


ts=(2.8-3)/(0.6/(50**0.5))
pvalue=stats.norm.cdf(ts)
pvalue


# In[ ]:


ts=(573-520)/(124/6)
pvalue=1-stats.norm.cdf(ts)
pvalue


# In[ ]:


ts=(530-520)/(124/6)
pvalue=1-stats.norm.cdf(ts)
pvalue


# ## T-test:
# 
# *  parametric test
# *  only for normal or near normal distributions
# 
# 

# In[ ]:


from scipy.stats import ttest_1samp


# In[ ]:


np.random.seed(10)
population_ages1=stats.poisson.rvs(loc=18, mu=35, size=150000)
population_ages2=stats.poisson.rvs(loc=18, mu=10, size=100000)
population_ages=np.concatenate((population_ages1,population_ages2))


# In[ ]:


np.random.seed(10)
sample=np.random.choice(population_ages,1000)


# In[ ]:


sample.mean(),population_ages.mean()


# In[ ]:


sns.distplot(sample) # not a normal distribution


# In[ ]:


ttest_1samp(sample, 43.002372) # ttest is for normal distributions only, but here we are using it just for sake of example.


# pvalue is greater than 0.05, so we fail to reject the null hypothesis. that implies sample is the true representaion of the population.

# In[ ]:


arr=np.random.normal(2.8,0.6,50)


# In[ ]:


# for one tailed test using ttest_1samp we divide pvalue by 2 to get the pavalue for one tailed test since the area decreases to half.
ts, pval=ttest_1samp(arr,3)
pval=pval/2
print(pval)


# ##Shapiro Test:
# test for normality

# In[ ]:


from scipy.stats import shapiro


# In[ ]:


sns.distplot(sample)
plt.show()


# In[ ]:


shapiro(sample)


# pvalue less than 0.05 implies not normal

# In[ ]:


sns.distplot(arr)
plt.show()


# In[ ]:


shapiro(arr)


# pvalue greater than 0.05 hence normal distribution

# ## Wilcoxon test:
# *   Non-Parametric Test
# *   for not normal distributions
# *   more stringent test than ttest, hence pvalue is lower than ttest

# In[ ]:


from scipy.stats import wilcoxon


# In[ ]:


np.random.seed(10)
population_ages1=stats.poisson.rvs(loc=18, mu=35, size=150000)
population_ages2=stats.poisson.rvs(loc=18, mu=10, size=100000)
population_ages=np.concatenate((population_ages1,population_ages2))


# In[ ]:


np.random.seed(10)
sample=np.random.choice(population_ages,1000)


# In[ ]:


sample.mean(),population_ages.mean()


# In[ ]:


wilcoxon(sample-43.002372)


# #Two sample test of mean for unpaired samples(independent)

# In[ ]:


# energy expenditure in mJ and stature (0=obese, 1=lean)
energ = np.array([
[9.21, 0],
[7.53, 1],
[7.48, 1],
[8.08, 1],
[8.09, 1],
[10.15, 1],
[8.40, 1],
[10.88, 1],
[6.13, 1],
[7.90, 1],
[11.51, 0],
[12.79, 0],
[7.05, 1],
[11.85, 0],
[9.97, 0],
[7.48, 1],
[8.79, 0],
[9.69, 0],
[9.68, 0],
[7.58, 1],
[9.19, 0],
[8.11, 1]])


# In[ ]:


#creating samples
g1=energ[energ[:,1]==0][:,0]
g2=energ[energ[:,1]==1][:,0]


# ## Shapiro test
# 
# *   Test for normality
# *   P<0.05 implies sample not normal
# 
# 

# In[ ]:


#check for normality
shapiro(g1),shapiro(g2)


# g2 pvalue in shapiro test just less than 0.05 so we need to perform manwhitneyu. but lets do all tests (levene, bartlett, ttest_ind) just as an example.

# ## levene test:
# for testing equality of the variances of normal samples

# In[ ]:


from scipy.stats import levene


# In[ ]:


levene(g1,g2)


# pvalue>0.05 implies variances are equal but we have to perform bartlett ideally

# ## Two sample ttest_ind:
# 
# *   parametric t test for 2 samples
# *   performed when the samples pass both the shapiro and levene tests
# 
# 

# In[ ]:


from scipy.stats import ttest_ind


# In[ ]:


ttest_ind(g1,g2)


# pvalue<0.05 implies that the means are not equal (ideally we should perform mannwhitneyu as they one of the sample did not pass the shapiro test)

# ## Bartlett test
# for testing equality of variances when atleast one sample is not normal

# In[ ]:


from scipy.stats import bartlett


# In[ ]:


bartlett(g1,g2)


# pvalue greater than 0.05 implies variances are equal

# ## Mann-Whitney U Test:
# 
# 
# *   two sample non parametric test for equality of means
# *   performed when samples either fail shapiro or levene test
# 
# 

# In[ ]:


from scipy.stats import mannwhitneyu


# In[ ]:


mannwhitneyu(g1,g2)


# pvalue<0.05 hence the means of the samples are not equal

# # Two sample test of means for Paired samples

# In[ ]:


# pre and post-surgery energy intake
intake = np.array([
[5260, 3910],
[5470, 4220],
[5640, 3885],
[6180, 5160],
[6390, 5645],
[6515, 4680],
[6805, 5265],
[7515, 5975],
[7515, 6790],
[8230, 6900],
[8770, 7335],
])


# In[ ]:


#creating samples
g1=intake[:,0]
g2=intake[:,1]


# ## Checking parametric or non-parametric:

# In[ ]:


#shapiro test for normality
shapiro(g1),shapiro(g2)


# both the samples are normal

# In[ ]:


#levene test for quality of varicances (perform bartlett instead of levene if atleast one of the samples failed shapiro test)
levene(g1,g2)


# pvalue>0.05 implies variances are equal so we need to perform ttest_1samp(g1-g2), but we will perform wilcoxon too just for example.

# ## Two sample paired parametric ttest:

# In[ ]:


ttest_1samp(g1-g2,0)


# pvalue < 0.05 implies the means are not equal

# ## Two sample paired non-parametric ttest:(wilcoxon)

# In[ ]:


wilcoxon(g1,g2)


# pvalue<0.05 implies the means are not equal

# #Test of proportions

# ## z test

# In[ ]:


from statsmodels.stats.proportion import proportions_ztest


# In[ ]:


df=pd.read_csv("../input/learn-ml-datasets/HR.txt",sep='\t',index_col=0)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.Gender.value_counts()


# In[ ]:


ct=pd.crosstab(df.Attrition, df.Gender)
ct


# 1 is male and 2 is female

# ## One sample z test:

# In[ ]:


# lets compare the attrition proportion of the males to the population.
p1= 150/882
pp=237/1470
zdata=(p1-pp)/np.sqrt(pp*(1-pp)/882)
pval=2*(1-stats.norm.cdf(abs(zdata)))
zdata,pval


# In[ ]:


#using the inbuilt function
proportions_ztest(150,882,237/1470)


# ##Two sample z test:

# In[ ]:


#comparing the attrition proportions of males and females
p1=150/882
p2=87/588
pp=237/1470
p1,p2,pp


# In[ ]:


zdata=(p1-p2)/(((pp*(1-pp))*((1/882)+(1/588)))**0.5)
zdata


# In[ ]:


pval=2*(1-stats.norm.cdf(abs(zdata)))
pval


# In[ ]:


#using inbuilt function
counts=np.array([150,87])
obs=np.array([882,588])
zdata,pvalue=proportions_ztest(counts,obs)
zdata,pvalue


# In[ ]:


df=pd.read_csv('../input/learn-ml-datasets/Migraine.csv',index_col=0)


# In[ ]:


df.head()


# In[ ]:


g1=df.age[df.headache=='yes']
g2=df.age[df.headache=='no']
df.boxplot(column='age',by='headache')


# In[ ]:


ct=pd.crosstab(df.headache,df.Gender)
ct


# In[ ]:


def twosampztest(x1,x2,n1,n2):
  p1=x1/n1
  p2=x2/n2
  pp=(x1+x2)/(n1+n2)
  zdata=(p1-p2)/np.sqrt(pp*(1-pp)*((1/n1)+(1/n2)))
  pval=2*(1-stats.norm.cdf(abs(zdata)))
  return zdata,pval


# In[ ]:


twosampztest(2279,387,2279+1266,387+220)


# In[ ]:


#using inbuilt function
counts=np.array([2279,387])
obs=np.array([2279+1266,387+220])
zdata,pvalue=proportions_ztest(counts,obs)
zdata,pvalue


# pval>0.05 implies proportions are same 

# ## Chi-square Test (proportion test for more than 2 samples)

# In[ ]:


from scipy.stats import chisquare, chi2_contingency


# In[ ]:


ct=pd.crosstab(df.hatype, df.Gender)
ct


# In[ ]:


chi2_contingency(ct)


# pvalue<0.05 so we can say gender does influence the type of headache

# In[ ]:


obs=np.array([410,340,250,95,85,70])


# In[ ]:


df=pd.DataFrame(obs.reshape(2,3),columns=['healthy','mild','severe'],index=['sample1','sample2'])
df


# In[ ]:


exp=[]
for i in df.index:
  for j in df.columns:
    exp.append((df.loc[i].sum())*(df.loc[:,j].sum())/1250)
exp=np.array(exp)
exp


# In[ ]:


chi_data=(((obs-exp)**2)/exp).sum()
chi_data


# In[ ]:


chi_value, pval, nrows, ec=chi2_contingency(df)
print('chi_value:',chi_value)
print('p-value:',pval)
print('degrees of freedom:',nrows)
print('expected counts array: \n',ec)


# pvalue>0.05 implies proportions of all the groups are statistically equal

# In[ ]:


n=500
observed_values=np.array([190,185,90,35])
expected_values=np.array([n*0.3,n*0.45,n*0.2,n*0.05])
chisquare(observed_values,expected_values)


# pvalue<0.05 implies proportions are not same as that of the population(expected)

# # AnoVa (f test) One way

# In[ ]:


from scipy.stats import f_oneway


# In[ ]:


a=np.array([30,40,50,60])
b=np.array([25,30,50,55])
c=np.array([25,30,40,45])
a.var(),b.var(),c.var()


# In[ ]:


sstr=0
for i in [a,b,c]:
  sstr=sstr+(len(i)*((i.mean()-40)**2))
mstr=sstr/(3-1)
mstr


# In[ ]:


sse=0
for i in [a,b,c]:
  sse=sse+((len(i)-1)*i.var(ddof=1))
mse=sse/(12-3)
mse


# In[ ]:


fdata=mstr/mse
fdata


# In[ ]:


#using built-in function
fdata,pvalue=f_oneway(a,b,c)
fdata,pvalue


# In[ ]:


df=pd.read_csv('../input/learn-ml-datasets/fair_pay_data.csv')


# In[ ]:


df.head()


# In[ ]:


g1=df.salary[df.department=='Sales']
g2=df.salary[df.department=='Engineering']
g3=df.salary[df.department=='Finance']


# In[ ]:


f_oneway(g1,g2,g3)


# pvalue > 0.05 implies average salary is same accross all departments

# #Anova Two way

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[ ]:


a=np.array([30,40,50,60])
b=np.array([25,30,50,55])
c=np.array([25,30,40,45])
p=np.hstack([a,b,c])
df1=pd.DataFrame({'age':p})
df1['group']=pd.Series('a a a a b b b b c c c c'.split())
df1


# In[ ]:


d=np.array([43,45,45,47])
e=np.array([37,40,40,43])
f=np.array([34,35,35,36])
q=np.hstack([d,e,f])
df2=pd.DataFrame({'age':q})
df2['group']=pd.Series('d d d d e e e e f f f f'.split())
df2


# In[ ]:


mod=ols('age ~ group', data=df1).fit()
anv_table=sm.stats.anova_lm(mod, typ=1)
print(anv_table)


# In[ ]:


mod=ols('age ~ group', data=df2).fit()
anv_table=sm.stats.anova_lm(mod, typ=1)
print(anv_table)


# In[ ]:


sstr=0
for i in [a,b,c]:
  sstr=sstr+(len(i)*((i.mean()-40)**2))
mstr=sstr/(3-1)
sstr,mstr


# In[ ]:


sse=0
for i in [a,b,c]:
  sse=sse+((len(i)-1)*i.var(ddof=1))
mse=sse/(12-3)
sse,mse


# In[ ]:


fdata=mstr/mse
fdata


# In[ ]:


df=pd.read_csv('../input/learn-ml-datasets/fair_pay_data.csv')


# In[ ]:


df.head()


# In[ ]:


mod=ols('salary ~ new_hire + job_level + new_hire:job_level', data=df).fit()
anv_table=sm.stats.anova_lm(mod, typ=2)
print(anv_table)


# In[ ]:


ct=pd.crosstab(df.new_hire,df.job_level)
ct


# #OLS

# In[ ]:


df=pd.read_csv('../input/advertising-mul/advertising.csv')


# In[ ]:


df.head()


# In[ ]:


mod=ols('Sales ~ TV + Radio + TV:Radio', data=df).fit()
mod.params


# In[ ]:


mod.summary()


# In[ ]:


mod=ols('Sales ~ TV ', data=df).fit()
mod.params

