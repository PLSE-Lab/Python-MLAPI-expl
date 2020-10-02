#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The design of competitions with  public training and test data-sets requires detailed planning. Key issues that are not trivial to consider include the choice of cost function to measure performance, the size the training/test sets, and the assessment of the risk of the tournament results being either too variable or over-fitted. Using a Bayesian approach, we address these issues from a more quantitative perspective. The proposed method relies on only three key statistics: number of test records used, number of entries submitted, and the top score. A probability distribution of the maximum achievable score is then calculate. A metric for estimating future improvement over the top score is also derived; which can be used to assess significant contributions or risk of published results being tarnished due to public exposure of the test records and scores. We present our analysis  on several [PhysioNet][1] competitions, the [MIT-BIH][2] ECG beat detection task, and the  [Kaggle seizure prediction][3] competition.  The Bayesian estimator presented here can be a useful for:
# 
# 1. Designing public challenges and competitions. Using preliminary entry scores, (collected during an initial phase of the competition), the organizers can better ***estimate the required test set size or cost function*** for the competition in order to obtain reasonably small 95% confidence intervals around the final top score.
# 
# 2. External parties wishing to ***access over-fitting risk on the claimed performance of an algorithm on a public test set***, can use the 95% confidence interval from that data-set in order to gauge if the algorithm has over-trained on the public test records ( or the public test-set has be been overlay exposed, such as by using features from previous algorithms that reported high scores on the test sets). 
# 
# 3. Students and  researchers, can use the estimated chance of improvement on a challenge in order to properly ***allocate resources to  underperforming competitions***,  so that their efforts are more likely to have a meaningful impact on the field.
# 
#   [1]: https://www.physionet.org/challenge/
#   [2]: https://physionet.org/physiobank/database/mitdb/
#   [3]: http://blog.kaggle.com/2016/10/14/getting-started-in-the-seizure-prediction-competition-impact-history-useful-resources/

# # Estimating maximum achievable performance
# 
# The maximum achievable score in a competition, \( \theta \), is estimated via a Bayesian approach. It is assumed that the maximum score, \( \theta \), $\theta$, is uniformly distributed between 0 and 1. A competitor's score can then be modeled as a random variable, \( X \), with distribution
# 
# $$
# X \sim U[0,\theta]\,. \qquad (1)  
# $$
# 
# The maximum likelihood function for a set of \( N\)  independent samples, \( \bar{x}\), from  \( X\) is then given by ([\[Scharf1991][1],[DudaHart2012\]][2]):
# 
# $$
# l(\theta,\bar{x}) = \frac{1}{\theta^N}I_{[max(\bar{x}),1)} \qquad (2) 
# $$
# 
# where :
# 
# $$
#     I_{[max(\bar{x}),1)}= 
# \begin{cases}
#     1,& \text{if } max(\bar{x}) \leq x\leq 1\\
#    0,         & \text{otherwise}
# \end{cases}
# $$
# 
# 
# Estimation of the number of independent samples $\hat{N}$ is a non-trivial task.  For Challenges that log the errors in each record and for each submission, a more accurate estimate of \(\hat{N}\) may be achieved by looking at pair-wise independence of record errors across submissions, eliminating submissions whose distribution of errors across records are dependent with others via  a process similar to [Gram-Schmidt orthogonalization][3]. The log of errors across records would also allow challenge designers to measure which records are highly correlated and select a test set with less correlation in relation to the task at hand. Unfortunately this approach is not possible, or feasible, for several competitions. A more conservative estimate can be used by setting \(\hat{N}\) to the minimum between number of test records used and number of entries submitted.  This choice  of \(\hat{N}\) results on a confidence interval that will not decrease when number of entries exceed the test set size, yielding an estimator that is not [consistent][4], but which reflects our limited information from the test size. This second estimate for \(\hat{N}\) can be used on open test data-sets where the number of entries is unknown, such as the [MIT-BIH data-set][5]. Only this second estimate of \(\hat{N}\) is used on our analysis. 
# 
# 
#   [1]: http://tocs.ulb.tu-darmstadt.de/21055823.pdf
#   [2]: http://tocs.ulb.tu-darmstadt.de/21055823.pdf
#   [3]: https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
#   [4]: https://en.wikipedia.org/wiki/Consistent_estimator
#   [5]: https://physionet.org/physiobank/database/mitdb)

# In[ ]:


import math
import numpy as np
from numpy import arange, reciprocal, Inf,log, divide, loadtxt, inf, zeros,array,    NaN, exp
from scipy.stats import entropy
from matplotlib.pyplot import plot, show,  figure, legend, ylabel, xlabel,     fill_between, bar, scatter, ylim, text,title
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from numpy import Inf

def estimateTheta(N,best):
    "Use Baye's rule to estimate proportion of success from measurement"
    M=100 #Number of points to estimate in the likelihood function
    theta=arange(best,1,(1-best)/float(M))
    pdf=reciprocal(theta)**N
    pdf=pdf/sum(pdf)
    return theta,pdf
    


# ## Estimating score likelihoods on PhysioNet competitions

# In[ ]:


# Analysis of performance on PhysioNet Datasets 
#M=number of test records, N= number of team submissions
chal=[{"best":0.83,"M":15, "N":Inf,'title':'2010 Filling The Gap'}, 
      {"best":0.905,"M":2, "N":Inf,'title':'2004 ST Changes'},
      {"best":0.5353,"M":4000, "N":21,'title':'2012 Mortality Prediction'},
      {"best":0.8426,"M":500, "N":41,'title':'2015 False Alarm'},
      {"best":0.932,"M":500, "N":49,'title':'2011 ECG Quality'},
      {"best":0.9991,"M":22, "N":Inf,'title':'0 MIT-BIH Detection'}
      ]
#Estimate likelihoods using the method previously described
figure(figsize = (12,7))
for x in chal:
    n=min(x['N'],x['M'])
    theta,pdf=estimateTheta(n,x['best'])
    plot(theta,pdf,label=x['title'])
legend() 
xlabel('Top Score')
ylabel('Likelihood')
title('Estimated Top Score Likelihood')


# # Calculating chances of significant improvement 
# 
# From Equation ((2))  the maximum likelihood estimate for the top achievable score is, $max(\bar{x})$,     the current best score.  It will be interesting and useful to relate this maximum value with a measure of dispersion of the likelihood function. For instance, although the known top scores on the 2010 and 2015 PhysioNet challenges from the figure above are close to 0.85, intuitively we expect the 2010 challenge to be easier to obtain further significant improvement due to the long tail nature of it's likelihood function.  Some useful statistics that we can use to relate the top score on a competition to the concentration of its likelihood function include:  variance/standard deviation, entropy, range, and distance from the mean. Note that the use of the distance of our maximum from the mean as a measure of dispersion is valid because of the unimodal monotic likelihood in our case.  We chose to use the distance between the maximum and estimated mean of the likelihood for this analysis, as its easier to interpret in terms of percent top scores.  Thus, the distance between the current best score and the mean of Equation ((2)), \\(\\mu\\),  is used as our estimate of percent of improvement on a competition, $\phi$ , given by 
# 
# $$
#   \phi = \mu - max(\bar{x}). \qquad (4)
# $$

# In[ ]:


def estimatePhi(best,N):
        theta,pdf=estimateTheta(n,best)
        mu = sum(pdf*theta) #Estimate of the mean of the likelihood
        return mu-best


# ## Calculating chances of significant improvement on PhysioNet competitions

# In[ ]:


chal=[{"best":0.79,"M":Inf, "N":10,'title':'2001 Afib Prediction'},
          {"best":0.82,"M":Inf, "N":10,'title':'2002 RR Modeling'},
          {"best":0.83,"M":Inf, "N":15,'title':'2010 Filling The Gap'},
          {"best":0.905,"M":Inf, "N":2,'title':'2004 ST Changes'},
          {"best":0.9262,"M":35,"N":13,'title':'2000 Sleep Apnea'},
          {"best":0.92,"M":Inf, "N":21,'title':'2008 T Alternans'}, 
          {"best":0.93,"M":Inf, "N":19,'title':'2009 Hypo Prediction'}, 
          {"best":0.5353,"M":4000, "N":20+1,'title':'2012 Mortality Prediction'}, 
          {"best":1-1/float(log(16.34)),"M":Inf, "N":28,'title':'2006 QT Interval'},
          {"best":0.8426,"M":500, "N":38+3,'title':'2015 False Alarm'}, 
          {"best":0.932,"M":500, "N":49,'title':'2011 ECG Quality'}, 
          {"best":0.879,"M":300, "N":60+1,'title':'2014 Beat Detection'}, 
          {"best":0.97,"M":Inf, "N":20,'title':'2004 Afib Termination'},
          {"best":0.9991,"M":Inf, "N":22,'title':'0 MIT-BIH Beat Detection'}
          ]
N=[]
max_hat=[]
for x in chal:
    n=min(x['N'],x['M'])
    max_hat.append(estimatePhi(x['best'],n)) #Estimate Chance of improvement
    N.append(n)

figure(figsize = (12,7))
scatter(N,max_hat)
ylim((0,0.1))
ylabel("Chance of Improvement ( Phi )")
xlabel("Estimated N from number of submissions or test records")
LABELS=[]
for ind,x in enumerate(chal):
    LABELS.append(x['title'])
    text(N[ind]+1,max_hat[ind],' '.join([x for x in chal[ind]['title'].split(' ')[1:]]))

title('Estimated Chance of Improvement vs N')
show() 


# It is interesting to note that the top score by itself is not a good indicator of an algorithm beating the current top score. For instance, the 2012 Mortality Prediction challenge had a top score of 0.53, but even though this score is may seem very low, the estimated expected score for this challenge and data-set is also low, 0.56 ($\phi=0.02$). The low $\phi$ value on the 2012 Mortality Prediction challenge is due to the large number of submissions for that particular challenge (${\hat{N}$=200). On the other hand, it is obvious that challenges with very high top scores also have a very low $\phi$ because the estimated expected score is squeezed between the current top score and 1. The challenges most likely to see improvements are those with a low top score and a low ${\hat{N}$.

# # Running estimate of score limits on [Kaggle Seizure Prediction Challenge][1]
# 
# 
#   [1]: https://www.kaggle.com/c/melbourne-university-seizure-prediction/data

# In[ ]:


from numpy import loadtxt, cumsum

def percentile(theta, pdf):
    
    per5=NaN
    per95=NaN 
    bottom=0
    top=0
    for ind in range(len(theta)):
        bottom+= pdf[ind] 
        if(bottom>0.05):
            per5 = theta[ind]
            break
    for ind in range(len(theta)):
        top+= pdf[-(ind+1)]
        if(top>0.05):
            per95 = theta[-(ind+1)]
            break
    return per5, per95

#Kaggle data for seizure prediction ( cummulative number of entries, top score)
data=[(0,0.5),
      (2,0.513430270985),
      (13,0.534266358229),
      (15,0.630684071381),
      (22,0.650499008592),
      (42,0.663759087905),
      (61,0.695571711831),
      (125,0.735292465301),
      (173,0.756934897555),
      (276,0.790631196299),
      (397,0.792514871117),
      (400,0.800789821547),
      (428,0.8399322538),
      (504,0.8399322538)]

N=[x[0] for x in data]
score=[x[1] for x in data]
figure(figsize = (12,7))
plot(N,score,'b-o',label='Current Top Score')
M=len(score)
best=zeros((M,1))
pred=zeros((M,1))
pred_range=zeros((M,2))

opt=0
for ind in range(M):
    opt=score[ind] if score[ind]>opt else opt
    best[ind]=opt
    n=min([7,ind])
    theta, pdf=estimateTheta(n,opt)
    mu = sum(pdf*theta)
    pred[ind]=mu
    per5, per95 = percentile(theta, pdf)
    pred_range[ind,:]=[per5,per95]

plot(N,pred,'r',label='Predicted Running Average Top Score')
fill_between(N, pred_range[:,0], pred_range[:,1], color='r', alpha=0.2,
             label='95% Confidence Interval')
legend(loc='lower right')
ylabel('Top Score')
xlabel('Cumulative Number of Entries')
title('Running Estimate of Top Score Limits for Seizure Prediction')
show()


# A running prediction, figure above,  of the best achievable score for the [Kaggle Seizure Prediction competition][1] was performed across the 85 days of the start of the competition and 504 entries. Notice that a larger number of test records would have allowed us a tighter confidence interval on the top achievable performance. Very early on the challenge, $\hat{N}$ becomes fixed by the number of test records available, as the number of entries quickly exceed the number of test records (ie, our estimator is a consistent estimator only up to $\hat{N}$=10). Beyond that, our 95% confidence interval is not expected to decrease efficiently.
# 
# 
#   [1]: https://www.kaggle.com/c/melbourne-university-seizure-prediction/data
