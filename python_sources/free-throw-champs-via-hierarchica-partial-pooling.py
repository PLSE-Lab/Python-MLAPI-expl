#!/usr/bin/env python
# coding: utf-8

# # The best free throw shooters in the NBA via Hierarchica Partial Pooling
# 
# Who's the best free throw shooter in this dataset? We're going to answer this simple question using an analaysis technique called "Partial Pooling". We're trying to come up with the probability $p$ that each player will make his next shot, based on the data we already have. You can think of this as an inherent skill for each player- the closer to 1 the value of $p$ is, the better the player is at free throws.
# 
# One way of answering this question is to assume that every basketball player has exatly the same chance of scoring his next free throw- i.e. each have the same inherent skill. In this case, we'd work out this skill value by taking the total number of shots made divided by the number of shots in total.
# 

# In[ ]:


#Code here to take the average
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns #pretty plotting

df = pd.read_csv('../input/free_throws.csv')
global_average=float(df['shot_made'].sum())/df['shot_made'].count() #Could also say df['shot_made'].mean() here


print("Average for all players: p={:.3f}".format(global_average))


# What we're really doing here (perhaps without realising it) is assuming that this data is a series of [Bernoulli trials](https://en.wikipedia.org/wiki/Bernoulli_trial), and taken together they will follow a [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution). In a binomial distribution, the expected number of successes (free throws) is given by $np$, where $n$ is the number of attempts and $p$ is the probablity of success (what we want to measure). Our data give us the number of trails and the number of sucesses, so by doing the average above (succeses/attempts) we get the probability of sucess.
# 
# This is known as "Complete Pooling"- we're lumping all players together as one to get a single number for the whole population. This is obviously not a good thing to do in this case- we know that some players are better than others!

# At the totally opposite end of the spectrum, we can assume that each player has a skill level _completely unrelated_ to any other player. In other words, they all follow completely separate binomial distributions. In this case, we can come up with a skill level for each player by taking the total number of shots made _by that player_ and dividing it by the total number he takes. This gives a plot like this:
#  

# In[ ]:


#Plot the stats for each player, as well as this average

player_grouping=df.groupby(['player'])

shots_made=player_grouping['shot_made'].sum().values.astype(float)
attempts=player_grouping['shot_made'].count().values.astype(float)

names=np.array([name for name, _ in player_grouping['player']])

#Plot it all
with plt.style.context(('seaborn')):
    fig, ax=plt.subplots(figsize=(10, 10))
    scatter=ax.scatter(attempts, shots_made/attempts, c=np.log10(attempts), cmap='plasma')
    ax.axhline(global_average, c='k', linewidth=2.0, linestyle='dashed', label='Global Average')
    cb=fig.colorbar(scatter, ax=ax)
    cb.set_label(r'$\log_{10}$ Attempts', fontsize=20)
    
    ax.legend(fontsize=20, loc='lower right')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xscale('log')
    ax.set_xlabel('Attempts', fontsize=20)
    ax.set_ylabel('Conversion Rate', fontsize=20)
    ax.set_title('NBA Free Throws', fontsize=30)


# This is known as a "No Pooling" model, because we treat each player completely independently. It's better than the complete pooling case, but it still runs into some issues. When looking at a dataset like this, it's inevitable that not all players take the same number of free throws. So how can we compare a player who's taken 1,000 with someone who's only taken 50? Is a player who makes 45/50 free throws better than one who makes 18/20? Or 900/1000? 
# 
# Also, we get strange values for some players- there are a few who have made 10/10 shots, so do we asisgn them a probability of making their next shot of 1? Likewise, a couple have missed 4 out of 4. Are they no hopers, destined never to make a free throw in their whole career? We could arbitrarily exclude players with less than, say, 100 shots, but we don't want to throw away data if we don't have to. 
# 
# Finally, we know that it's not correct to assume that the skill levels of the players are _completely_ independent. They're still taking the same shots with (mostly) similar techniques, and all NBA players can't have got to where they are without a pretty high baseline of free throw shooting skilll.
# 
# 

# ## Partial Pooling
# 
# This is where "Partial Pooling" comes in, with the best of both approach. We assume that each player has his own skill value $p_{i}$, but that each of these $p_{i}$ values are drawn from one global probability distribution common to all players. Now, a truly great free thrower's impressive record will shine through and shift his measured skill level higher, whilst we still get reasonable answers for those players who haven't taken many free throws yet- they'll be biased towards the global average until they've taken enough shots to shift them to their rightful place. Another way of saying this is that we're going to set an informative "prior" on the skills of individual players, based on the data from all the other players put together.  
# 
# So what do we choose as this prior probability distribution? Let the data decide! We'll get this distribution straight from the data too, which is why this is known as a "hierarchical" method. Just to round off the upsides of this approach, we'll also get _error bars_ and a full posterior for our estimates with this technique, unlike the simple point-estimates we worked out before.
# 
# 

# Just to recap what we're after: we want to estimate the skill level, $p_i$, for each player in the dataset. We'll also get (for free) the global skill distribution of all the players in the dataset (our prior), from which these $p_i$ values are drawn. We know that the repeated bernoulli trials of free throws follow a binomial distribution, and we know that each player's $p_i$ must be between 0 and 1. We'll approximate the global skill level distribution as a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution). This makes sense for a couple of reasons. Firstly, a beta distribution is only defined on the interval [0, 1], which is reasonable for probabilities. Furthermore, it's the "conjugate prior probability distribution" for a binomial distribution, which essentially means that the _output_ ("posterior") probability distribution for each player's $p_i$ is also guaranteed to be a beta distribution, which makes sense intuitively and guarantees that we'll have $0<p_i<1$.

# We'll have to have some "hyperparameters" to parameterise our global skill distribution, which we'll call $\phi$ and $\kappa$. $\phi$ parameterises where the peak of our global skill distribution is. We'd expect this to be pretty close to the global average conversion rate we calculated above (0.757), because this is where most of the data seem to lie. $\kappa$ descibes the width of our global skill distribution. We can see that we have players with many free throw attemps make anywhere between 0.9 and 0.4 of their free throws, so we'd assume that we should get a width of the distribution encompassing that range.

# I'm going to use a really great python module called [pymc3](http://docs.pymc.io/index.html) to do this analysis. We'll set up all these parameters and then sample from the "posterior" probability for each player to get our results. 

# In[ ]:


import pymc3 as pm, theano.tensor as tt

#Number of different players we have
N_players=len(attempts)

with pm.Model() as model:
    
    #Hyper parameters on the global skill probability distibution
    phi = pm.Uniform('phi', lower=0.0, upper=1.0)

    kappa_log = pm.Exponential('kappa_log', lam=1.5)
    kappa = pm.Deterministic('kappa', tt.exp(kappa_log))
    
    #Here are the individual beta distributions for each player. We start them off at their values based on
    #the values of phi and kappa, but these will be updated as they see the data.    
    #We can either describe a beta distribution by two variables alpha and beta, or by its mean and standard deviation.
    #Here we have alpha=phi*kappa, and beta=kappa*(1-phi). This implies (via some maths) that phi is the mean of the 
    #beta distribution and kappa is related to the variance. 
    #See http://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html for more infomation!
    individual_player_skills =pm.Beta('individual_player_skills', alpha=phi*kappa, beta=(1.0-phi)*kappa, shape=N_players)
    
    #Our binomial likelihood function
    likelihood = pm.Binomial('likelihood', n=attempts, p=individual_player_skills, observed=shots_made)

    trace = pm.sample(2000, tune=1000, chains=2)

_=pm.traceplot(trace, varnames=['phi', 'kappa'])


# Okay, having done all of the sampling, let's plot the results. We'll make the same plot as before, and see where the peak of the global skill distribution lies with respect to the global average we fo.

# In[ ]:


#get the skill level assigned to each player
skill_levels=np.mean(trace['individual_player_skills'], axis=0)

#Get a KDE of the global skill and kappa traces- we'll use these in a bit
from scipy import stats
xs_kappa=np.linspace(15.0, 30.0, 1000)
kde_kappa=stats.gaussian_kde(trace['kappa'])
kde_vals_kappa=kde_kappa(xs_kappa)

xs_skill=np.linspace(0.70, 0.74, 1000)
kde_skill=stats.gaussian_kde(trace['phi'])
kde_vals_skill=kde_skill(xs_skill)


#Plot it all
with plt.style.context(('seaborn')):
    fig, ax=plt.subplots(figsize=(10, 10))
    scatter=ax.scatter(attempts, skill_levels, c=np.log10(attempts), cmap='plasma')
    ax.axhline(global_average, c='k', linewidth=2.0, linestyle='dashed', label='Global Average')
    cb=fig.colorbar(scatter, ax=ax)
    cb.set_label(r'$\log_{10}$ Attempts', fontsize=20)
    
    #Details
    ax.legend(fontsize=20, loc='lower left')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xscale('log')
    ax.set_xlabel('Attempts', fontsize=20)
    ax.set_ylabel('Conversion Rate', fontsize=20)
    ax.set_title('NBA Free Throws', fontsize=30)
    ax.set_xlim(0.9, 10**4)
    ax.set_ylim(0.0, 1.0)
    
    # Shade the peak of the Beta distribution (plus uncertainty)
    x=np.linspace(0.1, 2*10**4)
    for i in range(0, 100,  1):
        inds=np.where(kde_vals_kappa>np.percentile(kde_vals_kappa, i))[0]    
        plt.fill_between(x, xs_skill[inds[0]], xs_skill[inds[-1]], alpha=0.8/100.0, facecolor='k')


# So this looks okay! The peak of the prior distribution (and its uncertainty) as shown by the shaded region is also very close to the global average we found before. 
# We can also see that for the players with lots of free throw attempts, our results really haven't changed too much. There's enough data for them that their binomial distributions are well known already, and our analysis using the global skill distribution as a prior doesn't have much of an effect. 
# 
# The biggest change is seen for those players who haven't taken many shots. They've been pulled much closer to the peak of the skill distribution prior, which makes sense- we'd expect them to perform as well as the average NBA player, until we have some data to show otherwise. For example, the player with only one free throw attempt at the very left of the graph is placed pretty much slap bang on the peak of the prior distribution. 
# 
# 
# So let's plot that underlying distribution of player skills as well:

# In[ ]:


#Plot of the underlying distribution of player skills

#These are the most probable parameters- the peak of the histograms for each parameter
gskill_map=xs_skill[np.argmax(kde_vals_skill)]
kapp_map=xs_kappa[np.argmax(kde_vals_kappa)]
beta_map=stats.beta(a=gskill_map*kapp_map, b=(1.0-gskill_map)*kapp_map)
#Plot them
with plt.style.context(('seaborn')):
    x=np.linspace(0.0, 1.0, 1000)
    fog, ax=plt.subplots(figsize=(10, 10))
    ax.plot(x, beta_map.pdf(x), c='r', zorder=10, label='Probability distribution\nof player skill')

    #Draw random samples from our chain to get an idea of the uncertainity
    randoms=np.random.randint(0, len(trace['phi']), size=1000)
    for gskill, kapp in zip(trace['phi'][randoms], trace['kappa'][randoms]):

        beta=stats.beta(a=gskill*kapp, b=(1.0-gskill)*kapp)
        ax.plot(x, beta.pdf(x), c='k', alpha=0.1)

    #Add a histogram of the original shots data
    hist=ax.hist(shots_made/attempts.astype(float), 50, normed=True, facecolor='b', alpha=0.8, label='Original data: \nshots/attempts')  

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel(r'$p_{i}$', fontsize=20)
    ax.legend(loc='upper left', fontsize=20)


# So the underlying distribution of player skills is peaked at around $p=0.75$, with a bit of a tail to lower values, which makes intuitive sense. Note that this red line _not_ a fit to the blue histogram. We've taken the observed data points and inferred the underlying probability distribution from which they're drawn, taking into account the fact that each player hasn't taken the same number of shots.

# ## Individual player results
# 
# So who is actually the best free thrower in the data? Here are the skill levels for the top and bottom 10 players, as well as the uncertainties in our estimations.

# In[ ]:


#The indices which would sort the array of player skills
top_inds=np.argsort(skill_levels)[-10:]
bottom_inds=np.argsort(skill_levels)[:10]
inds=np.concatenate((bottom_inds, top_inds))

#Traces and names
t=trace['individual_player_skills'][:, inds]
n=names[inds]
n=np.insert(n, 10, [''])

#Make a violin plot
with plt.style.context(('seaborn')):
    fig, ax=plt.subplots(figsize=(10, 10))
    parts=ax.violinplot(t, positions=np.delete(np.arange(21), 10), vert=False, showextrema=True, showmedians=True)
    ax.axhline(10.0, linestyle='dashed', linewidth=2.0, c='k')
    #Details
    ax.set_yticks(np.arange(0, 21))
    ax.set_xticks(np.arange(1, 11)/10.)
    ax.set_yticklabels(list(n), fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(r'p_i')
    #Colour the violins by log10(attempts)
    cm=plt.get_cmap('plasma')
    for pc, ind in zip(parts['bodies'], inds):
        c=cm(np.log10(attempts[ind])/np.max(np.log10(attempts)))
        pc.set_facecolor(c)
        pc.set_edgecolor('k')
        pc.set_alpha(1)
    #Reuse the same colorbar from before
    cb=fig.colorbar(scatter, ax=ax)
    cb.set_label(r'$\log_{10}$ Attempts', fontsize=20)
    


# It's most likely that Steve Nash wins this contest- he has a mean skill level of >0.91. However, everything is so close together at the top that it's really a toss up between any of these guys. For example, the probability that Brian Roberts has a skill level above this pretty high, since his $1-\sigma$ confidence region extends to above 0.91 too. We'd need all of these players to take an infinite number of free throws fo these confidence regions to shrink down exctly on their true values. I guess that this debate will rage on!
# 
# What we can say with some confidence, however, is that all of these top 10 guys are better at free throws than the guys down the bottom! But perhaps we didn't need such a complicated analysis to come to that conclusion... 

# 

# 

# In[ ]:




