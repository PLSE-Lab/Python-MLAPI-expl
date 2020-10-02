#!/usr/bin/env python
# coding: utf-8

# ## Initial exploration

# Let's load some needed libraries and the dataset, and get a first glance and the contents. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


#read the file
df = pd.read_csv('../input/voice.csv',sep=',')

#explore content
df.describe()


# Here you can configure some settings we gonna play with afterwards

# In[ ]:


#------------- CONFIG -----------------
#wanna do all plotting stuff??
doGraphs=True

#Debug mode 
global debug
debug=False

#Reduce test mode  (only first 4 features are removed in the IPA)
reduced_scan=False
#------------- CONFIG -----------------


# Next, we can take a look at the different features, for both male and female. From here we can already have a feeling of the most powerful acoustic attributes to distinguish one from the other.

# In[ ]:


#---plot some variables
def draw_comparison(df, vars):
    cols=4
    rows=int(len(vars)/cols)

    f, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))

    for vv,axid in zip(vars,range(len(vars))):
        #plot variable for male
        sns.distplot(df[df['label']=='male'][vv], hist=False, color="g", kde_kws={"shade": True}, ax=axes[int(axid/cols),(axid%cols)], label='Male')
        #plot variable for female
        sns.distplot(df[df['label']=='female'][vv], hist=False, color="m", kde_kws={"shade": True}, ax=axes[int(axid/cols),(axid%cols)], label='Female')

        #legend
        ax = axes[int(axid/cols),(axid%cols)]
        ax.legend(loc='best')
        ax.set_xlabel(vv,fontsize=18)
        ax.tick_params(top='on',right='on',direction='in')
        
        
    
    plt.tight_layout()
    plt.show()
    #plt.savefig('distcomp.png')

if doGraphs:
    draw_comparison(df, vars = list(df)[:-1]) #['Q25','IQR','meanfun','meanfreq','maxfun','kurt'])


# It is evident that some attributes like `meanfun` and `IQR` will yield a good separation power, while others like `modindx` and `kurt` have little to offer to the classification task.

# ### Correlations and dimensionality
# 
# But now, how are they correlated? They all really help / add something new?

# In[ ]:


#heatmap version
def draw_corr_matrix(df):
    #compute correlation matrix
    corr = df.corr()

    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=False, fmt=".1f")
    plt.title('Acoustic properties correlations',fontweight='bold',fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    #plt.savefig('corr_matrix_hmap.png')
    
if doGraphs:
    draw_corr_matrix(df)


# We can observe that some variables are ~ fully correlated, so we can reduce the dimensionality of our problem and remove a few features from our dataset.
# 
# 
# Although is already clear from the correlation matrix, let's check some cases explicitly (extend at will!):

# In[ ]:


#correlation plots
def plot_corr(df,var1,var2):
    sns.jointplot(df[var1],df[var2], kind="hex", color="#4CB391")
    plt.tick_params(top='on',right='on',direction='in')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.show()
    #plt.savefig('corr_%s_%s.png' % (var1.replace(' ','_'),var2.replace(' ','_')))

if doGraphs:
    plot_corr(df,'dfrange','maxdom')
    plot_corr(df,'meanfreq','centroid')
    plot_corr(df,'meanfreq','median')
    plot_corr(df,'skew','kurt')

if debug:
    print(list(df))


# The four pairs of variables are strongly correlated, so we will remove one for each and reduce our exploratory space.

# In[ ]:


#remove redundant info here 
df.drop(['dfrange','kurt','median','centroid'], axis=1, inplace=True)

if debug: 
    print(list(df))


# ## Classification models
# 
# Let's get ready to train and test some algorithms for the classificationt task. We extract and scale the features and encode the target labels first.

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #ignore some DeprecationWarnings for now. Need to move to model_selection module.

from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale

#Features (scaled)
X=scale(df.iloc[:,:-1])

#Labels (encode to binary)
y=df.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder
bincoder = LabelEncoder()
y = bincoder.fit_transform(y)


# Then declare some models, in this case three SVC variants. We defined a generic function for model training-testing, using cross validation (5 K-fold). (the last argument will make sense in a moment...)

# In[ ]:


## Define models here
models = {'SVC_Linear' : SVC(kernel='linear'), 'SVC_RBF' : SVC(kernel='rbf'), 'SVC_poly' : SVC(kernel='poly')}


# Eval models for shrinking dataframes  .  (The excl_vars argument is a list of the features to be ignored in the dataset)
def eval_models(models,df,y,excl_vars=[]):
    global debug
    
    #new features excluding some     
    X = df.drop(excl_vars, axis=1).iloc[:,:-1]

    mscores=[]
    for ml in models:
        #K-fold train-test
        scores = cross_val_score(models[ml], X, y, cv=5)
        if debug:
            print('The K-fold %s model <score> is %.4f   :  ' % (ml, scores.mean()), scores)
        mscores.append(scores.mean())

    return mscores
 


# Then we evaluate the models and booked the scores for later use.

# In[ ]:


##Create dictionary for scores 
score_dict = {}

## EVAL model (on full features space)
score_dict['IP0'] = eval_models(models, df,y,[])


# In[ ]:


for nmod in range(len(models)):
    print('%-10s : %.2f' %  (list(models)[nmod], score_dict['IP0'][nmod]))


# 
# ## Increasingly Pessimistic Analysis (IPA)
# 
# ----------

# As noted already, the mean frequency seems to be the most powerful attribute to distinguish between male and female voices. A sensible selection of this variable alone can reach almost perfect score. So how do we make things a bit more interesting?
# 
# Well, imagine for a moment we don't have access to this acoustic attribute, i.e. some people might not be sensitive to the actual pitch, something like daltonism for visual recognition. So let's repeat the classification exercise removing that feature from our database. 

# In[ ]:


## EVAL regression model (on reduced features space)
score_dict['IP1'] = eval_models(models,df,y,['meanfun'])

for nmod in range(len(models)):
    print('%-10s : %.2f' %  (list(models)[nmod], score_dict['IP1'][nmod]))


# Although the performance has expectedly degraded a bit, the model still resolves reasonably well as there are other variables similarly powerful. 
# 
# Let's then continue the game a bit further, in what I called here a Increasingly Pessimistic Analysis (IPA). (Could have been Gradual Truncation Analysis for game lovers, but I like beer :P).
# 
# The idea is removing the most powerful variables one at a time, to force the model to more challenging scenarios. 
# So the sequence go:
# 
#  1. IP1 remove the top variable from the features space (i.e. what we just did above)
#  2. IP2 remove the top two variables from the features space
#  3. ...
# 
# ...
# 
# You get the idea. How far can we go without losing too much predicting power? 
# 
# First we need to identify the discriminating power or importance ranking for each individual variable:
#  

# In[ ]:


##Get ranking list 

# Make ranking of individual discrimination power (by Recursive Feature Elimination, using the SVCLinear estimator here)
from sklearn.feature_selection import RFE
rfe = RFE(estimator=models['SVC_Linear'], n_features_to_select=1, step=1)
rfe.fit(X,y)
ranking = rfe.ranking_

#print ranking
ranked_vars = [a for (b,a) in sorted(zip(ranking,list(df)))]
print('Ranked Features')
print('-'*40)
for pos,var in enumerate(ranked_vars):
    print('%-12i%s' % (pos,var))
print('-'*40)


# Then we train and test the model for each sub-dataset, as defined above. 
# 
# *Note*: It might take a few minutes to run the full scan over all the features list. You can enable `reduced_scan` in the Config block at the beginning, or modify the `maxScan` variable by hand here below.

# In[ ]:


### train and test the model on the shrinking dataset 
#maxScan = 6
maxScan = len(ranked_vars)  #release the kraken!

if reduced_scan: #as configured at the top
    maxScan=4
for att in range(2,maxScan):
    score_dict['IP%s'%str(att)] = eval_models(models,df,y,ranked_vars[:att])

if debug:  #print scores for each IPA-step
    print(score_dict)


# Shall we put it all together?

# In[ ]:


### make plot of score (mean of K-fold validation) vs IPX (X=0,1...N-1) . And we can add X=N=random guessing! We better do slighly better than that in all cases :)
def plot_scores(sdict, models):
    colors=['r','b','m','g']
    xp = range(maxScan)
    for mid,ml in enumerate(models.keys()):
        yp=[]
        for sc in range(maxScan):
            yp.append(sdict['IP%d'%sc][mid])
        plt.plot(xp,yp,colors[mid],label=ml)
    plt.xlabel('IP step',fontsize=14)
    plt.ylabel('score',fontsize=14)
    plt.tick_params(top='on',right='on',direction='in')
    plt.legend(loc='upper right')
    plt.show()
    #plt.savefig('IPA_scores.png')


plot_scores(score_dict, models)


# The performance of the three models follows roughly the loss of information at each step, until we are left with only the last feature (maxfun). As seen in the plot above the distributions for male and female are very similar for this variable, so we get close to a poor random estimator. 

# 
# 
# Some things in the pipeline:
# 
# * I may want to do the normal RFE (with x-val) analysis 
# * Try with other models 
# * Compare ranking to ExtraTreesClassifier()
# * ...
# 
# **Comments are welcome!  :)**

# In[ ]:




