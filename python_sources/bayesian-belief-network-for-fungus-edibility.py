#!/usr/bin/env python
# coding: utf-8

# # Is it a mushroom or is it a toadstool?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy
import pandas
import collections
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# To classify fungi as edible (mushrooms) or poisonous (toadstools), I use a Bayesian Belief Network with one hidden variable, on which each of the observables is dependent. This is inferred by clustering the samples as follows.
# A cluster is characterised by a probability distribution P(Xi|C) for each variable. For each observation, the average of P(Xi|Cj) is calculated for each existing cluster Cj, and the most similar cluster is found. If the probability of the observation belonging to the most similar cluster is greater than 0.5, the observation is added to that cluster. If not, a new cluster is created, starting with that cluster.

# In[ ]:


class FungusClassifier(object):
    """Infers a hidden variable and uses Bayesian classification to predict whether a fungus is 
    edible or poisonous"""
    def __init__(self,filename):
        data=pandas.read_csv(filename,index_col=False)
        clusters=[]
        for (i,row) in data.iterrows():
            best=-1
            sim=0.5
            for (j,cluster) in enumerate(clusters):
                x=sum(cluster[key][value]/sum(cluster[key].values())
                      for (key,value) in row.iteritems())/(data.shape[1])
                if x>sim:
                    best=j
                    sim=x
            if best==-1:
                clusters.append(collections.defaultdict(lambda: collections.defaultdict(float)))
                print(i+1,'rows analysed',len(clusters),'clusters found')
            for (key,value) in row.iteritems():
                clusters[best][key][value]+=1.0
        index=[]
        for column in data.columns:
            index.extend([(column,value) for value in data[column].unique()])
        self.probabilities=pandas.DataFrame({(key,value):[cluster[key][value]+1.0 for cluster in clusters]
                                            for (key,value) in index}).T
        self.prior=self.probabilities.sum(axis=0)
        self.prior/=self.prior.sum()
        self.edibility_prior=self.probabilities.loc['class'].sum(axis=1)
        self.edibility_prior/=self.edibility_prior.sum()
        def normalize(group):
            return group.div(group.sum(axis=0),axis='columns')
        self.probabilities=self.probabilities.groupby(axis=0,level=0).apply(normalize)
        
    def __call__(self,**kwargs):
        "Estimates the probability that a fungus is edible given the features in kwargs"
        category=self.prior.copy()
        for (key,value) in kwargs.items():
            category*=self.probabilities.loc[(key,value)]
            category/=category.sum()
        result=self.edibility_prior*((self.probabilities.loc['class']*category).sum(axis=1))
        return result/result.sum()
    
    def test(self,filename):
        """Produces KDE plots of the estimated probability"""
        data=pandas.read_csv(filename,index_col=False)
        observables=[column for column in data.columns if column!='class']
        results=pandas.DataFrame([self(**row) for (i,row) in data[observables].iterrows()])
        results.loc[:,'class']=data['class']
        return results


# This creates 11 clusters

# In[ ]:


BBN=FungusClassifier('../input/mushrooms.csv')


# Here is the prior probability of a fungus being edible.

# In[ ]:


BBN.edibility_prior.plot.bar()


# Here is the prior probability of each cluster.

# In[ ]:


BBN.prior.plot.bar()


# Here is the probability of edibility given each cluster. Most clusters discriminate very strongly between mushrooms and toadstools.

# In[ ]:


BBN.probabilities.loc['class'].T.plot.bar()


# Here is a KDE plot of the posterior probability of edibility over the entire sample.

# In[ ]:


result=BBN.test('../input/mushrooms.csv')
result['e'].plot.kde()


# All edible fungi are classified as edible with a high degree of confidence.

# In[ ]:


result[result['class']=='e']['e'].plot.kde()


# Poisonous fungi are mostly classified a poisonous with a high degree of confidence. There are however a few that are misclassified (these probably belong to cluster 0).

# In[ ]:


result[result['class']=='p']['e'].plot.kde()


# Fungi identified as edible with >50% confidence are 90% likely to be correctly classified.

# In[ ]:


result[result['e']>0.5]['class'].value_counts(normalize=True).plot.bar()


# With a confidence threshold of 90%, correct classification is close to 99%

# In[ ]:


result[result['e']>0.9]['class'].value_counts(normalize=True).plot.bar()


# This is a pretty good classifier, but given the safety-critical nature of the problem, I'd like to do better.
