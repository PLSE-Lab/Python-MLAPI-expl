#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy
import numpy.linalg
import pandas
import scipy.cluster.hierarchy
import sklearn
import sklearn.cluster
import sklearn.linear_model
import re
import matplotlib.pyplot
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[ ]:


def ReduceDimensions(data,keep_variance=0.8,tolerance=1.0E-12,max_iter=1024):
    """Reduces dimensions of data"""
    residuals=data-data.mean(axis=1,keepdims=True)
    noise=(1.0-keep_variance)*residuals.var()
    convergence=1.0-tolerance
    result=None
    n=0
    while n<max_iter and residuals.var()>noise:
        component=residuals.var(axis=1,keepdims=True)
        if result is not None:
            component-=(result*(component.T.dot(result))).sum(axis=1,keepdims=True)
        component/=numpy.linalg.norm(component)
        corr=0.0
        n=0
        while n<max_iter and corr<convergence:
            projection=residuals.dot(component.T.dot(residuals).T)
            projection/=numpy.linalg.norm(projection)
            corr=projection.T.dot(component)
            component=projection
            n+=1
        if n<max_iter:
            if result is None:
                result=component
            else:
                result=numpy.hstack([result,component])
            residuals-=numpy.outer(component,component.T.dot(residuals))
    return result


# In[ ]:


split_name=re.compile('[.-]')
   
def rename_patients(name):
    components=split_name.split(name)
    return '-'.join(('TCGA',components[0],components[1]))
    
class BreastCancerAnalysis(object):
    """Analyses the breast cancer proteome date from Kaggle"""
    def __init__(self):
        """Loads the data tables"""
        self.protein_activity=pandas.read_csv("../input/77_cancer_proteomes_CPTAC_itraq.csv")
        self.protein_activity.set_index(['RefSeq_accession_number','gene_symbol','gene_name'],inplace=True)
        self.protein_activity.fillna(self.protein_activity.median(),inplace=True)
        self.protein_activity.rename(columns=rename_patients,inplace=True)
        self.clinical_data=pandas.read_csv("../input/clinical_data_breast_cancer.csv").set_index('Complete TCGA ID')
        self.principal_components=None
        self.protein_clusters=None
        self.patient_protein_activity=None
        self.clinical_prediction_models=None
        
    def fit_principal_components(self):
        """Reduces the dimensionality of the data"""
        self.principal_components=pandas.DataFrame(ReduceDimensions(self.protein_activity.values),
                                                   index=self.protein_activity.index)
        return self.principal_components
        
    def dendrogram(self):
        """Performs heirarchical clustering and plots a dendrogram to chose number
           of clusters to fit"""
        result=matplotlib.pyplot.gca()
        scipy.cluster.hierarchy.dendrogram(scipy.cluster.hierarchy.ward(self.principal_components),ax=result)
        return result
        
    def cluster_proteins(self,n=2):
        """Finds clusters of proteins whose activity is related"""
        clusters=sklearn.cluster.AgglomerativeClustering(n_clusters=n,
                                                         memory='/tmp')
        self.protein_clusters=pandas.DataFrame(clusters.fit_predict(self.principal_components),
                                               index=self.principal_components.index,
                                               columns=['cluster'])
        return self.protein_clusters
        
    
    def patient_cluster_activity(self):
        """Calculates the activity of each protein cluster for each patient"""
        mean_subtracted_protein_activity=self.protein_activity.sub(self.protein_activity.values.mean(axis=1),
                                                                   axis='index')
        patient_components=mean_subtracted_protein_activity.T.dot(self.principal_components)
        kernel=self.principal_components.merge(self.protein_clusters,
                                               left_index=True,
                                               right_index=True).groupby('cluster').mean().T
        self.patient_protein_activity=patient_components.dot(kernel)
        return self.patient_protein_activity

    def train_clinical_models(self):
        """Fits a model to predict each clinical feature from the protein activity"""
        self.clinical_prediction_models={'Gender':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'Age at Initial Pathologic Diagnosis':sklearn.linear_model.LassoLars(copy_X=True),
                                         'ER Status':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'PR Status':sklearn.linear_model.LogisticRegression(solver='lbfgs'),
                                         'HER2 Final Status':sklearn.linear_model.LogisticRegression(solver='lbfgs'),
                                         'Tumor':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'Node':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'Metastasis-Coded':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'AJCC Stage':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'Vital Status':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'OS Time':sklearn.linear_model.LassoLars(copy_X=True),
                                         'PAM50 mRNA':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'SigClust Unsupervised mRNA':sklearn.linear_model.LassoLars(copy_X=True),
                                         'miRNA Clusters':sklearn.linear_model.LassoLars(copy_X=True),
                                         'methylation Clusters':sklearn.linear_model.LassoLars(copy_X=True),
                                         'RPPA Clusters':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'Integrated Clusters (with PAM50)':sklearn.linear_model.LassoLars(copy_X=True),
                                         'Integrated Clusters (no exp)':sklearn.linear_model.LassoLars(copy_X=True),
                                         'Integrated Clusters (unsup exp)':sklearn.linear_model.LassoLars(copy_X=True)}
        combined_data=self.patient_protein_activity.merge(self.clinical_data,
                                                          how='inner',
                                                          left_index=True,
                                                          right_index=True)
        for (column,model) in self.clinical_prediction_models.items():
            model.fit(combined_data[self.patient_protein_activity.columns].values,
                      combined_data[column].values)
        return pandas.Series({column:model.score(combined_data[self.patient_protein_activity.columns].values,
                                                 combined_data[column].values)
                              for (column,model) in self.clinical_prediction_models.items()}).plot.barh()
    
    def visualize(self,model):
        """Plots a chart of the chosen model"""
        data=None
        coefficients=self.clinical_prediction_models[model]
        if coefficients.__class__.__name__=='LogisticRegression':
            data=pandas.DataFrame(coefficients.coef_.T,
                                  columns=coefficients.classes_[:coefficients.coef_.shape[0]],
                                  index=self.patient_protein_activity.columns)
        else:
            data=pandas.Series(coefficients.coef_,
                               index=self.patient_protein_activity.columns)
        return data.plot.bar(figsize=(6,4))


# ## Clustering protein activity
# Previous work on this dataset has involved clustering patients according to their protein activity. For this analysis, I decided to cluster proteins according to their activity in different patients, and then use the activity of the different clusters to predict clinical features. Biologically speaking, it is assumed that proteins whose activity clusters together comprise interacting systems.
# First, the dimensionality of dataset is reduced, and then a hierarchical cluster is fitted to the principal components, which is visualised as a dendrogram to select the appropriate number of components.

# In[ ]:


models=BreastCancerAnalysis()
models.fit_principal_components()
models.dendrogram()


# From this it appears that 8 components are appropriate. Therefore, hierarchical clustering is rerun and the proteins assigned to 8 clusters

# In[ ]:


models.cluster_proteins(8)


# ## Protein activity per patient
# Projecting the patient's protein activity onto the eigenvectors derived above, and then projecting those vectors into the space of the means of the clusters, the patients' activity for each cluster is calculated.
# 

# In[ ]:


models.patient_cluster_activity()


# ## Modelling clinical features
# 
# A set of generalized linear predictors (LassoLars for numerical variables, Logistic Regression for categorical variables) is then fit to the clinical data, using the protein cluster activity as the independent variables

# In[ ]:


models.train_clinical_models()


# Strong predictors can be found for several clinical factors. These are discussed below.
# 
# ## Gender

# In[ ]:


models.visualize('Gender')


# Activity of cluster 0 is greater in female patents, while activity in clusters 2, 3, 6 and 7 is greater in male patients.
# 
# ## ER Status

# In[ ]:


models.visualize('ER Status')


# Activity in clusters 2, 4 and 5 makes positive ER status more likely, while it is less likely in patients with activity in clusters 1 and 7.
# 
# ## PR Status

# In[ ]:


models.visualize('PR Status')


# The same clusters are correlated in the same way with ER status.
# 
# ## HER2 Final Status

# In[ ]:


models.visualize('HER2 Final Status')


# HER2 Final status is much more likely to be negative in patients with activity in clusters 1 and 7, and more likely to be positive in patients with activity in cluster 4.
# 
# ## Tumor

# In[ ]:


models.visualize('Tumor')


# * T1 tumors are most likely in patients with activity in clusters 4 and 5, and least likely in patients with activity in clusters 0 and 1.
# * T2 tumors are most likely in patients with activity in cluster 5.
# * T3 tumors are significantly less likely in patients with activity in cluster 0.
# * T4 tumors are most likely in patients with activity on cluster 0 and least likely in patients with activity in cluster 5.
# 
# ## Node

# In[ ]:


models.visualize('Node')


# * N0 Node is slighlty more likely in patients with activity in cluster 0, and slightly less likely in patients with activity in clusters 6 and 7.
# * N1 Node is most likely in patients with activity in cluster 4 and least likely in patients with activity in cluster 1
# * N2 Node is most likely in patients with activity in clusters 5 and 6 and least likely patients with activity in cluster 3.
# * N3 Node is most likely in patients with activity in cluster 7 and least likely in patients with activity in cluster 0.
# 
# ## Metastasis

# In[ ]:


models.visualize('Metastasis-Coded')


# Metastasis is least likely in patients with activity in cluster 4. This result should be treated with caution, as few patients in this sample exhibit metastasis.

# In[ ]:


models.visualize('AJCC Stage')


# ## Vital Status

# In[ ]:


models.visualize('Vital Status')


# Patients with activity in cluster 0 are most likely to die. Patients with activity in cluster 1 are least likely to die.

# In[ ]:


models.visualize('OS Time')


# Activity in cluster 1 predicts longer OS times
# 
# ## PAM50 mRNA

# In[ ]:


models.visualize('PAM50 mRNA')


# Luminal B PAM50 RNA is most likely in patients with activity in cluster 1, and less likely in patients with activity in clusters 2 and 4. HER2-enriched PAM50 RNA is lest likely in patients with activity in cluster 1. Basal-like PAM50 RNA is most likely in patients with activity in cluster 2.
# 
# ## RPPA Clusters

# In[ ]:


models.visualize('RPPA Clusters')


# HER2 RPPA Clusters are most likely in patients with activity in cluster 4. Lum A RPPA clusters are least likely in patients with activity in cluster 5. Lum A/B clusters are least likely in patients with activity in cluster 6.
# 
# 
