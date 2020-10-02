#!/usr/bin/env python
# coding: utf-8

# **EXPLORATION ET TRAITEMENT DES DONNEES:**
# 1. objets galactiques et extragalactiques.(chque classe est soit galactique(redshift=0) soit extragalactique)
# 1.  affichage en histagram
# 1. valeur nan dans les attributs (distmod, et hostgal_spectz pour les donnees de test)
# 1.  les attributs correles sont: 
#                                           * hostgal_photoz et distmod.
#                                           * flux et distmod.
#                                           * hostgal_photoz et hostgal_spectz.
#                                           * passband et flux
#                                           * ra et gal_l
#                                           * dec et gal_b
#                                               
#                                               
# 
# 
# 

# 

# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


training_set = pd.read_csv('../input/training_set.csv')
meta_data = pd.read_csv('../input/training_set_metadata.csv')
test_meta_data = pd.read_csv('../input/test_set_metadata.csv')
training_set.head()


# In[ ]:


targets = np.hstack([np.unique(meta_data['target']), [99]])
target_map = {j:i for i, j in enumerate(targets)}
target_ids = [target_map[i] for i in meta_data['target']]
meta_data['target_id'] = target_ids


# In[ ]:


galactic_cut = meta_data['hostgal_specz'] == 0
plt.figure(figsize=(10, 8))
plt.hist(meta_data[galactic_cut]['target_id'], 15, (0, 15), label='Galactique')
plt.hist(meta_data[~galactic_cut]['target_id'], 15, (0, 15), label='Extragalactique')
plt.xticks(np.arange(15)+0.5,   targets)
plt.gca().set_yscale("log")
plt.xlabel('Classe')
plt.ylabel('nombre')
plt.xlim(0, 15)
plt.legend();


# attribuant a chque classe son nom tel que definie par les astronautes .

# In[ ]:



target_types={6:'Microlensing', 15:'Explosive Type V', 16:'Transits', 42:'Explosive type W', 52:'Explosive Type X', 
                  53:'Long periodic', 62:'Explosive Type Y', 64:'Near Burst', 65:'Flare', 67:'Explosive Type Z',
                  88:'AGN', 90:'SN Type U', 92:'Periodic', 95:'SN Type T'}


# In[ ]:


object_list=times.groupby('object_id').apply(lambda x: x['object_id'].unique()[0]).tolist()


# In[ ]:


colors = ['purple', 'blue', 'green', 'orange', 'red', 'black']

def plot_one_object(obj_id):
        
    for band in range(len(colors)):
        sample = train_series[(train_series['object_id'] == obj_id) & (train_series['passband']==band)]
        plt.errorbar(x=sample['mjd'],y=sample['flux'],yerr=sample['flux_err'],c = colors[band],fmt='o',alpha=0.7)


# In[ ]:


#le nombre d'objet dans chaque classe.


# In[ ]:


for t in sorted(meta_data['target'].unique()):
    print (t,meta_data[meta_data['target']== t]['target'].count(),target_types[t],meta_data[meta_data['target']== t]['hostgal_specz'].mean())


# In[ ]:


print(meta_data[meta_data.isnull().any(axis=1)][null_columns].head())


# In[ ]:


null_columns=test_meta_data.columns[test_meta_data.isnull().any()]
test_meta_data[null_columns].isnull().sum()


# In[ ]:



groups = training_set.groupby(['object_id', 'passband'])
groups


# In[ ]:


times = groups.apply(
    lambda block: block['mjd'].values).reset_index().rename(columns={0: 'seq'})
flux = groups.apply(
    lambda block: block['flux'].values
).reset_index().rename(columns={0: 'seq'})
err = groups.apply(
    lambda block: block['flux_err'].values
).reset_index().rename(columns={0: 'seq'})
det = groups.apply(
    lambda block: block['detected'].astype(bool).values
).reset_index().rename(columns={0: 'seq'})
times_list = times.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
flux_list = flux.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
err_list = err.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
det_list = det.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()


# In[ ]:


flux


# In[ ]:


object_list=times.groupby('object_id').apply(lambda x: x['object_id'].unique()[0]).tolist()


# In[ ]:


colors = ['purple', 'blue', 'green', 'orange', 'red', 'black']

def plot_one_object(obj_id):
        
    for band in range(len(colors)):
        sample = training_set[(training_set['object_id'] == obj_id) & (training_set['passband']==band)]
        plt.errorbar(x=sample['mjd'],y=sample['flux'],yerr=sample['flux_err'],c = colors[band],fmt='o',alpha=0.7)


# In[ ]:


for t in sorted(meta_data['target'].unique()):
    print (t,meta_data[meta_data['target']== t]['target'].count(),target_types[t],meta_data[meta_data['target']== t]['hostgal_specz'].mean())


# In[ ]:


def fit_kernel_length_only(times_band,flux_band,err_band):
    
    def _kernel_likelihood(length):
        sigma=siguess
        #length=params
        kernel=np.exp(-(np.reshape(times_band,(-1,1)) - times_band)**2/2/length**2)
        np.fill_diagonal(kernel,0)
        sumw=kernel.dot(1./err_band**2) + 1./sigma**2
        pred=kernel.dot(flux_band/err_band**2) / sumw
        chi2 = (pred - flux_band)**2 / ( err_band**2 + 1./sumw )
        # -2 ln likelihood
        logl=np.sum(chi2 + np.log(err_band**2 + 1./sumw))
        return logl
    
    lguess=(np.max(times_band)-np.min(times_band))/len(times_band)
    siguess=np.std(flux_band)
    output=optimize.fmin(_kernel_likelihood,lguess,disp=False,xtol=0.01,full_output=1)
    return (siguess,output[0][0]), output[1]


# In[ ]:


def kernel_predict(params,times_band,flux_band,err_band):
    sigma=params[0]
    length=params[1]
    kernel=np.exp(-(np.reshape(time_grid,(-1,1)) - times_band)**2/2/length**2)
    sumw=kernel.dot(1./err_band**2) + 1./sigma**2
    pred=kernel.dot(flux_band/err_band**2) / sumw
    return pred, np.sqrt(1./sumw)


# In[ ]:


def make_kernel(tlist,flist,elist,fit_kernel_function=fit_kernel_length_only):
    flux_grid = []
    err_grid = []
    kernel_sigma = []
    kernel_length = []
    kernel_logl=[]
    for iobj,(times_obj,flux_obj,err_obj) in enumerate(zip(tlist,flist,elist)):
        flux_grid_obj=[]
        err_grid_obj=[]
        kernel_sigma_obj = []
        kernel_length_obj = []
        kernel_logl_obj=[]
        if iobj in meta_data[meta_data['hostgal_photoz']!=0.0].index:
            for times_band,flux_band,err_band in zip(times_obj,flux_obj,err_obj):
                (sigma,length),logl = fit_kernel_function(times_band,flux_band,err_band)
                k_flux,k_err=kernel_predict((sigma,length),times_band,flux_band,err_band)
                flux_grid_obj.append(k_flux)
                err_grid_obj.append(k_err)
                kernel_sigma_obj.append(sigma)
                kernel_length_obj.append(length)
                kernel_logl_obj.append(logl)
        else:
            kernel_sigma_obj=[0]*6
            kernel_length_obj=[0]*6
            kernel_logl_obj=[0]*6
        flux_grid.append(flux_grid_obj)
        err_grid.append(err_grid_obj)
        kernel_sigma.append(kernel_sigma_obj)
        kernel_length.append(kernel_length_obj)
        kernel_logl.append(kernel_logl_obj)
    return flux_grid,err_grid, kernel_sigma, kernel_length,kernel_logl


# In[ ]:


iobj=1
band=3
time_grid=(np.arange(59550,60705,5.))
(sigma,length),logl = fit_kernel_length_only(times_list[iobj][band],flux_list[iobj][band],err_list[iobj][band])
#length=4.0
k_flux,k_err=kernel_predict((sigma,length),times_list[iobj][band],flux_list[iobj][band],err_list[iobj][band])
plt.errorbar(times_list[iobj][band],flux_list[iobj][band],yerr=err_list[iobj][band],color=colors[band],fmt='o')
plt.plot(time_grid,k_flux)
plt.fill_between(time_grid,k_flux-k_err,k_flux+k_err,alpha=0.3)
plt.ylim(np.min(flux_list[iobj][band]*1.5,0),np.max(flux_list[iobj][band]*1.5,0))
#plt.xlim(60100,60300)
print (sigma,length,logl)


# In[ ]:


klonly_flux_grid,klonly_err_grid,klonly_sigma,klonly_length,klonly_logl = make_kernel(
    times_list,flux_list,err_list,fit_kernel_function=fit_kernel_length_only)


# In[ ]:


def plot_interpolations(iobj,times_list,flux_list,err_list,flux_grid,err_grid):
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))
    plt.title(target_types[meta_data.loc[iobj,'target']]) 
    for band in range(6):
        ax = axes[band // 3, band % 3]
        ax.errorbar(times_list[iobj][band],flux_list[iobj][band],yerr=err_list[iobj][band],color=colors[band],fmt='o')
        ax.plot(time_grid,flux_grid[iobj][band],color=colors[band])
        ax.fill_between(time_grid,flux_grid[iobj][band]-err_grid[iobj][band],
                        flux_grid[iobj][band]+err_grid[iobj][band],alpha=0.3,color=colors[band])
        ax.set_xlabel('MJD')
        ax.set_ylabel('Flux')
    plt.title(target_types[meta_data.loc[iobj,'target']])
plot_interpolations(300,times_list,flux_list,err_list,klonly_flux_grid,klonly_err_grid)
#plt.ylim(-50,200)
plt.xlim(60000,60250)


# In[ ]:


for iobj in meta_data[(meta_data['ddf']==0)]['object_id'][:25]:
   plt.figure()
   plot_one_object(iobj)


# In[ ]:


x1=meta_data["hostgal_specz"].tolist()
x2=meta_data["hostgal_photoz"].tolist()
plt.scatter(x1, meta_data["distmod"],color = 'red');
plt.scatter(x2, meta_data["distmod"],color = 'blue');


# In[ ]:


true_photoz= meta_data[~meta_data.hostgal_specz.isna()==~meta_data.hostgal_photoz.isna()]
true_photoz.plot.scatter(x="hostgal_specz", y="hostgal_photoz",color = 'green');


# In[ ]:



training_set.plot.scatter(y="passband", x="flux");


# In[ ]:


meta_data.plot.scatter(x="gal_l", y="ra")


# In[ ]:


meta_data.plot.scatter(x="decl", y="gal_b")


# In[ ]:


l=len(meta_data['target_id'])


# In[ ]:


l=len(unique(training_set['object_id']))


# In[ ]:


l

