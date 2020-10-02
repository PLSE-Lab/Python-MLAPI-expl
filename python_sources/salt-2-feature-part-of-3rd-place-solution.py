#!/usr/bin/env python
# coding: utf-8

# This notebook shows basic process of salt-2 template feature described [here](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75222). See also [official docs](https://sncosmo.readthedocs.io/en/v1.6.x/examples/plot_lc_fit.html) if you want to know in detail.

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().system('pip install iminuit')


# In[ ]:


import sncosmo
import pandas as pd
import time
from astropy.table import Table
from astropy import wcs, units as u
from sncosmo.bandpasses import read_bandpass
from contextlib import contextmanager
get_ipython().run_line_magic('matplotlib', 'inline')

@contextmanager
def timer(name):
    s = time.time()
    yield
    
    print('[{}] {}'.format(time.time() - s, name))

with timer('load data'):
    lc = pd.read_csv('../input/training_set.csv', nrows=10000)
    meta = pd.read_csv('../input/training_set_metadata.csv')
    meta.set_index('object_id', inplace=True)

# only use data with signal-to-noise ratio (flux / flux_err) greater than this value
minsnr = 3


# ## Specify a type of template
# At first, you need to specify a source template (which type of light-curve you want to fit). There are a lot of available source templates (you can see a full list in [source](https://github.com/sncosmo/sncosmo/blob/master/sncosmo/builtins.py)). In this notebook, I choose salt2-extended template.

# In[ ]:


# template to use
model_type = 'salt2-extended'
model = sncosmo.Model(source=model_type)


# ## Preprocessing
# You also need to specify passband to tell sncosmo about observation wavelength. sncosmo already has passband for lsst, so it's process looks quite simple :)

# In[ ]:



passbands = ['lsstu','lsstg','lsstr','lssti','lsstz','lssty']
with timer('prep'):
    lc['band'] = lc['passband'].apply(lambda x: passbands[x])
    lc['zpsys'] = 'ab'
    lc['zp'] = 25.0
    


# ## FItting the light curve
# Ok, let's start fitting process. 

# In[ ]:


object_id = 1598

data = Table.from_pandas(lc[lc.object_id == object_id])

photoz = meta.loc[object_id, 'hostgal_photoz']
photoz_err = meta.loc[object_id, 'hostgal_photoz_err']

# run the fit
with timer('fit_lc'):
    result, fitted_model = sncosmo.fit_lc(
        data, model,
        model.param_names,
        # sometimes constant bound ('z':(0,1.4)) gives better result, so trying both seems better
        bounds={'z':(max(1e-8,photoz-photoz_err), photoz+photoz_err)},
        minsnr=minsnr)  # bounds on parameters

sncosmo.plot_lc(data, model=fitted_model, errors=result.errors, xfigsize=10)


print('chisq:{}'.format(result.chisq))
print('hostgal_photoz: {}, hostgal_specz: {}, estimated z by the model: {}'.format(meta.loc[object_id,'hostgal_photoz'],
                                                                                   meta.loc[object_id,'hostgal_specz'],
                                                                                   result.parameters[0]))


# You can use estimated parameters and chi-sq value as feature. Be careful that it takes a long time (1sec/object) and don't forget to catch exception when you try it yourself. I also attached all template features on my discussion ([link](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75222)), so just download and use it if you don't want to wait. Enjoy.

# In[ ]:


df = pd.DataFrame(columns=['chisq'] + model.param_names)
df.index.name = 'object_id'
df.loc[object_id] = [result.chisq] + list(result.parameters)
df

