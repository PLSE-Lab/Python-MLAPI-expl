#!/usr/bin/env python
# coding: utf-8

# This notebook combines output files form different AutoML training notebooks. I've decided to break the problem in this way becasue we effectively have five different targets, adn each regression taks can be handled differently. Furthermore, AutoML tasks are very computationally intensive, and can use up as much of the resurces - memory, compute, disk space, time - as one can get from a kernels. The individual training kernes can be found here:
# 
# age: https://www.kaggle.com/tunguz/trends-h2o-automl-age
# 
# domain1_var1: https://www.kaggle.com/tunguz/trends-h2o-automl-domain1-var1
# 
# domain1_var2: https://www.kaggle.com/tunguz/trends-h2o-automl-domain1-var2
# 
# domain2_var1: https://www.kaggle.com/tunguz/trends-h2o-automl-domain2-var1
# 
# domain2_var1: https://www.kaggle.com/tunguz/trends-h2o-automl-domain2-var2
# 
# The train and test files for the AutoML kernels have been generated in this separate kernel:
# 
# https://www.kaggle.com/tunguz/trends-train-test-creator

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Let's load up all the individual predictions

# In[ ]:


preds_age = np.load('../input/trends-h2o-automl-age/preds_age.npy')
preds_domain1_var1 = np.load('../input/trends-h2o-automl-domain1-var1/preds_domain1_var1.npy')
preds_domain1_var2 = np.load('../input/trends-h2o-automl-domain1-var2/preds_domain1_var2.npy')
preds_domain2_var1 = np.load('../input/trends-h2o-automl-domain2-var1/preds_domain2_var1.npy')
preds_domain2_var2 = np.load('../input/trends-h2o-automl-domain2-var2/preds_domain2_var2.npy')


# In[ ]:


test = pd.read_csv('../input/trends-train-test-creator/test.csv')
test.head()


# In[ ]:


preds_age


# In[ ]:


Id = test.Id.values


# In[ ]:


Id.shape


# In[ ]:


sub_df = pd.DataFrame({'Id': Id, 'age': preds_age.flatten(), 
                       'domain1_var1': preds_domain1_var1.flatten(), 'domain1_var2': preds_domain1_var2.flatten(), 
                       'domain2_var1': preds_domain2_var1.flatten(), 'domain2_var2': preds_domain2_var2.flatten()})
sub_df.head()


# In[ ]:


sub_df = pd.melt(sub_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df.head()


# In[ ]:


sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")


# In[ ]:


sub_df.head()


# In[ ]:


sub_df.to_csv("submission_h2o_automl.csv", index=False)


# In[ ]:




