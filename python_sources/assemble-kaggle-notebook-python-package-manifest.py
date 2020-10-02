#!/usr/bin/env python
# coding: utf-8

# ## Procedure ##
# 
# 1. Gather [installed packages found using code inside a Kaggle notebook](https://www.kaggle.com/ottoschnurr/query-notebook-for-installed-packages).
# 1. Gather [package metadata from Anaconda](https://www.kaggle.com/ottoschnurr/query-anaconda-for-package-metadata).
# 1. For remaining notebook packages not identified by Anaconda, gather [package metadata from PyPI](https://www.kaggle.com/ottoschnurr/gather-pypi-package-metadata).
# 1. Collate and write the manifest
# 
# ## Reference ##
# 
# - Thanks to @nagadomi for the [original script](https://www.kaggle.com/nagadomi/list-of-installed-packages).
# - The [64-bit Linux Python package list](https://docs.anaconda.com/anaconda/packages/py3.6_linux-64/).
# - [HTML Scraping - The Hitchhiker's Guide to Python](https://docs.python-guide.org/scenarios/scrape/)
# - The [PyPI JSON API](https://warehouse.readthedocs.io/api-reference/json/).
# - [Rate Limiting for the PyPI API](https://warehouse.readthedocs.io/api-reference/#rate-limiting)

# In[ ]:


import numpy as np
import pandas as pd

package_list_url = '../input/query-pip-for-installed-packages/kaggle-notebook-packages.csv'

package_list = pd.read_csv(package_list_url)
package_list.set_index('package_name', inplace=True)
package_list.head()


# In[ ]:


anaconda_url = '../input/query-anaconda-for-package-metadata/anaconda-package-metadata.csv'

anaconda_metadata = pd.read_csv(anaconda_url)
anaconda_metadata.set_index('package_name', inplace=True)
anaconda_metadata.insert(2, 'metadata_source', 'Anaconda')
anaconda_metadata.head()


# In[ ]:


pypi_url = '../input/query-pypi-for-package-metadata/pypi-package-metadata.csv'

pypi_metadata = pd.read_csv(pypi_url)
pypi_metadata.set_index('package_name', inplace=True)
pypi_metadata.insert(2, 'metadata_source', 'PyPI')
pypi_metadata.head()


# In[ ]:


manifest = package_list     .merge(anaconda_metadata, how='left', on='package_name')     .combine_first(pypi_metadata)     [['version', 'summary', 'license', 'metadata_source']]

print(manifest.info())

manifest.head()


# In[ ]:


manifest.to_csv('package-manifest.csv', header=True)

