#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import requests
from lxml import html


# Reference: [HTML Scraping - The Hitchhiker's Guide to Python](https://docs.python-guide.org/scenarios/scrape/)
# 
# For package descriptions, we'll request the [64-bit Linux Python package list](https://docs.anaconda.com/anaconda/packages/py3.6_linux-64/) from Anaconda. (Other distributions listed [here](https://docs.anaconda.com/anaconda/packages/pkg-docs/).)

# In[ ]:


package_list_url = 'https://docs.anaconda.com/anaconda/packages/py3.6_linux-64/'
page = requests.get(package_list_url)
tree = html.fromstring(page.content)


# As of September 2019, the package list table is embedded within the HTML with a `docutils` class name.

# In[ ]:


rows = tree.xpath('//table[@class="docutils"]//tr')

# Drop the first row containing column titles.
rows.pop(0);


# Within each row of the table, the first cell contains the package name and the third cell contains summary / license information.

# In[ ]:


package_names = [row.xpath('td[1]/a/text()')[0] for row in rows]

summaries_and_licenses = [row.xpath('td[3]/text()')[0].split(' / ') for row in rows]
summaries, licenses = zip(*summaries_and_licenses)

print(
    '{} package names, {} summaries, {} licenses'
    .format(len(package_names), len(summaries), len(licenses))
)


# We'll package this up.

# In[ ]:


columns = {
    'package_name': package_names,
    'summary': summaries,
    'license': licenses
}
anaconda_metadata = pd.DataFrame(columns)
anaconda_metadata.set_index('package_name', inplace=True)
anaconda_metadata.head()


# And write it out.

# In[ ]:


anaconda_metadata.to_csv('anaconda-package-metadata.csv', header=True)

