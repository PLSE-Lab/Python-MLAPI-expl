#!/usr/bin/env python
# coding: utf-8

# ## Finding papers of interest through MeSH ids
# 
# The JSON files in this dataset present a high number of MeSH extractions. Each document is so identified, globally, by all the MeSH concepts that occur into it and are judged relevant. 
# 
# This notebook shows how to navigate the dataset to easily find documents that contain certain entries of interest. 

# ## Get acquainted with the dataset
# 
# You can skip the next cell if you are familiar with the first notebook, [CORD-19_ExpertSystem_MeSH: how to](http://www.kaggle.com/expsys/cord-19-expertsystem-mesh-how-to).
# 
# First things first: all the dataset files are in '/kaggle/input'. Here's a snippet in case you want to have a look at the contents of the subfolders:

# In[ ]:


# Input data files are available in the "../input/" directory.

import os
for dirname, _, filenames in os.walk('/kaggle/input/cord19-expertsystem-mesh/cord19_expertsystem_mesh_060320'):
    if 'json' in dirname:
        print(f"{'/'.join(dirname.split('/')[-2:])} has {len(filenames)} files")
        #uncomment to list all files under the input directory
        #for filename in filenames:  
            #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Let's first import the modules we are going to need and define the functions that will carry out the filtering.

# In[ ]:


import logging
import json
from pathlib import Path
from pprint import pprint

logger = logging.getLogger(__name__)

logging.basicConfig(format="%(asctime)s %(levelname)s [PID: %(process)d - %(filename)s %(funcName)s] - %(message)s",
                    level=logging.INFO)
logger.info('start')


# In[ ]:


# local functions
def load_taxonomies(dataset_root):
    """Given a path with the root directory, we navigate the path and read all the files having .json as extension.
    Filename by filename, we build up a dictionary in the form {key : value, key1 : value1, ...} in which the key is the name of an entry in MeSH taxonomy, while the value is a set of paper ids for all the papers that contain that entry"""
    tax_in_files = {}

    for json_file in dataset_root.glob('**/*'):
        if json_file.is_dir():
            continue
        if not json_file.name.lower().endswith('.json'):
            logger.warning(f'unused file {json_file.name}')
            continue
        json_data = json.load(json_file.open())
        for tax_entry in json_data.get('MeSH_tax', []):
            tax = tax_entry['tax']
            tax_in_files[tax] = tax_in_files.get(tax, set())
            tax_in_files[tax].add(json_data['paper_id'])
    return tax_in_files

def filter_data(tax_in_files, taxonomy_entry):
    """Takes as input the dictionary built in load_taxonomies, tax_in_files, and a specific taxonomy entry.
    Returns a new dictionary tax_in_files in which only the entries that have documents in common with the latest taxonomy entry are maintained. 
    In other words, the function can be called progressively on multiple taxonomies entries, so to filter the dictionary and have, in the end, only the sets of documents that share the entries all of interest."""
    if taxonomy_entry not in tax_in_files:
        logger.error(f"In the current set of entries there are no documents with tax = {taxonomy_entry}")
        return None
    selected_set = tax_in_files[taxonomy_entry]
    return {
        taxonomy_entry: docs & selected_set
        for taxonomy_entry, docs in tax_in_files.items()
        if docs & selected_set
    }

def dump_most_common_taxes(tax_in_files, rank=10):
    """Given the output of load_taxonomies(), the fuction returns the top entries (up to "rank") of the taxonomy, sorted decreasingly with respect to the number of documents they appear in."""
    tax_in_files_counter = [(taxonomy_entry, len(set_files))
                            for taxonomy_entry, set_files in tax_in_files.items()]
    return sorted(tax_in_files_counter, key=lambda x: x[1], reverse=True)[:rank]


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Input data files are available in the "../input/" directory.\n# Let\'s load the latest version of the data.\ndataset_root = Path(\'/kaggle/input/cord19-expertsystem-mesh/cord19_expertsystem_mesh_060320/\')\ntax_in_files = load_taxonomies(dataset_root)\n\n# for instance, let\'s filter documents by only selecting those that show the following selected_tax, about Vaccines\nselected_tax = \'/MeSH Taxonomy/Chemicals and Drugs/Complex Mixtures/Biological Products/Vaccines\'\nselection_step = filter_data(tax_in_files, selected_tax)\npprint(dump_most_common_taxes(selection_step, 10))\nprint()')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Let's progressively select a subset of categories and find all the documents that contain all the following categories\n\nrequested_tax_entries = [\n    '/MeSH Taxonomy/Organisms/Viruses/RNA Viruses/Nidovirales/Coronaviridae/Coronavirus',\n    '/MeSH Taxonomy/Phenomena and Processes/Immune System Phenomena/Immunity', \n    '/MeSH Taxonomy/Phenomena and Processes/Microbiological Phenomena/Virulence',\n    '/MeSH Taxonomy/Health Care/Environment and Public Health/Public Health/Disease Outbreaks/Epidemics/Pandemics',\n    '/MeSH Taxonomy/Phenomena and Processes/Physiological Phenomena/Virus Shedding',\n    '/MeSH Taxonomy/Organisms/Viruses/DNA Viruses/Herpesviridae/Alphaherpesvirinae/Simplexvirus'] \n\nfor selected_tax in requested_tax_entries:\n    selection_step = filter_data(selection_step, selected_tax)\n    pprint(dump_most_common_taxes(selection_step, 10))\n    print()")


# Finally, let's print out the list of the documents matching the filter criteria.

# In[ ]:


selected_docs = set()
for _, docs_with_tax in selection_step.items():
    selected_docs |= docs_with_tax

pprint(selected_docs)


# All done!
