#!/usr/bin/env python
# coding: utf-8

# # Demo 2M Gene Search
# 
# Compare to https://oncoscape.v3.sttrcancer.org/atlas.gs.washington.edu.mouse.rna/landing

# In[ ]:


from clustergrammer2 import net
from sc_data_lake import SC_Data_Lake
# base_dir = '../data/primary_data/big_data/cao_2million-cell_2019_parquet_files/'
base_dir = '../input/cao-2millioncell-2019-parquet-files/cao_2million-cell_2019_parquet_files/cao_2million-cell_2019_parquet_files/'
lake = SC_Data_Lake(base_dir)
from dask.distributed import Client, progress
client = Client()
client


# ## Search for Genes

# In[ ]:


search_gene_list = ['Asap1', 'Malat1']


# ## Add Categories to Search Results

# In[ ]:


lake.add_cats(['Main_cell_type', 'Main_trajectory', 'development_stage'])


# # Run Gene Search

# In[ ]:


get_ipython().run_cell_magic('time', '', "df_gene = lake.dask_get_genes_from_all_samples(gene_list=search_gene_list)\nprint('\\nGene Search Results:')\nprint('-------------------------')\nprint(df_gene.shape)")


# #### Convert to multiindex 

# In[ ]:


df_gene_mi = net.row_tuple_to_multiindex(df_gene)


# # Gene Level Across Cell Types

# In[ ]:


inst_gene = 'Malat1'
inst_level = 'Main_cell_type'
ser_mean = df_gene_mi[inst_gene].to_dense().unstack(level=inst_level).mean().sort_values(ascending=False)
ser_mean.plot('bar', figsize=(15,7))


# In[ ]:


inst_gene = 'Asap1'
inst_level = 'Main_cell_type'
ser_mean = df_gene_mi[inst_gene].to_dense().unstack(level=inst_level).mean().sort_values(ascending=False)
ser_mean.plot('bar', figsize=(15,7), title=inst_gene + ' expression across ' + inst_level)


# In[ ]:




