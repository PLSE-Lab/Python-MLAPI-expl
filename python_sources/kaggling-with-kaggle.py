#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook shows how to take a template kernel and make a bunch of parallel runs with slightly different hyper-parameters. In order for a parameter to be identified as a _hyperparameter_ it needs to be written in the code field in the following form
# ```py
# LEARNING_RATE=0.1
# EPOCHS=5
# MODEL='VGG16'
# ```
# The notebook shows how we can automatically extract those parameters, create a series of runs based on them and submit all of the runs to Kaggle as Kernels

# # Setup the Environment
# Here we setup the variables for our kaggle account
# - you need a USER_ID and USER_SECRET which you can get by following the instructions here: https://github.com/Kaggle/kaggle-api#api-credentials
# - the credentials below have already been invalidated and so you cannot use them

# In[ ]:


USER_ID = 'kevinbot'
USER_SECRET = '94fdb3f3cb764ee6f414374909d7ca34'


# In[ ]:


import os, json, nbformat, pandas as pd
kaggle_conf_dir = os.path.join(os.path.expandvars('$HOME'), '.kaggle')
os.makedirs(kaggle_conf_dir, exist_ok = True)
with open(os.path.join(kaggle_conf_dir, 'kaggle.json'), 'w') as f:
    json.dump({'username': USER_ID, 'key': USER_SECRET}, f)
get_ipython().system('chmod 600 {kaggle_conf_dir}/kaggle.json')


# # Notebook Metadata
# Here we make the notebook metadata template for submitting new notebooks. Basically it has what datasets we want to include, if the notebook should be private, if we want GPU enabled and so forth

# In[ ]:


notebook_meta_template = lambda user_id, title, file_id, nb_path: {'id': f'{user_id}/{file_id}',
 'title': f'{title}',
 'code_file': nb_path,
 'language': 'python',
 'kernel_type': 'notebook',
 'is_private': False, # probably better to make them private but for the demo notebook it is useful to see them
 'enable_gpu': True,
 'enable_internet': False,
 'keywords': [],
 'dataset_sources': ['gaborfodor/keras-pretrained-models', 'kmader/food41'],
 'kernel_sources': [],
 'competition_sources': []}


# # Download a Template Notebook/Kernel
# Here we use Use the "Hot Dog not Hot Dog" Kernel as a Basis. In order to be a good kernel, the file should have a number of simple lines like the ones below that can be changed through the script below
# ```py
# LEARNING_RATE=0.1
# EPOCHS=5
# MODEL='VGG16'
# ```

# In[ ]:


base_dir = os.path.join('.', 'base_kernel')
os.makedirs(base_dir, exist_ok = True)
kernel_path = os.path.join(base_dir ,'hot-dog-not-hot-dog-gpu.ipynb')
if not os.path.exists(kernel_path):
    get_ipython().system('kaggle kernels pull -k kmader/hot-dog-not-hot-dog-gpu -p {base_dir}')


# ## Parse the Notebook
# Here we parse the notebook looking for parameters to play with, we use pandas to show a bit what is inside.

# In[ ]:


kernel_data = nbformat.read(kernel_path, as_version=4)
cell_df = pd.DataFrame(kernel_data['cells'])
cell_df.query('cell_type=="code"')


# ## Use Abstract Syntax Tree
# We can use the abstract syntax tree to find relevant code that we can change to run notebooks with new settings

# In[ ]:


import ast
all_asgn = []
for cell_idx, c_cell in enumerate(kernel_data['cells']):
    if c_cell['cell_type']=='code':
        c_src = c_cell['source']
        # remove jupyter things
        c_src = '\n'.join(['' if (c_block.strip().startswith('!') or 
                                  c_block.strip().startswith('%')) else
                           c_block
                           for c_block in c_src.split('\n')])
        
        for c_statement in ast.parse(c_src).body:
            if isinstance(c_statement, ast.Assign):
                # only keep named arguments that are not assigned from function calls
                if all([isinstance(c_targ, ast.Name) 
                        for c_targ in c_statement.targets]) and not (isinstance(c_statement.value, ast.Call) or 
                                                                     isinstance(c_statement.value, ast.Lambda)) and len(c_statement.targets)==1:
                    
                    all_asgn += [{'cell_id': cell_idx,
                                  'line_no': c_statement.lineno,
                                  'line_code': c_src.split('\n')[c_statement.lineno-1],
                                  #'value': c_statement.value,
                                  'target':  c_statement.targets[0].id}
                                  ]
assignment_df = pd.DataFrame(all_asgn)
assignment_df['line_replacement'] = assignment_df['line_code'] 
assignment_df


# # Make our batches
# Here we can make the batches of code to run. Each batch has a parameter data.frame associated it with that we write into the first block in the notebook.
# 
# We use the product function to perform a grid search over all the possibilities

# In[ ]:


from itertools import product
batch_dict = {'IMG_SIZE': [(139, 139), (299, 299), (384, 384), (512, 512)],
             'use_attention': [False, True]}
batch_keys = list(batch_dict.keys())
batches = []
for c_vec in product(*[batch_dict[k] 
                       for k in batch_keys]):
    cur_df = assignment_df.copy()
    sub_lines = dict(zip(batch_keys, c_vec))
    print(sub_lines)
    for c_key, c_value in sub_lines.items():
        cur_df.loc[cur_df['target']==c_key, 'line_replacement'] = cur_df[cur_df['target']==c_key]['line_code'].map(lambda x: '{}= {}'.format(
            x.split('=')[0],
            c_value))
    batches+=[(sub_lines, cur_df)]


# ## Replace the lines in the notebook
# The code here surgically replaces just the necessary lines in the notebook and leaves (hopefully) everything else exactly the way it is

# In[ ]:


import copy
def replace_line(in_code, in_line_idx, in_replacement):
    return '\n'.join([j if i!=in_line_idx else in_replacement for i, j in enumerate(in_code.split('\n'), 1)])
def apply_replacement_df(in_nb, rep_df):
    cur_nb = copy.deepcopy(in_nb)
    for _, c_row in rep_df.iterrows():
        if c_row['line_code']!=c_row['line_replacement']:
            # lines to fix
            cell_idx = c_row['cell_id']
            cur_nb['cells'][cell_idx]['source'] = replace_line(cur_nb['cells'][cell_idx]['source'], c_row['line_no'], c_row['line_replacement'])
    return cur_nb


# # Add the relevant information
# So we want to add a first field with all the info about the current run so we can harvest it later

# In[ ]:


from nbformat import v4 as nbf
import json
from time import time
import hashlib
run_start_time = time()
run_id = hashlib.md5('{:2.2f}-{}'.format(run_start_time, kernel_data).encode('ascii')).hexdigest()


# # Launch the kernels
# Here we use the Kaggle API to launch the kernels with the different settings

# In[ ]:


launched_kernels_list = []
kernel_id_list = []
cur_nb = nbformat.read(kernel_path, as_version = 4)
for i, (sub_lines, cur_df) in enumerate(batches):
    out_name = '{}-{:04d}'.format(run_id, i)
    out_kernel_path = '{}.ipynb'.format(out_name)
    new_nb = apply_replacement_df(kernel_data, cur_df)
    # append cells containing useful metadata we might need later
    last_cells = [nbf.new_markdown_cell('# Notebook Settings\nThe last cell is just for metadata settings that will be read out later.')]
    last_cells += [nbf.new_markdown_cell(json.dumps({'run_id': run_id,
                                      'run_time': run_start_time,
                                      'run_settings': sub_lines,
                                                     'run_df': list(cur_df.T.to_dict().values())
                                                    }))]
    new_nb['cells']+=last_cells
    nbformat.write(new_nb, out_kernel_path)
    with open('kernel-metadata.json', 'w') as f:
        meta_dict = notebook_meta_template(USER_ID, 
                           out_name, 
                           out_name, 
                           out_kernel_path)
        json.dump(meta_dict, f)
    out_str = get_ipython().getoutput('kaggle kernels push -p .')
    kernel_id_list += [meta_dict['id']]
    launched_kernels_list += [out_str] 


# In[ ]:


for c_line in launched_kernels_list:
    print(c_line[0])


# # Status
# We can check the status like so if we want to follow up on the kernels

# In[ ]:


get_ipython().system("kaggle kernels status -k {meta_dict['id']}")


# In[ ]:




