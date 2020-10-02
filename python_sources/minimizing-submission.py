#!/usr/bin/env python
# coding: utf-8

# Submision file for this competition is about 500M, so it makes sense to minimize it. The traditional way is to use "float_format" parameter of pandas "to_csv" method + compression. 
# The problem with "float_format" is that different predictions may become equal ones - this situation depends on specific model and it is hard to predict what can be impact to the score (may assume it is minimal, but who knows). 
# 
# This kernel shows how submission file can be minimized, while ensuring that the score is not changed.
# It is based on the following ideas:
# 
# 1. ROC AUC stays the same if relative position of predictions remain the same. In other words, if one sorts predictions by is_attributed and change predictions so their position in sorted list remains, then AUC is not changed.
# 2. Usually number of unique predictions are less than total number of rows in predictions (in my case unique predictions is about half  of rows)
# 3. Number of digits after decimal point can be decreased. Extra precision is not needed here, minimization procedure should guarantee only that different predictions remain different.
# 4. Leading and trailing zeros can be omitted
# 
# Potential disadvantage: after such minimization, blending may (or may not) get different result. So keep original submission.
# 
# The kernel uses "FTRL revisited 22" data as an example. Plain and compressed size for original, formatted (float_format) and minimized (this algorithm) submission files are calculated.

# In[ ]:


import numpy as np
import pandas as pd
import os
import math


def get_minimized(submission):
    """
    Minimizes size of column 'is_attributed' from submission
    :param submission: panda DataFrame
    :return: minimized column 'is_attributed' as pandas Series
    """
    unique_values = np.sort(submission.is_attributed.unique())
    size = unique_values.shape[0]
    digits = int(math.ceil(math.log10(size)))
    print('Unique size {:,}, digits: {}'.format(size, digits))
    step = 10 ** -digits
    format_string = '{:.' + str(digits) + 'f}'
    mapping = {}
    value = step
    for i in range(size):
        original = unique_values[i]
        text_value = format_string.format(value).strip('0')
        mapping[original] = text_value
        value += step

    minimized = submission['is_attributed'].map(mapping.get)
    return minimized


def save_submission(file_path, submission, compression=None, line_terminator='\r'):
    submission.to_csv(file_path, index=False, line_terminator=line_terminator, chunksize=1024, compression=compression)


# In[ ]:


sample_file_path = '../input/ftrl-revisited-22/sub_proba.csv'
minimized_file_name = 'minimized.csv'
formatted_file_name = 'formatted.csv'


submission = pd.read_csv(sample_file_path, dtype={'click_id': 'int32', 'is_attributed': 'float64'}, engine='c',
                         na_filter=False, memory_map=True)

# save file with formatting
submission.to_csv(formatted_file_name, float_format='%.8f', index=False)
formatted_size = os.path.getsize(formatted_file_name)
os.remove(formatted_file_name)

# save file with formatting with compression
formatted_file_name_gz = formatted_file_name + ".bz2"
submission.to_csv(formatted_file_name_gz, float_format='%.8f', index=False, compression='bz2')
formatted_size_gz = os.path.getsize(formatted_file_name_gz)
os.remove(formatted_file_name_gz)

minimized = get_minimized(submission)

submission['is_attributed'] = minimized

# save minimized file
save_submission(minimized_file_name, submission)
minimized_size = os.path.getsize(minimized_file_name)
os.remove(minimized_file_name)

# save minimized with compression
minimized_file_name_gz = minimized_file_name + '.bz2'
save_submission(minimized_file_name_gz, submission, compression='bz2')
minimized_size_gz = os.path.getsize(minimized_file_name_gz)
os.remove(minimized_file_name_gz)

original_size = os.path.getsize(sample_file_path)

print('original file size: {:,}'.format(original_size))
print('minimized file size: {:,}'.format(minimized_size))
print('formatted file size: {:,}'.format(formatted_size))
print('minimized + compressed file size: {:,}'.format(minimized_size_gz))
print('formatted + compressed file size: {:,}'.format(formatted_size_gz))

