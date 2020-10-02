#!/usr/bin/env python
# coding: utf-8

# The zip data files are very large and therefore, it takes much time to unzip them using python or a shel script
# If you use VM, you pay for every second the VM runs and you may want to unzip the files as quaick as possible.
# 
# Here I suggest an efficient method for unzipping the full trining data. The unzipping uses all the cpu that your VM has and therefore it will take less time than unzipping the data files one by one.
# 
# Note that I assume that you have already downloaded all the zip files to your VM.
# 
# Just copy the following script to your VM and run it.
# Good luck

# In[ ]:


import os
import zipfile
from multiprocessing import Pool, cpu_count

# base_path is where the zip files are located. Change it to your location 
base_path = '/data/kaggle/deepfake/input/train/'
if not os.path.isdir(base_path):
    base_path = 'C:/Users/itama/Documents/kaggle/deepfake/input/train/'


def unzip(n):
    idx_str = str(n)
    if len(idx_str) == 1:
        idx_str = '0' + idx_str

    zip_file = base_path + 'dfdc_train_part_' + idx_str + '.zip'
    if not os.path.isfile(zip_file):
        return

    print('Unzipping dfdc_train_part_' + idx_str + '.zip')
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(base_path)
    zip_ref.close()

    os.remove(zip_file)


if __name__ == '__main__':
    n_cpu = cpu_count()
    n_cpu = min(n_cpu, 50)
    print('unzipping the files using ' + str(n_cpu) + ' processes.')
    pool = Pool(processes=n_cpu)

    pool.map(unzip, range(50), chunksize=1)

