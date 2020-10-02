# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

import logging

import os




def main(log):
    log.info('Begin main()')
    
    #log.info(check_output(['ls', '../input']).decode('UTF8'))
    log.info(check_output(['ls']).decode('UTF8'))
    # Any results you write to the current directory are saved as output.
    
    print('. contents: {}'.format(os.listdir('.')))
    print('.. contents: {}'.format(os.listdir('..')))
    
    
    log.info('End main()')
#



if __name__ == '__main__':
    log = logging.getLogger('main')
    log.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    log.addHandler(ch)

    main(log)