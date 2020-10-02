# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def content_based_filtering():
    # products=pd.read_csv('../input/products_FMCG.csv')
    products=pd.read_csv('../input/dataset-netmeds-products-collaborative-filtering/netmeds-products.csv')
    products_name_series = products['name']
    # test_product_list = ["Tooth", "Dabur", "hair", "Nestle", "Gel"]
    test_product_list = ["Dabur Amla Hair Oil", "Vicks Vaporub 50 ml"]
    
    '''
    print(type(products_name_series))
    
    hits = []
    
    for product in products_name_series:
        if fuzz.token_sort_ratio(test_product, product) > 70:
            hits.append(product)
            
    print(hits)
    '''
    for test_product in test_product_list:
        print("products vertically related to ", test_product,"are: \n")
        print(process.extract(test_product, products_name_series), "\n")
    
if __name__ == "__main__":
    content_based_filtering()