# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files
#in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


#Start

with open('../input/cities_r2.csv') as f:

    city_data = pd.read_csv(f)
    
    print('city_data DataFrame Info:\n')
    print('Dimensions: ' + str(city_data.shape))
    print('Attribute List: ' + str(city_data.columns))
    
    print('Correlation Matrix\n')
    
    
    #print(np.corrcoef(city_data['effective_literacy_rate_female'], city_data['female_graduates']))
    
    plt.matshow(city_data.corr())